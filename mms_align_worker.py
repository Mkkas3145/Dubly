"""One-stage MMS-FA worker. Only supplied text is aligned; no ASR decoding."""
from pathlib import Path
from math import gcd
import json,sys,time
import numpy as np
import soundfile as sf
from core import ROOT,atomic_json
from mms_text import text_units

MODEL='Meta/MMS-FA'
VERSION=8

def emission_windows(samples,core_frames=750,context_frames=50):
    # MMS convolution stride=320, receptive field=400 at 16 kHz.
    frames=max(1,(samples-400)//320+1)
    for start in range(0,frames,core_frames):
        end=min(frames,start+core_frames)
        yield start,end,max(0,start-context_frames),min(frames,end+context_frames)


def run(request,output):
    import torch,torchaudio
    from scipy.signal import resample_poly
    torch.set_num_threads(4)
    data=json.loads(Path(request).read_text('utf8'))
    progress=Path(output).with_suffix('.progress.json')
    atomic_json(progress,{'done':0,'total':len(data['jobs']),'phase':'load'})
    started=time.perf_counter();bundle=torchaudio.pipelines.MMS_FA
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=bundle.get_model(with_star=True,dl_kwargs={'model_dir':str(ROOT/'models/mms-forced-aligner')}).to(device).eval()
    tokenizer=bundle.get_tokenizer();aligner=bundle.get_aligner()
    load_seconds=time.perf_counter()-started
    peak=0;results=[]
    with torch.inference_mode():
        for i,job in enumerate(data['jobs']):
            atomic_json(progress,{'done':i,'total':len(data['jobs']),'phase':'align'})
            cache=Path(job['cache'])
            if cache.exists():
                old=json.loads(cache.read_text('utf8'))
                if old.get('identity')==job['identity']:
                    results.append(old);continue
            started=time.perf_counter()
            utterances=[text_units(text,job.get('language',data['language'])) for text in job['texts']]
            units=[unit for group in utterances for unit in group]
            row_ranges=[];cursor=0
            for group in utterances:
                row_ranges.append((cursor,cursor+len(group)));cursor+=len(group)
            if not units:raise ValueError('No lexical source tokens to align')
            pieces=[]
            with sf.SoundFile(job['audio']) as audio:
                sr=audio.samplerate;duration=len(audio)/sr
                sample_count=round(duration*16000)
                windows=list(emission_windows(sample_count))
                if device=='cuda':torch.cuda.reset_peak_memory_stats()
                for w,(start,end,left,right) in enumerate(windows):
                    atomic_json(progress,{'done':w,'total':len(windows),'phase':'acoustics'})
                    native_lo=round(left*320/16000*sr)
                    native_hi=round(((right-1)*320+400)/16000*sr)
                    audio.seek(native_lo)
                    values=audio.read(native_hi-native_lo,dtype='float32',always_2d=True)
                    mono=values[:,int(np.argmax(np.mean(values**2,axis=0)))]
                    divisor=gcd(sr,16000)
                    mono=resample_poly(mono,16000//divisor,sr//divisor).astype(np.float32)
                    expected=(right-left-1)*320+400
                    mono=np.pad(mono[:expected],(0,max(0,expected-len(mono))))
                    waveform=torch.from_numpy(mono).unsqueeze(0).to(device)
                    emission,_=model(waveform)
                    pieces.append(emission[0,start-left:end-left].float().cpu())
                    del emission,waveform
            emissions=torch.cat(pieces,dim=0);del pieces
            atomic_json(progress,{'done':len(windows),'total':len(windows),'phase':'ctc'})
            # The CTC path sees the ENTIRE original transcript and recording.
            # Only acoustic inference is chunked. Gemini times never crop audio.
            # CPU alignment avoids retaining a recording-sized GPU trellis.
            # A trailing wildcard consumes untranscribed credits/music/events;
            # it must not stretch the last phoneme across the remaining file.
            tokenized=tokenizer([u['roman'] for u in units])
            flat=[token for word in tokenized for token in word]
            minimum_frames=len(flat)+sum(a==b for a,b in zip(flat,flat[1:]))
            trailing_star=len(emissions)>minimum_frames
            spans=aligner(emissions,[*tokenized,*([[bundle.get_dict()['*']]] if trailing_star else [])])
            if trailing_star:spans=spans[:-1]
            from mms_alignment_quality import refine
            words,quality_review=refine(units,spans,tokenized,emissions,aligner,bundle.get_dict()['*'],duration,
                lambda done,total:atomic_json(progress,{'done':done,'total':total,'phase':'quality'}))
            # Recompute acoustics on bounded uncertain spans rather than asking
            # the same CTC emissions to resolve a mismatch they cannot represent.
            def local_acoustics(left,right,view='clean'):
                with sf.SoundFile(job.get('original_audio',job['audio']) if view=='original' else job['audio']) as source:
                    local_sr=source.samplerate
                    native_lo=round(left*320/16000*local_sr)
                    native_hi=round(((right-1)*320+400)/16000*local_sr)
                    source.seek(native_lo)
                    values=source.read(native_hi-native_lo,dtype='float32',always_2d=True)
                mono=values[:,int(np.argmax(np.mean(values**2,axis=0)))]
                divisor=gcd(local_sr,16000)
                mono=resample_poly(mono,16000//divisor,local_sr//divisor).astype(np.float32)
                expected=(right-left-1)*320+400
                mono=np.pad(mono[:expected],(0,max(0,expected-len(mono))))
                local,_=model(torch.from_numpy(mono).unsqueeze(0).to(device))
                return local[0,:right-left].float().cpu()
            from mms_alignment_quality import refresh_uncertain
            words,local_review=refresh_uncertain(words,units,tokenized,row_ranges,local_acoustics,aligner,
                bundle.get_dict()['*'],duration,
                lambda done,total:atomic_json(progress,{'done':done,'total':total,'phase':'quality'}))
            quality_review['local_acoustics']=local_review
            from mms_alignment_search import search
            from mms_alignment_signal import profile,contrast
            try:levels=profile(job['audio'])
            except (OSError,ValueError,RuntimeError):levels=None
            words,search_review=search(words,units,tokenized,row_ranges,local_acoustics,aligner,
                bundle.get_dict()['*'],duration,views=('clean','original') if job.get('original_audio') else ('clean',),
                progress=lambda done,total:atomic_json(progress,{'done':done,'total':total,'phase':'quality'}),
                acoustic_support=(lambda start,end:contrast(levels,start,end)) if levels is not None else None)
            quality_review['candidate_search']=search_review
            peak=max(peak,torch.cuda.max_memory_allocated() if device=='cuda' else 0)
            result={'words':words,'quality_review':quality_review,'identity':job['identity'],'model':MODEL,'device':device,
                    'seconds':time.perf_counter()-started,'peak_cuda_bytes':peak}
            atomic_json(cache,result);results.append(result)
            del emissions
    atomic_json(output,{'results':results,'model_load_seconds':load_seconds,'model_loads':1,
                        'peak_cuda_bytes':peak,'device':device})
    atomic_json(progress,{'done':len(results),'total':len(results),'phase':'complete'})

if __name__=='__main__':run(*sys.argv[1:])
