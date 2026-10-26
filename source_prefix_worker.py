"""Local audio/text alignment proposals, in the existing isolated Qwen runtime."""
import json,sys
from pathlib import Path
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from core import atomic_json

MODEL='Qwen/Qwen3-ForcedAligner-0.6B'
REVISION='c7cbfc2048c462b0d63a45797104fc9db3ad62b7'


def run(source,request,output):
    import torch
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ForcedAligner
    from faster_whisper.vad import get_vad_model
    data=json.loads(Path(request).read_text(encoding='utf8'))
    progress_file=Path(output).with_suffix('.progress.json')
    atomic_json(progress_file,dict(done=0,total=len(data['jobs']),phase='정렬 모델 로드'))
    model=Qwen3ForcedAligner.from_pretrained(
        snapshot_download(MODEL,revision=REVISION,local_files_only=True),
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='cuda:0' if torch.cuda.is_available() else 'cpu',attn_implementation='sdpa')
    result=[];prepared=[]
    with sf.SoundFile(source) as audio:
        sr=audio.samplerate;divisor=gcd(sr,16000)
        for job in data['jobs']:
            lo=round(job['context_start']*sr);hi=round(job['context_end']*sr)
            audio.seek(lo);values=audio.read(hi-lo,dtype='float32',always_2d=True)
            # Keep the stronger channel instead of cancelling opposite-phase speech.
            mono=values[:,int(np.argmax(np.mean(values**2,axis=0)))]
            mono=resample_poly(mono,16000//divisor,sr//divisor).astype(np.float32)
            prepared.append((job,lo,sr,mono))
    # One GPU forward handles several independent clips. Loading is once per
    # stage, and model autograd is disabled for inference.
    for batch_start in range(0,len(prepared),4):
        batch=prepared[batch_start:batch_start+4]
        atomic_json(progress_file,dict(done=batch_start,total=len(prepared),phase='GPU 묶음 추론'))
        try:
            with torch.inference_mode():
                aligned_batch=model.align(audio=[(x,16000) for _,_,_,x in batch],
                    text=[j['text'] for j,_,_,_ in batch],language=data['language'])
        except (ValueError,RuntimeError,IndexError) as error:
            result.extend({'source_index':j['source_index'],'error':type(error).__name__+': '+str(error)} for j,_,_,_ in batch);continue
        retries=[];proposals=[]
        for (job,lo,sr,mono),aligned in zip(batch,aligned_batch):
            try:
                words=[{'word':w.text,'start':float(w.start_time)+lo/sr,
                    'end':float(w.end_time)+lo/sr} for w in aligned]
                vad=get_vad_model()(np.pad(mono,(0,(-len(mono))%512))).reshape(-1)
                alternatives=[]
                # Search complete utterance alignments from quiet gaps nearest
                # the known ASR body. Do not force the target onto the first
                # voice in a wide context. The client validates the body too.
                from reference_edges import spans
                frame=160
                framed=np.pad(mono,(0,(-len(mono))%frame)).reshape(-1,frame)
                energy=np.sqrt(np.mean(framed**2,axis=1))
                body=energy[max(0,round((job['asr_start']-lo/sr)/.01)):]
                threshold=max(1e-6,float(np.percentile(body,90))*.04) if len(body) else 1e-6
                cuts=[(a+b)*.005 for a,b in spans(energy<threshold)
                    if (b-a)*.01>=.12 and b*.01<job['asr_start']-lo/sr]
                for cut in ([] if job.get('skip_context_retries') else sorted(cuts,reverse=True)[:6]):
                    retry_lo=round(cut*16000)
                    if retry_lo<1280 or len(mono)-retry_lo<1600:continue
                    retries.append((len(proposals),job,mono[retry_lo:],lo/sr+retry_lo/16000))
                proposals.append({'source_index':job['source_index'],'words':words,
                    'alternatives':alternatives,'vad_start':lo/sr,'vad_step':512/16000,'vad':vad.tolist()})
            except (ValueError,RuntimeError,IndexError) as error:
                result.append({'source_index':job['source_index'],'error':type(error).__name__+': '+str(error)})
        for start in range(0,len(retries),4):
            retry_batch=retries[start:start+4]
            try:
                with torch.inference_mode():
                    aligned_retries=model.align(audio=[(x,16000) for _,_,x,_ in retry_batch],
                        text=[j['text'] for _,j,_,_ in retry_batch],language=data['language'])
                for (index,job,mono,offset),aligned in zip(retry_batch,aligned_retries):
                    proposals[index]['alternatives'].append([{'word':w.text,'start':float(w.start_time)+offset,
                        'end':float(w.end_time)+offset} for w in aligned])
            except (ValueError,RuntimeError,IndexError):pass
        result.extend(proposals)
        atomic_json(progress_file,dict(done=min(len(prepared),batch_start+len(batch)),total=len(prepared),phase='정렬 완료'))
    atomic_json(output,{'model':MODEL,'revision':REVISION,'proposals':result})


if __name__=='__main__':run(*sys.argv[1:])
