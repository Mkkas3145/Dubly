"""Repair oversized ASR word windows using speech detection and quiet-edge evidence."""
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

POLICY={'version':1,'minimum_word_window_seconds':1.0,'minimum_excess_seconds':.5,
    'vad':'bundled_silero_v6_sensitive_no_minimum_speech_duration',
    'threshold':.25,'negative_threshold':.12,'speech_pad_ms':100,
    'acoustic_guard_seconds':.06,'speech_hint_guard_seconds':.15,'weak_edge_search_seconds':.25,
    'rule':'outer_edges_only_union_of_speech_and_acoustic_support_keep_all_interior_pauses',
    'failure':'preserve_asr_word_and_continue','language_specific_rules':False}


def speech_hints(values,rate):
    from faster_whisper.vad import get_speech_timestamps,VadOptions
    channel=int(np.argmax(np.mean(values.astype('float64')**2,axis=0)))
    mono=values[:,channel]
    divisor=math.gcd(rate,16000)
    signal=resample_poly(mono,16000//divisor,rate//divisor).astype('float32') if rate!=16000 else mono
    hints=get_speech_timestamps(signal,VadOptions(threshold=POLICY['threshold'],neg_threshold=POLICY['negative_threshold'],
        min_speech_duration_ms=0,min_silence_duration_ms=100,speech_pad_ms=POLICY['speech_pad_ms']))
    return [(h['start']/16000,h['end']/16000) for h in hints]


def measured_edges(values,rate,hints):
    if not hints or not len(values):return None
    hop=max(1,round(rate*.01));padded=np.pad(values,((0,(-len(values))%hop),(0,0)))
    levels=np.sqrt(np.mean(padded.reshape(-1,hop,values.shape[1])**2,axis=1)).max(axis=1)
    peak=float(levels.max())
    if not math.isfinite(peak) or peak<=1e-8:return None
    floor=float(np.percentile(levels,20))
    threshold=max(peak*.012,min(floor*3,peak*.1),1e-8)
    active=np.flatnonzero(levels>threshold)
    if not len(active):return None
    duration=len(values)/rate
    # Search acoustic edges around the detected FIRST/LAST voice, rather
    # than letting an unrelated click several seconds earlier become onset.
    # The padded VAD spans and extra guard retain weak consonants and breath.
    times=active*hop/rate
    margin=POLICY['weak_edge_search_seconds']
    leading=active[(times>=hints[0][0]-margin)&(times<=hints[0][0]+margin)]
    trailing=active[(times>=hints[-1][1]-margin)&(times<=hints[-1][1]+margin)]
    start=hints[0][0]-POLICY['speech_hint_guard_seconds']
    end=hints[-1][1]+POLICY['speech_hint_guard_seconds']
    if len(leading):start=min(start,leading[0]*hop/rate-POLICY['acoustic_guard_seconds'])
    if len(trailing):end=max(end,(trailing[-1]+1)*hop/rate+POLICY['acoustic_guard_seconds'])
    return max(0.,float(start)),min(duration,float(end))


def repair(clean,words,*,detector=None):
    repaired=[dict(w) for w in words];audit=[];detector=detector or speech_hints
    disabled=False
    with sf.SoundFile(clean) as audio:
        for i,word in enumerate(repaired):
            a,b=word['start'],word['end']
            if b-a<POLICY['minimum_word_window_seconds']:continue
            if disabled:continue
            try:
                first=max(0,round(a*audio.samplerate));last=min(len(audio),round(b*audio.samplerate))
                audio.seek(first);values=audio.read(max(0,last-first),dtype='float32',always_2d=True)
                hints=detector(values,audio.samplerate)
                edges=measured_edges(values,audio.samplerate,hints)
                if edges is None:continue
                start,end=first/audio.samplerate+edges[0],first/audio.samplerate+edges[1]
                new_a=start if start-a>=POLICY['minimum_excess_seconds'] else a
                new_b=end if b-end>=POLICY['minimum_excess_seconds'] else b
                if new_a>=new_b or (new_a==a and new_b==b):continue
                word.update(start=new_a,end=new_b,asr_start=a,asr_end=b)
                audit.append({'word_index':i,'method':'vad_and_acoustic_supported_outer_edge_repair',
                    'original_start':a,'original_end':b,'start':new_a,'end':new_b})
            except Exception as error:
                audit.append({'word_index':i,'method':'preserve_asr_word_after_edge_detector_failure',
                    'reason':f'{type(error).__name__}: {str(error)[:200]}'})
                disabled=True
    return repaired,audit
