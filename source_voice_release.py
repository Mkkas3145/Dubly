"""Recover audible releases after MMS lexical ends, without fixed padding.

Only actual separated audio is copied. Every extension needs a measured quiet
boundary; unsupported/continuous tails retain the lexical end with diagnostics.
"""
import copy,math
import numpy as np
import soundfile as sf
from reference_edges import acoustic_boundaries

VERSION=1
POLICY={'version':VERSION,'method':'last_word_acoustic_release_to_sustained_quiet',
    'lookahead_seconds':[.25,.5,1.,2.,4.,8.],'neighbor_boundary':'next_utterance_start',
    'fixed_padding':False,'source_audio_unchanged':True,'pitch_correction':False}


def levels(values,rate):
    hop=max(1,round(rate*.01));x=values[:,None] if values.ndim==1 else values
    padded=np.pad(x,((0,(-len(x))%hop),(0,0)))
    return np.sqrt(np.mean(padded.reshape(-1,hop,x.shape[1])**2,axis=1)).max(axis=1),hop/rate


def active_cut(values,rate,start,end):
    rms,dt=levels(values,rate)
    a=max(0,int(start/dt));b=min(len(rms),max(a+1,math.ceil(end/dt)))
    body=rms[a:b]
    if not len(body) or float(body.max())<=1e-7:return False
    gate=max(1e-7,float(np.percentile(body,90))*.04)
    pre=rms[max(a,b-3):b];post=rms[b:min(len(rms),b+3)]
    return bool(len(post) and np.median(pre)>gate and np.median(post)>gate)


def recover_ends(audio,rows):
    result=copy.deepcopy(rows);audit=[]
    with sf.SoundFile(audio) as stream:
        rate=stream.samplerate;duration=len(stream)/rate
        for i,row in enumerate(result):
            words=row.get('aligned_words') or []
            check={'id':row.get('id',i),'before':row['end'],'after':row['end'],'extended':False};audit.append(check)
            if not words:check['reason']='no_word_anchor';continue
            last=words[-1];original_end=row['end']
            ceiling=min(duration,rows[i+1]['start'] if i+1<len(rows) else duration)
            origin=max(row['start'],last['start']-.08)
            lo=max(0,round(origin*rate));origin=lo/rate
            last_hi=-1
            for lookahead in POLICY['lookahead_seconds']:
                hi=min(round(ceiling*rate),round((original_end+lookahead)*rate))
                if hi<=lo or hi==last_hi:continue
                last_hi=hi;stream.seek(lo);values=stream.read(hi-lo,dtype='float32',always_2d=True)
                try:
                    if not active_cut(values,rate,last['start']-origin,original_end-origin):
                        check['reason']='lexical_end_already_quiet';break
                    local_word={**last,'start':last['start']-origin,'end':last['end']-origin}
                    candidate=acoustic_boundaries(values,rate,[{'words':[local_word],
                        'source_word_floor':0.,'source_word_ceiling':len(values)/rate,'backend':'MMS-FA'}])[0]
                    check['last_measurement']={k:candidate[k] for k in ('end','end_method','decay_end_seconds')}
                    if candidate['end_method']!='sustained_quiet_after_voice_decay':
                        check['reason']='tail_still_active_or_no_quiet_boundary';continue
                    end=min(ceiling,origin+candidate['end'])
                    if end<=original_end+1/rate:
                        check['reason']='no_extension_needed';break
                    row['lexical_end']=original_end;row['end']=round(end*rate)/rate
                    row['voice_release']={'method':POLICY['method'],'lexical_end':original_end,'acoustic_end':row['end']}
                    check.update(after=row['end'],extended=True,reason='measured_voice_release');break
                except (ValueError,RuntimeError,FloatingPointError) as error:
                    check.update(reason='measurement_unavailable',error=str(error));break
    return result,{'policy':POLICY,'extended_count':sum(c['extended'] for c in audit),'checks':audit}
