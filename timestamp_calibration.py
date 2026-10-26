"""Estimate a recording-wide timestamp bias from isolated acoustic attacks."""
import numpy as np

def calibrate_word_times(values,sr,segments):
    if values.ndim==1:values=values[:,None]
    hop=max(1,round(.01*sr));dt=hop/sr;duration=len(values)/sr
    padded=np.pad(values,((0,(-len(values))%hop),(0,0)))
    energy=np.sqrt(np.mean(padded.reshape(-1,hop,values.shape[1])**2,axis=1)).max(axis=1)
    offsets=[]
    for i,row in enumerate(segments):
        words=[w for w in row['words'] if w['end']>w['start']]
        if not words:continue
        first=words[0];previous=segments[i-1]['end'] if i else 0
        if first['start']-previous<.4:continue
        a=max(0,round((first['start']-.3)/dt));b=min(len(energy),round((first['end']+.04)/dt))
        if b<=a or energy[a:b].max()<1e-6:continue
        threshold=float(energy[a:b].max())*.1
        active=np.flatnonzero(energy[a:b]>threshold)
        if not len(active):continue
        onset=a+int(active[0])
        if onset-a<3 or np.max(energy[max(a,onset-5):onset])>threshold:continue
        difference=first['start']-onset*dt
        if abs(difference)<.3:offsets.append(difference)
    median=float(np.median(offsets)) if offsets else 0.
    mad=float(np.median(np.abs(np.asarray(offsets)-median))) if offsets else 0.
    shift=median if len(offsets)>=6 and mad<.08 else 0.
    result=[]
    for row in segments:
        words=[{**w,'provider_start':w['start'],'provider_end':w['end'],
                'start':max(0.,w['start']-shift),'end':min(duration,max(0.,w['end']-shift))} for w in row['words']]
        positive=[w for w in words if w['end']>w['start']]
        if not positive:
            result.append(dict(row));continue
        result.append({**row,'words':words,'start':positive[0]['start'],'end':positive[-1]['end']})
    return result,{'shift_seconds':-shift,'isolated_attacks':len(offsets),'median_absolute_deviation_seconds':mad,
        'raw_provider_timestamps_preserved':True}
