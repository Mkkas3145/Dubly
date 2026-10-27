"""Read-only acoustic contrast for ranking phonetic alignment candidates.

It never supplies timestamps itself or alters audio. Contrast supplements MMS
and context agreement; loudness alone cannot create a lexical candidate.
"""
import math
import numpy as np
import soundfile as sf
FRAME_SECONDS=.02


def profile(path):
    levels=[]
    with sf.SoundFile(path) as stream:
        hop=max(1,round(stream.samplerate*FRAME_SECONDS))
        while True:
            values=stream.read(hop*1000,dtype='float32',always_2d=True)
            if not len(values):break
            values=np.pad(values,((0,(-len(values))%hop),(0,0)))
            rms=np.sqrt(np.mean(values.reshape(-1,hop,values.shape[1])**2,axis=1)).max(axis=1)
            levels.append(rms)
    return np.concatenate(levels) if levels else np.zeros(0,np.float32)


def contrast(levels,start,end):
    lo=max(0,round(start/FRAME_SECONDS));hi=min(len(levels),round(end/FRAME_SECONDS))
    if hi<=lo:return 0.
    signal=float(np.percentile(levels[lo:hi],60))
    if signal<=1e-8:return 0.
    sides=[levels[max(0,lo-50):lo],levels[hi:min(len(levels),hi+50)]]
    floors=[float(np.percentile(side,20)) for side in sides if len(side)]
    if not floors:return 0.
    floor=max(min(floors),1e-8)
    db=20*math.log10(max(signal,1e-8)/floor)
    return max(0.,min(1.,(db-6)/30))
