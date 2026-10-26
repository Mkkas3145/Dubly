"""Shared streaming overlap-add for source separation."""
import numpy as np
import soundfile as sf
from contextlib import ExitStack


def overlap_add(source,destination,predict,progress,chunk=384000,hop=48000,label='음성 분리'):
    return overlap_add_stems(source,[destination],lambda part:np.asarray(predict(part))[None,...],progress,chunk=chunk,hop=hop,label=label)


def overlap_add_stems(source,destinations,predict,progress,chunk=384000,hop=48000,label='음성 분리'):
    """Infer all requested stems once per chunk and reconstruct each without summing."""
    if not 0<hop<=chunk or chunk%hop:raise ValueError('Invalid overlap')
    stems=len(destinations)
    if not stems:raise ValueError('No output stems')
    with sf.SoundFile(source) as src:
        total,channels,rate=len(src),src.channels,src.samplerate
        if total<1 or channels not in {1,2}:raise RuntimeError('Separation input audio is empty or has an invalid channel format.')
        positions=range(-chunk+hop,total,hop)
        window=np.hanning(chunk+1)[:-1].astype(np.float32)
        accum=np.zeros((stems,chunk,channels),dtype=np.float32);weights=np.zeros(chunk,dtype=np.float32)
        with ExitStack() as stack:
            outputs=[stack.enter_context(sf.SoundFile(path,'w',samplerate=rate,channels=channels,subtype='FLOAT',format='RF64')) for path in destinations]
            for n,start in enumerate(positions):
                lo,hi=max(0,start),min(total,start+chunk)
                src.seek(lo);part=src.read(hi-lo,dtype='float32',always_2d=True)
                part=np.pad(part,((max(0,-start),max(0,start+chunk-total)),(0,0)),mode='reflect' if len(part)>1 else 'edge')
                prediction=np.asarray(predict(part),dtype=np.float32)
                if prediction.shape!=(stems,*part.shape) or not np.isfinite(prediction).all():
                    raise RuntimeError('Invalid separated output format or values.')
                accum+=prediction*window[None,:,None];weights+=window
                left,right=max(0,-start),min(hop,total-start)
                if right>left:
                    for index,out in enumerate(outputs):out.write(accum[index,left:right]/np.maximum(weights[left:right,None],1e-8))
                accum[:,:-hop]=accum[:,hop:].copy();accum[:,-hop:]=0
                weights[:-hop]=weights[hop:].copy();weights[-hop:]=0
                progress((n+1)/len(positions)*100,f'{label} · {n+1}/{len(positions)} 구간')
            if any(out.frames!=total for out in outputs):raise RuntimeError('Separated audio duration validation failed.')
