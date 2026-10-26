from __future__ import annotations
import gc
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
from core import ROOT, SR, DEFAULT_SEPARATOR

CHUNK=352800
MODELS={'leap-xe-vocals':'BS-RoFormer Leap Xe Vocals','mel-band-roformer-deux':'Mel-Band RoFormer Deux','hyperace-v2-vocals':'BS-RoFormer HyperACE v2 Vocals','bandit-v2-multi':'BandIt v2 Multi','melband-kim':'Mel-Band RoFormer Kim'}


def model_ready(name):
    if name=='leap-xe-vocals':
        from install_leap_xe import ready
        return ready()
    if name=='mel-band-roformer-deux':
        from install_deux import ready
        return ready()
    if name=='hyperace-v2-vocals':
        from install_hyperace import ready
        return ready()
    if name=='bandit-v2-multi':
        return (ROOT/'models/bandit-v2-multi.ckpt').is_file() and (ROOT/'vendor/bandit-v2/src/models/bandit/bandit.py').is_file()
    return name=='melband-kim' and (ROOT/'models/MelBandRoformer.ckpt').is_file()


def cache_directory(work,name):
    if name not in MODELS:raise RuntimeError('Invalid SEPARATION_MODEL setting.')
    path=Path(work)/'runs'/f'{name}-v1';path.mkdir(parents=True,exist_ok=True)
    return path


def load_model():
    import torch
    import yaml
    upstream=ROOT/'vendor'/'mel-band-roformer'
    sys.path.insert(0,str(upstream))
    from models.mel_band_roformer import MelBandRoformer
    from models.mel_band_roformer.attend import Attend, FlashAttentionConfig
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU was found.')
    config=yaml.load((upstream/'configs/config_vocals_mel_band_roformer.yaml').read_text(encoding='utf-8'),Loader=yaml.FullLoader)
    model=MelBandRoformer(**config['model'])
    model.load_state_dict(torch.load(ROOT/'models/MelBandRoformer.ckpt',map_location='cpu',weights_only=True),strict=True)
    for module in model.modules():
        if isinstance(module,Attend):
            module.cuda_config=FlashAttentionConfig(True,True,True)
    return model.eval().to('cuda')


def separate_melband(source, destination, progress):
    import torch
    model=load_model()
    torch.set_num_threads(4)
    step=CHUNK//2
    window=np.ones(CHUNK,dtype=np.float32)
    fade=CHUNK//10
    window[:fade]=np.linspace(0,1,fade)
    window[-fade:]=np.linspace(1,0,fade)
    accum=np.zeros((CHUNK,2),dtype=np.float32)
    weights=np.zeros(CHUNK,dtype=np.float32)
    temporary=destination.with_name('dialogue.partial.wav')
    try:
        with sf.SoundFile(source) as src,sf.SoundFile(temporary,'w',samplerate=SR,channels=2,subtype='PCM_24',format='RF64') as dst:
            total=len(src)
            if total==0:
                raise RuntimeError('Audio is empty.')
            positions=range(-step,total,step)
            with torch.inference_mode():
                for n,start in enumerate(positions):
                    lo,hi=max(0,start),min(total,start+CHUNK)
                    src.seek(lo)
                    part=src.read(hi-lo,dtype='float32',always_2d=True)
                    part=np.pad(part,((max(0,-start),max(0,start+CHUNK-total)),(0,0)),mode='reflect' if len(part)>1 else 'edge')
                    tensor=torch.from_numpy(part.T.copy()).unsqueeze(0).to('cuda')
                    with torch.autocast(device_type='cuda',dtype=torch.float16):
                        prediction=model(tensor)[0]
                    prediction=prediction.float().cpu().numpy().T
                    if not np.isfinite(prediction).all():
                        raise RuntimeError('Separation model returned invalid audio.')
                    accum+=prediction*window[:,None]
                    weights+=window
                    left,right=max(0,-start),min(step,total-start)
                    if right>left:
                        dst.write(np.clip(accum[left:right]/np.maximum(weights[left:right,None],1e-8),-1,1))
                    accum[:-step]=accum[step:].copy();accum[-step:]=0
                    weights[:-step]=weights[step:].copy();weights[-step:]=0
                    progress((n+1)/len(positions)*100,f'{n+1}/{len(positions)} 구간')
            if dst.frames!=total:
                raise RuntimeError('Separated audio duration validation failed.')
        temporary.replace(destination)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def separate(source,destination,progress,model_name=DEFAULT_SEPARATOR):
    if model_name=='leap-xe-vocals':
        from leap_xe import separate as separate_leap
        return separate_leap(source,destination,progress)
    if model_name=='mel-band-roformer-deux':
        from deux import separate as separate_deux
        return separate_deux(source,destination,progress,stem='Vocals')
    if model_name=='hyperace-v2-vocals':
        from hyperace import separate as separate_hyperace
        return separate_hyperace(source,destination,progress)
    if model_name=='bandit-v2-multi':
        from bandit import separate as separate_bandit
        return separate_bandit(source,destination,progress)
    if model_name=='melband-kim':return separate_melband(source,destination,progress)
    raise RuntimeError('Unsupported audio separation model.')
