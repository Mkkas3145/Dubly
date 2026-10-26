from __future__ import annotations
import gc
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import soundfile as sf
from core import ROOT,SR,command
from separation_audio import overlap_add
from install_bandit import CHECKPOINT,MD5,UPSTREAM,checksum

NATIVE_SR=48000
CHUNK=NATIVE_SR*8
HOP=NATIVE_SR


def load_model():
    import torch
    import yaml
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU was found.')
    if not CHECKPOINT.is_file() or checksum(CHECKPOINT)!=MD5:
        raise RuntimeError('BandIt v2 Multi installation or checksum verification is required. Run install_bandit.py.')
    name='_fast_dubbing_bandit_v2'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,UPSTREAM/'src/__init__.py',submodule_search_locations=[str(UPSTREAM/'src')])
        package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    Bandit=importlib.import_module(name+'.models.bandit.bandit').Bandit
    params=yaml.safe_load((UPSTREAM/'configs/models/bandit-mus64.yaml').read_text(encoding='utf-8'))['kwargs']
    model=Bandit(**params,fs=NATIVE_SR,stems=['speech','music','sfx'])
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    state={key.removeprefix('model.'):value for key,value in checkpoint['state_dict'].items() if key.startswith('model.')}
    model.load_state_dict(state,strict=True)
    # Keep the pretrained speech decoder; background is the untouched mixture minus speech.
    model.mask_estim=torch.nn.ModuleDict({'speech':model.mask_estim['speech']})
    model.stems=['speech']
    return model.eval().to('cuda')


def separate(source,destination,progress):
    import torch
    source,destination=Path(source),Path(destination)
    info=sf.info(source)
    if info.samplerate!=SR or info.frames<1:raise RuntimeError('Input audio must be nonempty and sampled at 44.1 kHz.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name('dialogue.partial.wav')
    progress(0,'BandIt v2 Multi 로딩 중')
    model=load_model();torch.set_num_threads(4)
    try:
        # Disk intermediates keep RAM independent of video duration and are removed on success/error.
        with tempfile.TemporaryDirectory(prefix='bandit-',dir=destination.parent) as scratch:
            native=Path(scratch)/'input48.wav';speech=Path(scratch)/'speech48.wav'
            command(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-ar',str(NATIVE_SR),'-c:a','pcm_f32le','-rf64','auto',str(native)])
            def predict(part):
                # Upstream's inference also treats each stereo channel as a mono batch item.
                tensor=torch.from_numpy(part.T.copy()).unsqueeze(1).to('cuda')
                with torch.inference_mode():
                    output=model({'mixture':{'audio':tensor}})['estimates']['speech']['audio']
                return output[:,0].float().cpu().numpy().T
            overlap_add(native,speech,predict,lambda p,d:progress(3+p*.94,d),label='BandIt')
            command(['ffmpeg','-nostdin','-v','error','-y','-i',str(speech),'-af',f'aresample={SR},apad,atrim=end_sample={info.frames}',
                '-c:a','pcm_f32le','-rf64','auto',str(temporary)])
            result=sf.info(temporary)
            if result.frames!=info.frames or result.samplerate!=SR or result.channels!=info.channels:
                raise RuntimeError('BandIt output format or duration validation failed.')
            temporary.replace(destination);progress(100,'BandIt v2 Multi 완료')
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
