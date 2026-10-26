from __future__ import annotations
import gc
import importlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from core import SR
from install_hyperace import ASSETS,CHECKPOINT,WEIGHT_SHA256,CONFIG_SHA256,PATCHED_SHA256,ATTEND_SHA256,digest,ready
from separation_audio import overlap_add

CHUNK=960000
HOP=240000


def load_model(instrumental=False):
    import torch
    import yaml
    # Both targets use the identical architecture, with independent pinned assets.
    import install_background as background
    assets=background.ASSETS if instrumental else ASSETS
    checkpoint=background.CHECKPOINT if instrumental else CHECKPOINT
    weight_hash=background.WEIGHT_SHA256 if instrumental else WEIGHT_SHA256
    config_hash=background.CONFIG_SHA256 if instrumental else CONFIG_SHA256
    installed=background.ready if instrumental else ready
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU was found.')
    if not installed():raise RuntimeError('HyperACE is not installed. Run the installation script.')
    for path,expected in [(checkpoint,weight_hash),(assets/'config.yaml',config_hash),
                          (assets/'bs_roformer.py',PATCHED_SHA256),(assets/'attend.py',ATTEND_SHA256)]:
        if digest(path)!=expected:raise RuntimeError('HyperACE installation checksum verification failed. Run install_hyperace.py.')
    name='_fast_dubbing_hyperace_v2_'+('instrumental' if instrumental else 'vocals')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,assets/'__init__.py',submodule_search_locations=[str(assets)])
        package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    module=importlib.import_module(name+'.bs_roformer');attend=importlib.import_module(name+'.attend')
    config=yaml.load((assets/'config.yaml').read_text(encoding='utf-8'),Loader=yaml.FullLoader)
    if (config['training']['target_instrument']!=('instrument' if instrumental else 'vocals') or not config['model']['stereo']
        or config['audio']['chunk_size']!=CHUNK or config['inference']['num_overlap']!=4):
        raise RuntimeError('Invalid HyperACE dialogue extraction configuration.')
    model=module.BSRoformer(**config['model'])
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'state_dict' in state:state=state['state_dict']
    model.load_state_dict(state,strict=True)
    for layer in model.modules():
        if isinstance(layer,attend.Attend):layer.cuda_config=attend.FlashAttentionConfig(True,True,True)
    return model.eval().to('cuda')


def separate(source,destination,progress,*,instrumental=False):
    import torch
    source,destination=Path(source),Path(destination)
    info=sf.info(source)
    if info.frames<1 or info.samplerate!=SR or info.channels!=2:
        raise RuntimeError('HyperACE input must be nonempty 44.1 kHz stereo audio.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.stem+'.partial.wav')
    label='HyperACE v2 '+('Instrumental' if instrumental else 'Vocals')
    progress(0,label+' 로딩 중');torch.set_num_threads(4)
    model=load_model(instrumental=True) if instrumental else load_model()
    try:
        def predict(samples):
            tensor=torch.from_numpy(samples.T.copy()).unsqueeze(0).to('cuda')
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                prediction=model(tensor)[0]
            return prediction.float().cpu().numpy().T
        overlap_add(source,temporary,predict,lambda p,d:progress(p*.99,d),chunk=CHUNK,hop=HOP,label='HyperACE')
        result=sf.info(temporary)
        if (result.frames,result.samplerate,result.channels)!=(info.frames,SR,2):
            raise RuntimeError('HyperACE output duration or channel validation failed.')
        temporary.replace(destination);progress(100,label+' 완료')
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
        temporary.unlink(missing_ok=True)
