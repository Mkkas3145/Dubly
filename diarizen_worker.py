"""Pinned official DiariZen v2 pipeline, isolated from production diarization."""
import os,sys,json,time
from pathlib import Path
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/diarizen/pyannote-audio'))
sys.path.insert(0,str(ROOT/'vendor/diarizen'))
sys.stdout.reconfigure(encoding='utf8')

def main(source,destination):
    import torch,torchaudio,soundfile as sf,toml
    from diarizen.pipelines.inference import DiariZenPipeline
    # Lightning inspects module filenames; do not import optional SpeechBrain
    # k2 integrations merely to inspect their lazy module metadata.
    import speechbrain
    from speechbrain.utils.importutils import LazyModule
    for module in list(sys.modules.values()):
        if isinstance(module,LazyModule):
            target=vars(module).get('target','')
            path=Path(speechbrain.__file__).parent.joinpath(*target.split('.')[1:])
            origin=path/'__init__.py' if path.is_dir() else path.with_suffix('.py')
            vars(module).setdefault('__file__',str(origin) if origin.exists() else None)
    from core import atomic_json
    from install_diarizen import ASSETS,MODEL,REVISION
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('DiariZen CUDA unavailable')
    # The original project uses torchaudio.load. Read PCM with SoundFile to
    # avoid Windows TorchCodec DLL requirements without changing the signal.
    def load_pcm(path,frame_offset=0,num_frames=-1,normalize=True,channels_first=True,**kwargs):
        with sf.SoundFile(path) as audio:
            audio.seek(frame_offset);values=audio.read(num_frames,dtype='float32',always_2d=True);sr=audio.samplerate
        return torch.from_numpy(values.T.copy() if channels_first else values.copy()),sr
    torchaudio.load=load_pcm
    started=time.monotonic();configuration=toml.load(ASSETS/'segmentation/config.toml')
    configuration['inference']['args']['batch_size']=4
    print(json.dumps(dict(stage='load',model=MODEL,revision=REVISION,max_speakers=configuration['clustering']['args']['max_speakers'])),flush=True)
    pipeline=DiariZenPipeline(ASSETS/'segmentation',str(ASSETS/'embedding/pytorch_model.bin'),config_parse=configuration)
    pipeline.embedding_batch_size=16
    # Retain the model's frame votes for reproducible boundary experiments.
    # This capture does not modify its decoding, labels or native boundaries.
    native_reconstruct=pipeline.reconstruct
    def capture_reconstruction(*args,**kwargs):
        import numpy as np
        discrete,activations=native_reconstruct(*args,**kwargs)
        np.savez_compressed(destination.with_suffix('.frames.npz'),votes=activations.data,
            active=discrete.data,frame_start=activations.sliding_window.start,
            frame_step=activations.sliding_window.step,frame_duration=activations.sliding_window.duration)
        return discrete,activations
    pipeline.reconstruct=capture_reconstruction
    print(json.dumps(dict(stage='inference',duration=sf.info(source).duration)),flush=True)
    with torch.inference_mode():annotation=pipeline(str(source),sess_name='comparison')
    mapping={};turns=[]
    for turn,_,speaker in annotation.itertracks(yield_label=True):
        label=mapping.setdefault(str(speaker),f'D{len(mapping)}')
        turns.append(dict(start=float(turn.start),end=float(turn.end),speaker=label,raw_speaker=str(speaker)))
    result=dict(model=MODEL,revision=REVISION,duration=sf.info(source).duration,
        speakers=len(mapping),turns=turns,elapsed_seconds=time.monotonic()-started,
        parameters=configuration,signal_processing='mono 16kHz input; model-native boundaries; no acoustic boundary refinement')
    atomic_json(destination,result)
    print(json.dumps(dict(stage='complete',speakers=len(mapping),turns=len(turns),seconds=result['elapsed_seconds'])),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
