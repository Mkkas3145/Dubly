"""Direct Instrumental stem from the official stereo two-stem Deux model."""
import gc,hashlib,json,sys
from pathlib import Path
import soundfile as sf
from core import ROOT,SR,atomic_json
from install_deux import CHECKPOINT,ASSETS,WEIGHT_SHA256,CONFIG_SHA256,ready,digest
from separation_audio import overlap_add_stems

CHUNK=573300
HOP=286650


def select_instrumental(prediction,frames):
    if tuple(prediction.shape)!=(1,2,2,frames):raise RuntimeError('Invalid Deux output stems, channels, or duration.')
    # The official configuration declares Vocals first, Instrumental second.
    return prediction[0,1]


def load_model():
    import torch,yaml
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU was found.')
    if not ready():raise RuntimeError('Mel-Band RoFormer Deux is not installed. Run install_deux.py.')
    if digest(CHECKPOINT)!=WEIGHT_SHA256 or digest(ASSETS/'config.yaml')!=CONFIG_SHA256:
        raise RuntimeError('Deux model checksum mismatch. Run install_deux.py to repair.')
    upstream=ROOT/'vendor/mel-band-roformer'
    for relative,expected in [
        ('models/mel_band_roformer/mel_band_roformer.py','d0e958c58114872c3a0af54f97719e07690ff5d98473fcd8b48485f776737b8f'),
        ('models/mel_band_roformer/attend.py','9ffd7d06b22a53bb8e6162adf674e7298d2227e1eea4f6404cb89ff1bbefecbe')]:
        if hashlib.sha256((upstream/relative).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise RuntimeError('Deux source code checksum mismatch. Run setup.ps1 to repair.')
    sys.path.insert(0,str(upstream))
    from models.mel_band_roformer import MelBandRoformer
    from models.mel_band_roformer.attend import Attend,FlashAttentionConfig
    config=yaml.load((ASSETS/'config.yaml').read_text(encoding='utf-8'),Loader=yaml.FullLoader)
    if (config['training']['instruments']!=['Vocals','Instrumental'] or config['model']['num_stems']!=2
        or not config['model']['stereo'] or config['audio']['sample_rate']!=SR
        or config['audio']['chunk_size']!=CHUNK or config['inference']['num_overlap']!=2):
        raise RuntimeError('Deux configuration differs from the official configuration.')
    model=MelBandRoformer(**config['model'])
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    if 'state_dict' in state:state=state['state_dict']
    model.load_state_dict(state,strict=True)
    for layer in model.modules():
        if isinstance(layer,Attend):layer.cuda_config=FlashAttentionConfig(True,True,True)
    return model.eval().to('cuda')


def separate(source,destination,progress,*,stem='Instrumental'):
    return separate_stems(source,{stem:destination},progress)


def separate_stems(source,destinations,progress):
    import torch
    source=Path(source)
    if not destinations or any(name not in {'Vocals','Instrumental'} for name in destinations):raise ValueError('Invalid Deux stems')
    destinations={name:Path(path) for name,path in destinations.items()}
    if len(set(destinations.values()))!=len(destinations):raise ValueError('Duplicate output path')
    info=sf.info(source)
    if info.frames<1 or info.samplerate!=SR or info.channels!=2:
        raise RuntimeError('Deux input must be nonempty 44.1 kHz stereo audio.')
    for path in destinations.values():path.parent.mkdir(parents=True,exist_ok=True)
    temporary=[path.with_name(path.stem+'.partial.wav') for path in destinations.values()]
    indices=[['Vocals','Instrumental'].index(name) for name in destinations]
    progress(0,'Mel-Band RoFormer Deux 로딩 중');torch.set_num_threads(4)
    model=load_model()
    try:
        def predict(samples):
            tensor=torch.from_numpy(samples.T.copy()).unsqueeze(0).to('cuda')
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                prediction=model(tensor)
                select_instrumental(prediction,len(samples)) # Validate both stems and stereo dimensions.
            return prediction[0].float().cpu().numpy().transpose(0,2,1)[indices]
        overlap_add_stems(source,temporary,predict,lambda p,d:progress(p*.99,d),chunk=CHUNK,hop=HOP,label='Deux 음성·배경 분리')
        for path in temporary:
            result=sf.info(path)
            if (result.frames,result.samplerate,result.channels)!=(info.frames,SR,2):raise RuntimeError('Deux output validation failed.')
        for path,destination in zip(temporary,destinations.values()):path.replace(destination)
        progress(100,'Mel-Band RoFormer Deux 완료')
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
        for path in temporary:path.unlink(missing_ok=True)


def prepare_pair(source,folder,progress):
    """Validate a matched pair; never accept an Instrumental-only or old-model cache."""
    source,folder=Path(source),Path(folder)
    voice,background=folder/'dialogue.wav',folder/'background.wav'
    metadata=folder/'separation-pair.json'
    signature={'source_sha256':digest(source),'weight_sha256':WEIGHT_SHA256,'version':'deux-dual-v1','overlap':2}
    valid=False
    if metadata.is_file() and voice.is_file() and background.is_file():
        try:
            record=json.loads(metadata.read_text(encoding='utf-8'))
            info=sf.info(source)
            valid=(record['signature']==signature and all(
                (sf.info(path).frames,sf.info(path).samplerate,sf.info(path).channels)==(info.frames,SR,2)
                and record['outputs'][path.name]==digest(path) for path in [voice,background]))
        except (KeyError,ValueError,RuntimeError,OSError):pass
    if not valid:
        separate_stems(source,{'Vocals':voice,'Instrumental':background},progress)
        atomic_json(metadata,{'signature':signature,'outputs':{path.name:digest(path) for path in [voice,background]},
            'forward_passes_per_chunk':1,'stems':['Vocals','Instrumental'],'sample_rate':SR,'channels':2})
    progress(100,'Deux 대사·배경 분리 완료')
    return voice,background
