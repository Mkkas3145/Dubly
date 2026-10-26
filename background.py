"""Direct instrumental estimation, independent of ASR/TTS references."""
import json
import soundfile as sf
from core import atomic_json, SR, DEFAULT_BACKGROUND
from install_hyperace import digest

MODELS={'leap-xe-vocals':'Leap Xe residual background','hyperace-v2-instrumental':'HyperACE v2 Instrumental','mel-band-roformer-deux':'Mel-Band RoFormer Deux'}


def prepare_background(original,work,progress,model_name=DEFAULT_BACKGROUND):
    if model_name=='leap-xe-vocals':
        from leap_xe import prepare_pair
        return prepare_pair(original,work/'runs/leap-xe-vocals-v1',progress)[1]
    if model_name=='mel-band-roformer-deux':
        from install_deux import WEIGHT_SHA256
        from deux import separate
        infer=lambda:separate(original,output,progress)
        overlap=2
    elif model_name=='hyperace-v2-instrumental':
        from install_background import WEIGHT_SHA256
        from hyperace import separate
        infer=lambda:separate(original,output,progress,instrumental=True)
        overlap=4
    else:raise RuntimeError('Unsupported background model.')
    output=work/'runs'/f'{model_name}-v1'/'background.wav'
    metadata=output.with_suffix('.json')
    signature={'input_sha256':digest(original),'model_sha256':WEIGHT_SHA256,'overlap':overlap}
    valid=False
    if output.is_file() and metadata.is_file():
        try:
            actual,source=sf.info(output),sf.info(original)
            valid=(json.loads(metadata.read_text(encoding='utf-8'))==signature and
                (actual.frames,actual.samplerate,actual.channels)==(source.frames,SR,2))
        except (ValueError,RuntimeError,OSError):pass
    if not valid:
        infer()
        atomic_json(metadata,signature)
    progress(100,MODELS[model_name]+' 완료')
    return output
