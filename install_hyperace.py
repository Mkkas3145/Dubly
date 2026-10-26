"""Install exactly the HyperACE v2 Vocals weights and runtime used in the audition."""
from pathlib import Path
import hashlib
import shutil
import urllib.request
from core import ROOT,atomic_json

MODEL_ID='hyperace-v2-vocals'
LABEL='BS-RoFormer HyperACE v2 Vocals'
REPO='pcunwa/BS-Roformer-HyperACE'
REVISION='5b1f8283125d5e4a3614d0e3635a636e09c84059'
WEIGHT_FILE='v2_voc/bs_roformer_voc_hyperacev2.ckpt'
WEIGHT_SHA256='54cf516f621f2f460bf660ed137e244b8931bf7a2ce85ddceecff816dbc4d668'
WEIGHT_BYTES=288724430
CHECKPOINT=ROOT/'models/hyperace-v2-vocals.ckpt'
ASSETS=ROOT/'vendor/hyperace-v2-vocals'
CONFIG_SHA256='033e57abee226a480ef410e0631b9fe00e8b7bc9a3bb5fe55659d0b43a238701'
SOURCE_SHA256='48571e20d70ea8f245cffc6afbfa279f62042e7ba16fbaa3fe43dd2cbc25e1db'
PATCHED_SHA256='ebdcbb88e8100d33e2a8d48ce5653b14ff3638bf2380b593633ebd17bc808136'
ATTEND_SHA256='9ffd7d06b22a53bb8e6162adf674e7298d2227e1eea4f6404cb89ff1bbefecbe'
ATTEND_URL='https://raw.githubusercontent.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/25f44ffb55ee3c301281bba21b2d6d311cb69ae2/models/mel_band_roformer/attend.py'


def digest(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def verified_bytes(content,expected):
    if hashlib.sha256(content).hexdigest()!=expected:raise RuntimeError('HyperACE installation checksum mismatch.')
    return content


def atomic_bytes(path,content):
    temporary=path.with_suffix(path.suffix+'.partial');temporary.write_bytes(content);temporary.replace(path)


def ready():
    return (CHECKPOINT.is_file() and CHECKPOINT.stat().st_size==WEIGHT_BYTES
        and all((ASSETS/name).is_file() for name in ['__init__.py','bs_roformer.py','attend.py','config.yaml','manifest.json']))


def install():
    from huggingface_hub import hf_hub_download
    trial=ROOT/'work/hyperace-v2/v2_voc'
    CHECKPOINT.parent.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    if not CHECKPOINT.is_file() or digest(CHECKPOINT)!=WEIGHT_SHA256:
        source=trial/'bs_roformer_voc_hyperacev2.ckpt'
        if not source.is_file() or digest(source)!=WEIGHT_SHA256:
            source=Path(hf_hub_download(repo_id=REPO,filename=WEIGHT_FILE,revision=REVISION))
        if digest(source)!=WEIGHT_SHA256:raise RuntimeError('HyperACE weights checksum mismatch.')
        temporary=CHECKPOINT.with_suffix('.partial');shutil.copyfile(source,temporary);temporary.replace(CHECKPOINT)
    for name,expected in [('config.yaml',CONFIG_SHA256),('bs_roformer.py',SOURCE_SHA256)]:
        source=trial/name
        if not source.is_file() or digest(source)!=expected:
            source=Path(hf_hub_download(repo_id=REPO,filename='v2_voc/'+name,revision=REVISION))
        content=verified_bytes(source.read_bytes(),expected)
        if name=='bs_roformer.py':
            # Package-relative import only; all inference/model computations stay unchanged.
            content=content.replace(b'from models.bs_roformer.attend import Attend',b'from .attend import Attend')
            verified_bytes(content,PATCHED_SHA256)
        atomic_bytes(ASSETS/name,content)
    attend=ROOT/'vendor/mel-band-roformer/models/mel_band_roformer/attend.py'
    if attend.is_file():content=attend.read_bytes().replace(b'\r\n',b'\n')
    else:
        with urllib.request.urlopen(ATTEND_URL,timeout=60) as response:content=response.read().replace(b'\r\n',b'\n')
    atomic_bytes(ASSETS/'attend.py',verified_bytes(content,ATTEND_SHA256))
    atomic_bytes(ASSETS/'__init__.py',b'')
    atomic_json(ASSETS/'manifest.json',{'model_id':MODEL_ID,'label':LABEL,'repo':REPO,'revision':REVISION,
        'weight_file':WEIGHT_FILE,'weight_sha256':WEIGHT_SHA256,'config_sha256':CONFIG_SHA256,
        'original_code_sha256':SOURCE_SHA256,'runtime_code_sha256':PATCHED_SHA256,'attention_sha256':ATTEND_SHA256,
        'attention_source':ATTEND_URL,'compatibility_patch':'package-relative Attend import only',
        'sample_rate':44100,'stereo_mode':'joint','chunk_frames':960000,'overlap':4,'precision':'float16_autocast'})
    print('Installed and verified HyperACE v2 Vocals (audition settings).',flush=True)


if __name__=='__main__':install()
