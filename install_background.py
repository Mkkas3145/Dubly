"""Pinned HyperACE v2 Instrumental; keep the dialogue model independent."""
from pathlib import Path
import shutil
from core import ROOT, atomic_json
from install_hyperace import (REPO, REVISION, SOURCE_SHA256, PATCHED_SHA256,
    ATTEND_SHA256, digest, verified_bytes, atomic_bytes, ASSETS as VOCAL_ASSETS)

MODEL_ID='hyperace-v2-instrumental'
LABEL='BS-RoFormer HyperACE v2 Instrumental'
CHECKPOINT=ROOT/'models/hyperace-v2-instrumental.ckpt'
ASSETS=ROOT/'vendor/hyperace-v2-instrumental'
WEIGHT_SHA256='4d61178ef966d2b4e9ad456ffbbc6fd5b2828df07a7af32931142c1a5ff1fe6f'
WEIGHT_BYTES=288725646
CONFIG_SHA256='85d10906007df21ee48dbb86faa09205609274eb38ea224dce767e2844d0a934'


def ready():
    return (CHECKPOINT.is_file() and CHECKPOINT.stat().st_size==WEIGHT_BYTES
        and all((ASSETS/name).is_file() for name in ['__init__.py','bs_roformer.py','attend.py','config.yaml','manifest.json']))


def install():
    from huggingface_hub import hf_hub_download
    trial=ROOT/'work/hyperace-v2/v2_inst'
    CHECKPOINT.parent.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    def source(name,expected):
        path=trial/name
        if not path.is_file() or digest(path)!=expected:
            path=Path(hf_hub_download(repo_id=REPO,filename='v2_inst/'+name,revision=REVISION))
        if digest(path)!=expected:raise RuntimeError('Background model checksum mismatch.')
        return path
    if not CHECKPOINT.is_file() or digest(CHECKPOINT)!=WEIGHT_SHA256:
        temporary=CHECKPOINT.with_suffix('.partial')
        shutil.copyfile(source('bs_roformer_inst_hyperacev2.ckpt',WEIGHT_SHA256),temporary)
        temporary.replace(CHECKPOINT)
    atomic_bytes(ASSETS/'config.yaml',source('config.yaml',CONFIG_SHA256).read_bytes())
    code=source('bs_roformer.py',SOURCE_SHA256).read_bytes().replace(
        b'from models.bs_roformer.attend import Attend',b'from .attend import Attend')
    atomic_bytes(ASSETS/'bs_roformer.py',verified_bytes(code,PATCHED_SHA256))
    atomic_bytes(ASSETS/'attend.py',verified_bytes((VOCAL_ASSETS/'attend.py').read_bytes(),ATTEND_SHA256))
    atomic_bytes(ASSETS/'__init__.py',b'')
    atomic_json(ASSETS/'manifest.json',{'model_id':MODEL_ID,'repo':REPO,'revision':REVISION,
        'weight_sha256':WEIGHT_SHA256,'config_sha256':CONFIG_SHA256,'code_sha256':PATCHED_SHA256,
        'sample_rate':44100,'channels':2,'chunk_frames':960000,'overlap':4})
    print('Installed HyperACE v2 Instrumental using verified local weights.',flush=True)


if __name__=='__main__':install()
