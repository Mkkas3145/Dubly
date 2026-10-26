"""Install the pinned official becruily Deux checkpoint and two-stem configuration."""
from pathlib import Path
import time,urllib.request
from core import ROOT,atomic_json
from install_hyperace import digest,atomic_bytes,verified_bytes

MODEL_ID='mel-band-roformer-deux'
LABEL='Mel-Band RoFormer Deux'
REPO='becruily/mel-band-roformer-deux'
REVISION='402dd9a19baaed8c6f1b54458ce6ae2ae19f354e'
WEIGHT_SHA256='10255c02295bf3e3865d4ee50ff752d7b19b124ed5fd93b147babc4333eda3aa'
WEIGHT_BYTES=435006815
CONFIG_SHA256='bb3ea9bce37ca96d63568490d5a92d7e41df3d7726788b6970c10d89eb62d902'
CHECKPOINT=ROOT/'models/mel-band-roformer-deux.ckpt'
ASSETS=ROOT/'vendor/mel-band-roformer-deux'


def ready():
    return (CHECKPOINT.is_file() and CHECKPOINT.stat().st_size==WEIGHT_BYTES
        and all((ASSETS/n).is_file() for n in ['config.yaml','manifest.json']))


def install():
    CHECKPOINT.parent.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    if not CHECKPOINT.is_file() or digest(CHECKPOINT)!=WEIGHT_SHA256:
        temporary=CHECKPOINT.with_suffix('.download')
        request=urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/becruily_deux.ckpt')
        started=time.monotonic();last=0;received=0
        with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as out:
            while block:=response.read(1024*1024):
                out.write(block);received+=len(block)
                if time.monotonic()-last>=5 or received==WEIGHT_BYTES:
                    last=time.monotonic()
                    print(f'Deux download {received/WEIGHT_BYTES*100:.1f}% | {received/1e6:.1f}/{WEIGHT_BYTES/1e6:.1f} MB | {received/1e6/max(.001,last-started):.1f} MB/s',flush=True)
        if received!=WEIGHT_BYTES or digest(temporary)!=WEIGHT_SHA256:raise RuntimeError('Deux model weights checksum mismatch.')
        temporary.replace(CHECKPOINT)
    with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/config_deux_becruily.yaml',timeout=60) as response:
        content=response.read()
    atomic_bytes(ASSETS/'config.yaml',verified_bytes(content,CONFIG_SHA256))
    atomic_json(ASSETS/'manifest.json',{'model_id':MODEL_ID,'repo':REPO,'revision':REVISION,
        'weight_sha256':WEIGHT_SHA256,'config_sha256':CONFIG_SHA256,'stems':['Vocals','Instrumental'],
        'selected_stem':'Instrumental','sample_rate':44100,'channels':2,'chunk_frames':573300,'overlap':2})
    print('Installed and verified Mel-Band RoFormer Deux.',flush=True)


if __name__=='__main__':install()
