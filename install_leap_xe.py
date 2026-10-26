"""Pinned public Leap Xe Vocals assets used in the approved comparison."""
from pathlib import Path
import hashlib
import requests
from core import ROOT,atomic_json

MODEL_ID='leap-xe-vocals'
REPO='pcunwa/BS-Roformer-Leap'
REVISION='4e47d6662ae82eaa8b4ac4329fe66099a843b48e'
ASSETS=ROOT/'models/leap-xe-vocals'
VENDOR=ROOT/'vendor/leap-xe'
CHECKPOINT=ASSETS/'bs_leap_xe_voc.ckpt'
CODE_REVISION='6fdad13686bb7e84d029ec66f9dbe76e65c93ebf'
FILES={
 'bs_leap_xe_voc.ckpt':(267796851,'b739c1d2d87a81cd3dd3844ed9ad0bd678708c7a0a761a03a1aaff9af79a096d'),
 'leap_xe_config_voc.yaml':(2190,'d3cb8c84be2e9bcbc64c1086e2256dc097ab5addebc5ce507818e90ff8cbdc25')}
CODE={
 'bs_roformer.py':('a6f670325cdb8a7a212914f62602119b23eb888df9bcfe4a85f9be74728b3ef4','5c2b9243435a9d550c84975148b54b7f2171df3465b4ebf31dab5f70b6d34cd9'),
 'attend.py':('c7abbc40a3fd20ff7f6001fa9f8ee9ad5df6452272712a5e719b8f8468bf2223','c7abbc40a3fd20ff7f6001fa9f8ee9ad5df6452272712a5e719b8f8468bf2223')}
def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ready():
 return all((ASSETS/n).is_file() and (ASSETS/n).stat().st_size==size for n,(size,_) in FILES.items()) and all((VENDOR/n).is_file() for n in ['__init__.py',*CODE])
def verify():
 if not ready():raise RuntimeError('Leap Xe is not installed. Run install_leap_xe.py.')
 for path,expected in [(ASSETS/n,h) for n,(_,h) in FILES.items()]+[(VENDOR/n,h[1]) for n,h in CODE.items()]:
  if digest(path)!=expected:raise RuntimeError('Leap Xe installation checksum verification failed.')
def install():
 ASSETS.mkdir(parents=True,exist_ok=True);VENDOR.mkdir(parents=True,exist_ok=True)
 for name,(size,sha) in FILES.items():
  p=ASSETS/name
  if p.is_file() and p.stat().st_size==size and digest(p)==sha:continue
  temp=p.with_suffix('.partial');n=0
  with requests.get(f'https://huggingface.co/{REPO}/resolve/{REVISION}/Xe/{name}',stream=True,timeout=(20,60)) as r:
   r.raise_for_status()
   with temp.open('wb') as out:
    for block in r.iter_content(4*1024*1024):out.write(block);n+=len(block)
  if n!=size or digest(temp)!=sha:raise RuntimeError('Leap Xe download verification failed.')
  temp.replace(p)
 for name,(original,patched) in CODE.items():
  p=VENDOR/name
  if p.is_file() and digest(p)==patched:continue
  r=requests.get(f'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{CODE_REVISION}/models/bs_roformer/{name}',timeout=60);r.raise_for_status()
  if hashlib.sha256(r.content).hexdigest()!=original:raise RuntimeError('Leap Xe source code verification failed.')
  content=r.content.replace(b'from models.bs_roformer.attend import Attend',b'from .attend import Attend')
  if hashlib.sha256(content).hexdigest()!=patched:raise RuntimeError('Leap Xe package path verification failed.')
  p.write_bytes(content)
 (VENDOR/'__init__.py').write_bytes(b'')
 verify()
 atomic_json(ASSETS/'production-installation.json',{'repo':REPO,'revision':REVISION,'code_revision':CODE_REVISION,'precision':'FP32','sample_rate':44100,'channels':2,'overlap':2,'batch_size':2})
 print('Leap Xe Vocals installed and verified.',flush=True)
if __name__=='__main__':install()
