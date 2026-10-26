"""FP32 stereo Leap Xe, bounded-memory MSST demix and residual background."""
import gc,sys,importlib,importlib.util,json
from pathlib import Path
import numpy as np,soundfile as sf
from core import SR,atomic_json
from install_leap_xe import VENDOR,ASSETS,CHECKPOINT,FILES,CODE,verify,digest

CHUNK=881559
STEP=CHUNK//2
BATCH=2
VERSION=1
def load_model():
 import torch,yaml
 if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU was found.')
 verify()
 name='_fast_dubbing_leap_xe'
 if name not in sys.modules:
  spec=importlib.util.spec_from_file_location(name,VENDOR/'__init__.py',submodule_search_locations=[str(VENDOR)])
  package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
 module=importlib.import_module(name+'.bs_roformer')
 cfg=yaml.load((ASSETS/'leap_xe_config_voc.yaml').read_text(),Loader=yaml.FullLoader)
 if cfg['audio']['sample_rate']!=SR or cfg['audio']['chunk_size']!=CHUNK or cfg['inference']['num_overlap']!=2 or not cfg['model']['stereo']:raise RuntimeError('Leap Xe configuration mismatch.')
 model=module.BSRoformer(**cfg['model'])
 state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
 model.load_state_dict(state.get('state_dict',state),strict=True)
 return model.eval().to('cuda',dtype=torch.float32)

def virtual_part(source,start,chunk=CHUNK):
 """Exact reflect border from MSST, without storing the whole track on GPU."""
 total=len(source);border=chunk-chunk//2 if total>2*(chunk-chunk//2) else 0
 length=total+2*border;n=min(chunk,length-start)
 indices=np.arange(start,start+n,dtype=np.int64)-border
 indices=np.where(indices<0,-indices,indices)
 indices=np.where(indices>=total,2*total-2-indices,indices)
 lo,hi=int(indices.min()),int(indices.max())+1
 source.seek(lo);part=source.read(hi-lo,dtype='float32',always_2d=True)[indices-lo]
 if n<chunk:part=np.pad(part,((0,chunk-n),(0,0)),mode='reflect' if n>chunk//2 else 'constant')
 return part,n,length,border

def stream(source,dest,predict,progress,*,chunk=CHUNK,batch_size=BATCH):
 step=chunk//2;fade=chunk//10
 window=np.ones(chunk,np.float32);window[:fade]=np.linspace(0,1,fade);window[-fade:]=np.linspace(1,0,fade)
 with sf.SoundFile(source) as src,sf.SoundFile(dest,'w',samplerate=SR,channels=2,format='RF64',subtype='FLOAT') as out:
  total=len(src);border=chunk-step if total>2*(chunk-step) else 0;length=total+2*border
  positions=range(0,length,step)
  accum=np.zeros((chunk,2),np.float32);weights=np.zeros(chunk,np.float32)
  for offset in range(0,len(positions),batch_size):
   locations=list(positions[offset:offset+batch_size]);parts=[virtual_part(src,start,chunk) for start in locations]
   results=np.asarray(predict(np.stack([p[0] for p in parts])),dtype=np.float32)
   if results.shape!=(len(parts),chunk,2) or not np.isfinite(results).all():raise RuntimeError('Invalid Leap Xe separated output.')
   for start,(part,n,_,_),prediction in zip(locations,parts,results):
    w=window.copy()
    if start==0:w[:fade]=1
    if start+chunk>=length:w[-fade:]=1
    accum[:n]+=prediction[:n]*w[:n,None];weights[:n]+=w[:n]
    left=max(0,border-start);right=min(step,border+total-start)
    if right>left:
     if np.any(weights[left:right]<=0):raise RuntimeError('Leap Xe synthesis boundary validation failed.')
     out.write(accum[left:right]/weights[left:right,None])
    accum[:-step]=accum[step:].copy();accum[-step:]=0
    weights[:-step]=weights[step:].copy();weights[-step:]=0
   done=min(offset+batch_size,len(positions));progress(done/len(positions)*100,f'Leap Xe · {done}/{len(positions)} 구간')
  if out.frames!=total:raise RuntimeError('Leap Xe output duration validation failed.')

def separate(source,destination,progress):
 import torch
 source,destination=Path(source),Path(destination);info=sf.info(source)
 if info.frames<1 or info.samplerate!=SR or info.channels!=2:raise RuntimeError('Leap Xe input must be 44.1 kHz stereo audio.')
 destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix('.partial.wav')
 progress(0,'Leap Xe Vocals FP32 로딩 중');torch.set_num_threads(4);model=load_model()
 try:
  def predict(parts):
   x=torch.from_numpy(parts.transpose(0,2,1).copy()).cuda()
   with torch.inference_mode(),torch.autocast('cuda',enabled=False):y=model(x)
   if y.ndim==4:
    if y.shape[1]!=1:raise RuntimeError('Leap Xe output stem count does not match.')
    y=y[:,0]
   return y.float().cpu().numpy().transpose(0,2,1)
  stream(source,temp,predict,progress)
  temp.replace(destination)
 finally:
  del model;gc.collect();torch.cuda.empty_cache();temp.unlink(missing_ok=True)

def prepare_pair(source,folder,progress):
 source,folder=Path(source),Path(folder);folder.mkdir(parents=True,exist_ok=True)
 voice,bg=folder/'dialogue.wav',folder/'background.wav';meta=folder/'separation-pair.json'
 signature={'version':VERSION,'source_sha256':digest(source),'weight_sha256':FILES['bs_leap_xe_voc.ckpt'][1],'config_sha256':FILES['leap_xe_config_voc.yaml'][1],'code_sha256':{n:h[1] for n,h in CODE.items()},'precision':'FP32','chunk':CHUNK,'overlap':2,'batch_size':BATCH,'background':'original_minus_leap_xe_vocals'}
 if voice.is_file() and bg.is_file() and meta.is_file():
  try:
   old=json.loads(meta.read_text());info=sf.info(source)
   if old['signature']==signature and all((sf.info(p).frames,sf.info(p).samplerate,sf.info(p).channels)==(info.frames,SR,2) and digest(p)==old['outputs'][p.name] for p in [voice,bg]):
    progress(100,'Leap Xe 음성·배경 캐시 재사용');return voice,bg
  except (OSError,ValueError,KeyError,RuntimeError):pass
 meta.unlink(missing_ok=True)
 separate(source,voice,lambda p,d:progress(p*.96,d))
 temp=bg.with_suffix('.partial.wav')
 try:
  with sf.SoundFile(source) as src,sf.SoundFile(voice) as clean,sf.SoundFile(temp,'w',samplerate=SR,channels=2,subtype='FLOAT',format='RF64') as dst:
   while True:
    original=src.read(1048576,dtype='float32',always_2d=True)
    if not len(original):break
    vocals=clean.read(len(original),dtype='float32',always_2d=True)
    if vocals.shape!=original.shape:raise RuntimeError('Leap Xe background duration validation failed.')
    dst.write(original-vocals)
  temp.replace(bg)
  atomic_json(meta,{'signature':signature,'outputs':{p.name:digest(p) for p in [voice,bg]},'sample_rate':SR,'channels':2})
  progress(100,'Leap Xe 음성·배경 분리 완료');return voice,bg
 finally:temp.unlink(missing_ok=True)
