"""Shared phonetic posterior engine for replay comparison and sample pages."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from numba import njit

MODEL='facebook/wav2vec2-xlsr-53-espeak-cv-ft'
class PhoneticEncoder:
 def __init__(self,folder='models/phonetic-xlsr',device='cuda'):
  import torch
  from transformers import Wav2Vec2ForCTC,Wav2Vec2FeatureExtractor
  torch.set_num_threads(4)
  self.torch=torch;self.device=device
  self.extractor=Wav2Vec2FeatureExtractor.from_pretrained(folder,local_files_only=True)
  self.model=Wav2Vec2ForCTC.from_pretrained(folder,local_files_only=True).to(device).eval()
  self.vocab={v:k for k,v in json.loads((Path(folder)/'vocab.json').read_text('utf8')).items()}
  self.blank=self.model.config.pad_token_id
 def encode(self,path,cache):
  path=Path(path);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
  digest=hashlib.sha256(path.read_bytes()+MODEL.encode()).hexdigest()
  dest=cache/(digest+'.npz')
  if dest.exists():return dict(np.load(dest,allow_pickle=False))
  audio,rate=sf.read(path,dtype='float32',always_2d=True);audio=audio.mean(axis=1)
  duration=len(audio)/rate
  if rate!=16000:
   g=math.gcd(rate,16000);audio=resample_poly(audio,16000//g,rate//g)
  inputs=self.extractor(audio,sampling_rate=16000,return_tensors='pt')
  inputs={k:v.to(self.device) for k,v in inputs.items()}
  with self.torch.inference_mode():
   probs=self.model(**inputs).logits[0].softmax(-1).float().cpu().numpy()
  ids=probs.argmax(-1);phones=[];starts=[];ends=[];post=[]
  step=.02
  for i in range(len(ids)):
   if ids[i]==self.blank:continue
   if i and ids[i]==ids[i-1]:
    ends[-1]=(i+1)*step;post[-1].append(probs[i])
   else:
    phones.append(int(ids[i]));starts.append(i*step);ends.append((i+1)*step);post.append([probs[i]])
  pp=np.array([np.mean(x,axis=0) for x in post],dtype=np.float32)
  # Exclude the CTC blank from phone identity and normalize distributions.
  if len(pp):
   pp[:,self.blank]=0;pp/=np.maximum(pp.sum(axis=1,keepdims=True),1e-9)
  result={'posterior':pp,'phones':np.array(phones,dtype=np.int32),'starts':np.array(starts),'ends':np.array(ends),'duration':np.array(duration),'frame_posterior':probs}
  np.savez_compressed(dest,**result)
  return result
 def text(self,result):return ' '.join(self.vocab[int(i)] for i in result['phones'])

def phone_kernel(vocab):
 """PanPhon articulatory similarity; no dialogue-specific phone substitutions."""
 import csv,re,unicodedata,panphon
 # Read UTF-8 explicitly: PanPhon's loader inherits Windows' cp949 locale.
 data=Path(panphon.__file__).parent/'data/ipa_all.csv'
 with data.open(encoding='utf8') as f:
  rows=list(csv.DictReader(f))
 names=[k for k in rows[0] if k!='ipa'];sign={'+':1,'-':-1,'0':0}
 table={unicodedata.normalize('NFD',r['ipa']):np.array([sign[r[k]] for k in names],dtype=np.float32) for r in rows}
 keys=sorted(table,key=len,reverse=True)
 vectors=[]
 for i in range(len(vocab)):
  token=unicodedata.normalize('NFD',re.sub(r'[0-9.ːˈˌ:]','',vocab[i]));parts=[]
  while token:
   key=next((s for s in keys if token.startswith(s)),None)
   if key:parts.append(table[key]);token=token[len(key):]
   else:token=token[1:]
  vectors.append(np.mean(parts,axis=0) if parts else np.full(len(names),np.nan))
 a=np.array(vectors)
 dist=np.nan_to_num(np.mean(np.abs(a[:,None,:]-a[None,:,:]),axis=-1),nan=1.)
 kernel=np.maximum(0,1-3*dist);np.fill_diagonal(kernel,1.)
 return kernel


@njit(cache=True,nogil=True)
def _candidate_scores(sim):
 """Reuse the DP columns for all endpoints at each start; no repeated paths."""
 m,n=sim.shape
 minimum=max(3,math.ceil(m*.65));maximum=math.floor(m*1.55)
 result=np.empty((n*max(0,maximum-minimum+1),3),dtype=np.float64);count=0
 for a in range(n-minimum+1):
  width=min(maximum,n-a)
  cost=np.full((m+1,width+1),np.inf);cost[0,0]=0.
  total=np.zeros_like(cost);length=np.zeros((m+1,width+1),dtype=np.int32)
  for i in range(1,m+1):
   for j in range(1,width+1):
    s=sim[i-1,a+j-1]
    best=cost[i-1,j-1]+2*(1-s)
    value=total[i-1,j-1]+s;size=length[i-1,j-1]+1
    if j>=2:
     s2=sim[i-1,a+j-2]
     candidate=cost[i-1,j-2]+1.5*(2-s-s2)
     if candidate<best:
      best=candidate;value=total[i-1,j-2]+s+s2;size=length[i-1,j-2]+2
    if i>=2:
     s2=sim[i-2,a+j-1]
     candidate=cost[i-2,j-1]+1.5*(2-s-s2)
     if candidate<best:
      best=candidate;value=total[i-2,j-1]+s+s2;size=length[i-2,j-1]+2
    cost[i,j]=best;total[i,j]=value;length[i,j]=size
  for b in range(minimum,width+1):
   if not math.isfinite(cost[m,b]) or length[m,b]==0:continue
   score=total[m,b]/length[m,b]-.12*abs(math.log(b/m))
   result[count,0]=a;result[count,1]=a+b;result[count,2]=score;count+=1
 return result[:count]


def candidates(reference,query,kernel):
 """Posterior-based, bounded subsequence DTW over CTC phone events."""
 r=reference['posterior'];q=query['posterior'];m,n=len(r),len(q)
 if m<3 or n<3:return []
 if math.ceil(m*.65)>n:return []
 sim=np.clip(r@kernel@q.T,0,1);found=[]
 for aa,bb,score in _candidate_scores(sim):
  a,b=int(aa),int(bb)
  found.append({'a':a,'b':b,'score':float(score),'start':float(query['starts'][a]),'end':float(query['ends'][b-1])})
 found.sort(key=lambda x:x['score'],reverse=True)
 return found

PROBE_POLICY={'version':1,'model':MODEL,'production_enabled':False,
 'method':'CTC_phonetic_posterior_subsequence_DTW_with_PanPhon_kernel',
 'minimum_target_score':.60,'minimum_target_uniqueness_margin':.12,
 'minimum_donor_score':.45,'minimum_donor_specificity':.10,
 'minimum_extra_phone_coverage':.50,'minimum_complete_target_phonetic_ratio':.82,
 'uncertain':'keep','asr_api_requests':0,'tts_requests':0}


def propose_crop(source,query,references,kernel):
 """Experimental conservative replay crop, no dialogue-specific locations.

 A complete requested cognate can be anchored to its source pronunciation.
 Other requests stay intact until a reliable target-language anchor exists.
 """
 from difflib import SequenceMatcher
 from tts_output_guard import phonetic
 from waveform_output_guard import quiet_cut,protected_text
 report={'policy':PROBE_POLICY,'action':'keep','reason':'no_complete_target_anchor','candidates':[]}
 target=next((r for r in references if r.get('own')),None)
 if target is None:return report
 source_text=phonetic(target['original_text']);output_text=phonetic(target['translated_text'])
 ratio=SequenceMatcher(None,source_text,output_text,autojunk=False).ratio() if source_text and output_text else 0.
 report['complete_target_phonetic_ratio']=ratio
 # Merely containing one cognate inside a longer requested sentence is NOT
 # enough: otherwise requested Korean mission/spy words would be erased.
 if ratio<PROBE_POLICY['minimum_complete_target_phonetic_ratio']:return report
 hypotheses=candidates(target['features'],query,kernel)
 if not hypotheses:return report
 anchor=hypotheses[0];report['target']=anchor
 other=max((h['score'] for h in hypotheses if h['b']<=anchor['a'] or h['a']>=anchor['b']),default=0.)
 report['target_uniqueness_margin']=anchor['score']-other
 if anchor['score']<PROBE_POLICY['minimum_target_score'] or anchor['score']-other<PROBE_POLICY['minimum_target_uniqueness_margin']:
  report['reason']='weak_or_ambiguous_target_anchor';return report
 covered=set()
 for ref in references:
  if protected_text(ref['original_text'],target['translated_text']):continue
  choices=candidates(ref['features'],query,kernel)
  if not choices:continue
  baseline=float(np.median([h['score'] for h in choices]))
  selected=[]
  for h in choices:
   # Do not consume even a single target phone for replay removal.
   if not (h['b']<=anchor['a'] or h['a']>=anchor['b']):continue
   if h['score']<PROBE_POLICY['minimum_donor_score'] or h['score']-baseline<PROBE_POLICY['minimum_donor_specificity']:continue
   if any(min(h['b'],c['b'])>max(h['a'],c['a']) for c in selected):continue
   selected.append(h);covered.update(range(h['a'],h['b']))
  for h in selected:report['candidates'].append({**h,'reference':ref['name'],'specificity':h['score']-baseline})
 n=len(query['phones']);sides=[]
 if anchor['a']:sides.append(('prefix',set(range(anchor['a']))))
 if anchor['b']<n:sides.append(('suffix',set(range(anchor['b'],n))))
 if not sides:report['reason']='target_already_covers_all_phones';return report
 report['extra_coverage']={name:len(side&covered)/len(side) for name,side in sides}
 if any(len(side&covered)/len(side)<PROBE_POLICY['minimum_extra_phone_coverage'] for _,side in sides):
  report['reason']='extra_audio_not_sufficiently_explained_by_references';return report
 values,rate=sf.read(source,dtype='float32',always_2d=True);lo=0;hi=len(values)
 if anchor['a']:
  lo=quiet_cut(values,rate,float(query['ends'][anchor['a']-1]),float(query['starts'][anchor['a']]))
 if anchor['b']<n:
  hi=quiet_cut(values,rate,float(query['ends'][anchor['b']-1]),float(query['starts'][anchor['b']]))
 if lo is None or hi is None or hi<=lo:
  report['reason']='no_safe_quiet_cut';return report
 report.update(action='crop_candidate',reason='unique_complete_target_and_reference_explained_extras',
  keep_start=lo/rate,keep_end=hi/rate,removed_seconds=(len(values)-hi+lo)/rate)
 return report
