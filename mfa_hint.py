"""Cached local word/phone evidence; no per-segment ASR or LLM requests."""
from __future__ import annotations
import hashlib,json,os,tempfile,threading,unicodedata
from pathlib import Path
import numpy as np
import soundfile as sf
from core import ROOT,SR,atomic_json
from mfa_client import MfaClient,WorkerError

VERSION=4  # Alignment semantics unchanged; existing successful caches stay valid.
ENVIRONMENT=ROOT/'.mfa-env'
MODEL_ROOT=ROOT/'models/mfa'
MODELS=('acoustic','dictionary','g2p')
_LOCK=threading.Lock();_CLIENT=None;_SESSION_FAILURE=None


def model_path(kind):return MODEL_ROOT/f"pretrained_models/{kind}/korean_mfa.{'dict' if kind=='dictionary' else 'zip'}"


def runtime_path(path):
    if os.name=='nt' and Path(path).exists():
        import ctypes
        buffer=ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(str(path),buffer,len(buffer)):return buffer.value
    return str(path)


def runtime_environment():
    env=os.environ.copy()
    paths=[ENVIRONMENT,ENVIRONMENT/'Library/mingw-w64/bin',ENVIRONMENT/'Library/usr/bin',ENVIRONMENT/'Library/bin',ENVIRONMENT/'Scripts',ENVIRONMENT/'bin']
    bundled_cuda=ROOT.parent/'runtime/python/Lib/site-packages/torch/lib'
    if bundled_cuda.is_dir():paths.append(bundled_cuda)
    env['PATH']=os.pathsep.join(map(runtime_path,paths))+os.pathsep+env.get('PATH','')
    env.update(MFA_ROOT_DIR=runtime_path(MODEL_ROOT),PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CONDA_PREFIX=str(ENVIRONMENT))
    for key in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):env.pop(key,None)
    return env


def runtime_ready():return (ENVIRONMENT/'python.exe').is_file() and all(model_path(k).is_file() for k in MODELS)


def close_worker():
    global _CLIENT,_SESSION_FAILURE
    with _LOCK:
        if _CLIENT is not None:_CLIENT.close()
        _CLIENT=None;_SESSION_FAILURE=None


def normalized(text):return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())


def parse_alignment(data,text,duration):
    tiers=data.get('tiers',{})
    def entries(name):
        tier=next((v for k,v in tiers.items() if k==name or k.endswith(' - '+name)),{})
        rows=tier.get('entries',[]) if isinstance(tier,dict) else [];result=[];last=0.
        for a,b,label in rows:
            a,b=float(a),float(b)
            if not np.isfinite([a,b]).all() or a<last-.002 or b<=a or a<-.002 or b>duration+.03:raise ValueError('Invalid MFA interval')
            result.append({'start':max(0.,a),'end':min(duration,b),'label':str(label)});last=b
        return result
    words,phones=entries('words'),entries('phones')
    spoken=[w for w in words if w['label'].casefold() not in ('','<eps>','sil','sp') and normalized(w['label'])]
    if not spoken or not phones or any(w['label'].casefold() in ('<unk>','spn') for w in spoken):raise ValueError('Incomplete MFA alignment')
    if normalized(' '.join(w['label'] for w in spoken))!=normalized(text):raise ValueError('MFA transcript mismatch')
    return {'words':spoken,'phones':phones}


def alignment_hint(values,text,language,cache):
    global _CLIENT,_SESSION_FAILURE
    if language!='ko':return {'status':'unsupported_language','version':VERSION}
    if not runtime_ready():return {'status':'unavailable','version':VERSION}
    manifest=MODEL_ROOT/'installed.json';fingerprint=manifest.read_bytes() if manifest.is_file() else b''
    identity=hashlib.sha256(values.astype('<f4').tobytes()+f'{VERSION}:{SR}:{text}'.encode()+fingerprint).hexdigest();cache=Path(cache)
    def cached():
        try:
            value=json.loads(cache.read_text(encoding='utf-8'))
            if value.get('identity')==identity:return value
        except (OSError,ValueError):pass
    saved=cached()
    if saved:return saved
    with _LOCK:
        saved=cached()
        if saved:return saved
        if _SESSION_FAILURE:return {'status':'failed','identity':identity,'version':VERSION,'reason':_SESSION_FAILURE}
        try:
            with tempfile.TemporaryDirectory(prefix='mfa-',dir=cache.parent) as temporary:
                folder=Path(temporary);audio=folder/'utterance.wav'
                sf.write(audio,values.mean(axis=1),SR,subtype='PCM_16')
                if _CLIENT is None:_CLIENT=MfaClient([str(ENVIRONMENT/'python.exe'),'-u',str(ROOT/'mfa_worker.py')],env=runtime_environment())
                stats=_CLIENT.request(audio,text,folder/'utterance.json')
                data=json.loads((folder/'utterance.json').read_text(encoding='utf-8'))
                evidence=parse_alignment(data,text,len(values)/SR)
                report={'status':'aligned','identity':identity,'version':VERSION,'runtime':'resident_mfa','runtime_stats':stats,**evidence}
        except (OSError,ValueError,TypeError,KeyError,RuntimeError) as error:
            report={'status':'failed','identity':identity,'version':VERSION,'reason':f'{type(error).__name__}: {error}'}
            if isinstance(error,WorkerError) and error.fatal:
                _SESSION_FAILURE=report['reason']
                if _CLIENT is not None:_CLIENT.close();_CLIENT=None
        atomic_json(cache,report);return report


def review_pause(pause,hint):
    if not hint or hint.get('status')!='aligned':return 'unavailable'
    lo,hi=pause['lo']/SR,pause['hi']/SR;a,b=lo+.025,hi-.025
    if any(w['start']<a and w['end']>b for w in hint.get('words',[])):return 'within_word'
    lexical=[p for p in hint.get('phones',[]) if p['label'].casefold() not in ('','sil','sp','<eps>')]
    overlap=sum(max(0.,min(b,p['end'])-max(a,p['start'])) for p in lexical)
    if overlap>max(.04,(b-a)*.35):return 'phoneme_conflict'
    for left,right in zip(hint.get('words',[]),hint.get('words',[])[1:]):
        if left['end']<=hi+.04 and right['start']>=lo-.04 and left['end']<=right['start']+.002:
            if max(lo,left['end'])<=min(hi,right['start'])+.03:return 'corroborated'
    return 'uncertain'
