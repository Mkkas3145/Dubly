"""Align the exact requested script, with acoustic and independent-ASR checks."""
from pathlib import Path
import hashlib,json,math,secrets,subprocess,time,unicodedata,urllib.request
from filelock import FileLock
import numpy as np
from core import ROOT,LANGUAGES,atomic_json

VERSION=1
MODEL='Qwen/Qwen3-ForcedAligner-0.6B'
RUNTIME=ROOT/'work/tts-forced-runtime'

def key(text):return ''.join(c.casefold() for c in unicodedata.normalize('NFKC',text) if c.isalnum())

def health(state):
    req=urllib.request.Request(f'http://127.0.0.1:{state["port"]}/health',headers={'Authorization':'Bearer '+state['token']})
    with urllib.request.urlopen(req,timeout=3) as response:return json.load(response)

def preload():
    RUNTIME.mkdir(parents=True,exist_ok=True);state_file=RUNTIME/'state.json'
    with FileLock(str(RUNTIME/'startup.lock'),timeout=120):
        if state_file.exists():
            try:
                state=json.loads(state_file.read_text('utf8'))
                if state.get('version')==VERSION and health(state)['version']==VERSION:return state
            except (OSError,ValueError,KeyError):pass
        token=secrets.token_hex(24)
        with (RUNTIME/'worker.log').open('ab') as log:
            process=subprocess.Popen([str(ROOT/'.align-venv/Scripts/python.exe'),str(ROOT/'tts_forced_worker.py'),str(state_file),token],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(900):
            if process.poll() is not None:raise RuntimeError('Qwen forced aligner startup failed: work/tts-forced-runtime/worker.log')
            if state_file.exists():
                state=json.loads(state_file.read_text('utf8'))
                if state.get('token')==token:return state
            time.sleep(.1)
        raise RuntimeError('Qwen forced aligner startup timed out')

def align(audio,texts,language,folder):
    audio=Path(audio);cache=Path(folder)/'qwen-script-alignment.json'
    identity={'version':VERSION,'model':MODEL,'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),
        'texts':texts,'language':language}
    if cache.exists():
        old=json.loads(cache.read_text('utf8'))
        if old.get('identity')==identity:return old
    if not any(key(t) for t in texts):return {'words':[],'identity':identity}
    state=preload();body={'audio':str(audio.resolve()),'text':' '.join(texts),'language':LANGUAGES[language]}
    req=urllib.request.Request(f'http://127.0.0.1:{state["port"]}/align',data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+state['token'],'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as response:result=json.load(response)
    result['identity']=identity;atomic_json(cache,result);return result

def locations(texts,words,values,rate,recognized=()):
    """Map exact script characters, never a fuzzy recognizer transcript, to rows."""
    from source_alignment import compact_match
    targets=[key(t) for t in texts]
    if ''.join(targets)!=''.join(key(w['word']) for w in words):raise RuntimeError('Forced alignment script token mismatch')
    owners=[]
    for i,w in enumerate(words):owners.extend([i]*len(key(w['word'])))
    cursor=0;result=[];previous=0.;duration=len(values)/rate
    for target in targets:
        ids=list(dict.fromkeys(owners[cursor:cursor+len(target)]));cursor+=len(target)
        if not target:result.append({'matched':False,'coverage':0.,'word_indices':[],'words':[],'reason':'no_lexical_dubbing_text'});continue
        spans=[words[i] for i in ids if words[i]['end']>words[i]['start']]
        reason=None
        if not spans:reason='collapsed_forced_timestamps'
        else:
            start,end=spans[0]['start'],spans[-1]['end']
            if not all(math.isfinite(x) for x in (start,end)) or start<previous-.002 or not 0<=start<end<=duration+.02:
                reason='invalid_or_overlapping_forced_timestamps'
            else:
                block=values[round(start*rate):min(len(values),round(end*rate))]
                if not len(block) or np.sqrt(np.mean(block**2))<max(1e-6,float(np.sqrt(np.mean(values**2)))*.025):reason='no_acoustic_support'
                # Reject a confident contradictory lexical span, particularly
                # invented short final turns placed inside a different word.
                heard=''.join(key(w['word']) for w in recognized if min(end,w['end'])-max(start,w['start'])>.02)
                scripts={unicodedata.name(c,'').split(' ')[0] for c in target}
                comparable=sum(unicodedata.name(c,'').split(' ')[0] in scripts for c in heard)
                # A recognizer occasionally changes Korean into Chinese text;
                # that is not valid contradictory evidence about pronunciation.
                if heard and comparable/len(heard)>=.8 and len(compact_match(target,heard))/len(target)<.25:
                    reason='independent_asr_contradiction'
        if reason:result.append({'matched':False,'coverage':0.,'word_indices':[],'words':[],'reason':reason});continue
        result.append({'matched':True,'coverage':1.,'start':start,'end':end,'words':[words[i] for i in ids],
            'word_indices':ids,'alignment_method':'exact_script_qwen_forced'});previous=end
    return result
