"""Start/preload the encoder once; reuse it across requests and videos."""
import io,json,secrets,subprocess,time,urllib.request
from pathlib import Path
import numpy as np
from filelock import FileLock
from core import ROOT
from embedding_service import VERSION

RUNTIME=ROOT/'work/embedding-runtime'


def health(state):
    request=urllib.request.Request(f'http://127.0.0.1:{state["port"]}/health',headers={'Authorization':'Bearer '+state['token']})
    return json.load(urllib.request.urlopen(request,timeout=3))


def preload():
    RUNTIME.mkdir(parents=True,exist_ok=True);state_file=RUNTIME/'state.json'
    with FileLock(str(RUNTIME/'startup.lock'),timeout=90):
        if state_file.exists():
            try:
                state=json.loads(state_file.read_text(encoding='utf8'))
                if state.get('version')==VERSION and health(state)['version']==VERSION:return state
            except (OSError,ValueError,KeyError):pass
        token=secrets.token_hex(24)
        with (RUNTIME/'worker.log').open('ab') as log:
            process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'embedding_service.py'),str(state_file),token],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(900):
            if process.poll() is not None:raise RuntimeError('Semantic matching model initialization failed. See work/embedding-runtime/worker.log.')
            if state_file.exists():
                state=json.loads(state_file.read_text(encoding='utf8'))
                if state.get('token')==token and health(state)['version']==VERSION:return state
            time.sleep(.1)
        raise RuntimeError('Semantic matching model initialization timed out.')


def encode(texts,progress=lambda *args:None):
    state=preload();progress(10,'미리 로드한 의미 매칭 모델 재사용')
    request=urllib.request.Request(f'http://127.0.0.1:{state["port"]}/encode',
        data=json.dumps({'texts':texts},ensure_ascii=False).encode(),
        headers={'Authorization':'Bearer '+state['token'],'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as response:result=np.load(io.BytesIO(response.read()),allow_pickle=False)
    progress(80,'다국어 의미 비교 완료');return result
