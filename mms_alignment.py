"""Run MMS-FA once per source stage; cached chunks resume without re-inference."""
from pathlib import Path
import json,subprocess,time,sys
from core import ROOT,atomic_json

MODEL='Meta/MMS-FA'
VERSION=2

def align_batch(jobs,language,folder,progress=lambda *a:None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    request=folder/'request.json';output=folder/'response.json'
    # Source stage owns this folder; concurrent jobs use distinct video folders.
    atomic_json(request,{'jobs':jobs,'language':language})
    state=output.with_suffix('.progress.json');state.unlink(missing_ok=True)
    python=ROOT/'.venv/Scripts/python.exe'
    if not python.exists():python=Path(sys.executable)
    with (folder/'worker.log').open('wb') as log:
        process=subprocess.Popen([str(python),str(ROOT/'mms_align_worker.py'),str(request),str(output)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        last=None
        while process.poll() is None:
            if state.exists():
                try:
                    value=json.loads(state.read_text('utf8'))
                    if value!=last:
                        last=value;done=value['done'];total=value['total']
                        progress(97 if value['phase']=='quality' else 95 if value['phase']=='ctc' else 90*done/max(1,total),f'MMS-FA 원본 정렬 {done}/{total}개 구간'+(' · 모델 준비' if value['phase']=='load' else (' · 전체 원문 정렬' if value['phase']=='ctc' else ' · 정렬 신뢰도 검사' if value['phase']=='quality' else '')))
                except (OSError,ValueError):pass
            time.sleep(.25)
        if process.returncode:
            raise RuntimeError(f'MMS-FA source alignment failed; see {folder / "worker.log"}')
    result=json.loads(output.read_text('utf8'))
    if len(result['results'])!=len(jobs):raise RuntimeError('MMS-FA did not return every source alignment window.')
    return result
