import os,subprocess,sys,threading,time
import pytest
from pathlib import Path
from queue_worker import VideoQueue
from storage import Store


@pytest.mark.parametrize('failure',['reported_error','exception','unfinished'])
def test_error_pauses_queue_until_explicit_start_and_survives_restart(tmp_path,failure):
 store=Store(tmp_path)
 for identifier in ['first','next']:store.add(identifier,identifier+'.mp4')
 calls=[]
 def process(job):
  calls.append(job['id'])
  if job['id']=='first':
   if failure=='exception':raise RuntimeError('failed worker')
   if failure=='reported_error':store.update('first',state='error',error='provider failure')
  else:store.update(job['id'],state='done')
 queue=VideoQueue(store,process);queue.start();thread=queue.thread;thread.join(3)
 assert not thread.is_alive() and not queue.running
 assert store.get('first')['state']=='error' and store.get('next')['state']=='queued'
 assert calls==['first'] and queue.failure_id=='first'
 queue.start(automatic=True)
 assert not queue.running and calls==['first']
 restored=VideoQueue(store,process)
 restored.start(editor_only=True,automatic=True)
 assert not restored.running and restored.failure_id=='first'
 restored.start();thread=restored.thread;thread.join(3)
 assert not thread.is_alive() and store.get('next')['state']=='done'
 assert restored.failure_id is None and calls==['first','next']


def test_next_video_starts_only_after_external_wait_and_inflight_is_bounded(tmp_path):
 store=Store(tmp_path)
 for id in ['a','b','c','d']:store.add(id,id+'.mp4')
 entered={id:threading.Event() for id in ['a','b','c','d']};release=threading.Event()
 def process(job):
  entered[job['id']].set();release.wait(5);store.update(job['id'],state='done')
 queue=VideoQueue(store,process);queue.max_inflight=3;queue.start();thread=queue.thread
 try:
  assert entered['a'].wait(2);assert not entered['b'].wait(.2)
  store.update('a',resource_state='external');assert entered['b'].wait(2);assert not entered['c'].wait(.2)
  store.update('b',resource_state='external');assert entered['c'].wait(2)
  store.update('c',resource_state='external');assert not entered['d'].wait(.25)
  assert len(queue.active_jobs)==3
 finally:queue.pause();release.set();thread.join(5)
 assert not queue.running and store.get('d')['state']=='queued'


def test_heavy_local_resources_are_exclusive_across_processes(tmp_path):
 root=Path(__file__).resolve().parents[1];log=tmp_path/'events'
 script=r"""import sys,time,os
from pathlib import Path
import local_resources as r
r.ROOT=Path(sys.argv[1]);r._state=lambda value:None
os.environ['DUBLY_VIDEO_ID']=sys.argv[2]
with r.local_resources():
 with Path(sys.argv[3]).open('a') as f:f.write('enter '+sys.argv[2]+'\n')
 time.sleep(.2)
 with Path(sys.argv[3]).open('a') as f:f.write('exit '+sys.argv[2]+'\n')
"""
 processes=[subprocess.Popen([sys.executable,'-c',script,str(tmp_path),id,str(log)],cwd=root) for id in ['a','b']]
 try:
  assert all(p.wait(timeout=8)==0 for p in processes)
 finally:
  for p in processes:
   if p.poll() is None:p.kill();p.wait()
 rows=log.read_text().splitlines();assert rows[0].startswith('enter ')
 assert rows[1]=='exit '+rows[0].split()[1]
 assert rows[2].startswith('enter ') and rows[3]=='exit '+rows[2].split()[1]


def test_same_video_threads_share_the_stage_without_reducing_fit_concurrency(tmp_path,monkeypatch):
 import local_resources as resources
 states=[];entered=threading.Event();release=threading.Event()
 monkeypatch.setenv('DUBLY_VIDEO_ID','sample');monkeypatch.setattr(resources,'ROOT',tmp_path);monkeypatch.setattr(resources,'_state',states.append)
 def one():
  with resources.local_resources():entered.set();release.wait(2)
 with resources.local_resources():
  thread=threading.Thread(target=one);thread.start();assert entered.wait(1)
 assert states[-1]=='local'
 release.set();thread.join(2);assert states[-1]=='external'



def test_cancel_queued_item_keeps_files_and_other_jobs_queued(tmp_path):
 from unittest.mock import patch
 from fastapi.testclient import TestClient
 import app as api
 import uuid
 store=Store(tmp_path);queue=VideoQueue(store);ids=[str(uuid.uuid4()) for _ in range(2)]
 for id in ids:store.add(id,'sample.mp4')
 cache=tmp_path/'work/videos'/ids[0]/'cached.wav';cache.parent.mkdir(parents=True);cache.write_bytes(b'keep')
 with patch.object(api,'store',store),patch.object(api,'queue',queue),TestClient(api.app) as client:
  path='/api/videos/'+ids[0]+'/cancel'
  assert client.post(path).status_code==422
  assert client.post(path,json={'confirmed':False}).status_code==422
  response=client.post(path,json={'confirmed':True});assert response.status_code==200 and response.json()['state']=='cancelled'
  assert client.post(path,json={'confirmed':True}).status_code==200
  assert store.get(ids[1])['state']=='queued' and cache.read_bytes()==b'keep'
  assert not queue.running


def test_individual_stop_ends_only_the_selected_worker_and_its_child(tmp_path):
 import json
 store=Store(tmp_path)
 for id in ['a','b','c']:store.add(id,id+'.mp4')
 project=Path(__file__).resolve().parents[1]
 child="import sys,time;from pathlib import Path\nwhile True:\n with Path(sys.argv[1]+'-heartbeat').open('a') as f:f.write('x')\n time.sleep(.03)\n"
 script="import sys,subprocess,time\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(project))+")\nfrom storage import Store\nvideo=sys.argv[1]\nStore(Path.cwd()).update(video,resource_state='external')\nsubprocess.Popen([sys.executable,'-c',"+repr(child)+",video])\nwhile True:time.sleep(.1)\n"
 (tmp_path/'pipeline.py').write_text(script,'utf8');queue=VideoQueue(store);queue.max_inflight=2;queue.start();thread=queue.thread
 try:
  deadline=time.monotonic()+8
  while time.monotonic()<deadline and not (tmp_path/'b-heartbeat').exists():time.sleep(.03)
  assert (tmp_path/'a-heartbeat').exists() and (tmp_path/'b-heartbeat').exists()
  assert queue.cancel('b')=='cancelled'
  before=(tmp_path/'b-heartbeat').stat().st_size;active=(tmp_path/'a-heartbeat').stat().st_size;time.sleep(.2)
  assert (tmp_path/'b-heartbeat').stat().st_size==before
  assert (tmp_path/'a-heartbeat').stat().st_size>active and store.get('a')['state']=='running'
 finally:queue.stop();thread.join(8)
 assert not queue.running


def test_separation_child_reuses_cached_audio_without_inference(tmp_path,monkeypatch):
 import numpy as np,soundfile as sf
 from separation_process import prepare
 original=tmp_path/'original.wav';stage=tmp_path/'stage';stage.mkdir();clean=stage/'dialogue.wav'
 sf.write(original,np.zeros((4410,2),np.float32),44100);sf.write(clean,np.zeros((4410,2),np.float32),44100)
 before=clean.read_bytes();monkeypatch.setenv('DUBLY_VIDEO_ID','not-a-real-video')
 result,background=prepare(original,stage,tmp_path,{'separator':'melband-kim','background':'residual'},lambda *args:None)
 assert result==clean and background is None and clean.read_bytes()==before

def test_default_queue_waits_for_video_completion_even_during_external_api(tmp_path):
 store=Store(tmp_path)
 for identifier in ['first','second']:store.add(identifier,identifier+'.mp4')
 entered={k:threading.Event() for k in ['first','second']};release=threading.Event()
 def process(job):
  entered[job['id']].set()
  if job['id']=='first':release.wait(5)
  store.update(job['id'],state='done')
 queue=VideoQueue(store,process);assert queue.max_inflight==1
 queue.start();thread=queue.thread
 try:
  assert entered['first'].wait(2)
  store.update('first',resource_state='external')
  assert not entered['second'].wait(.3)
  release.set();assert entered['second'].wait(2)
 finally:queue.pause();release.set();thread.join(5)

def test_render_lane_starts_during_local_processing_without_extra_main_video(tmp_path):
 store=Store(tmp_path)
 for key in ['main','next','watch','render']:store.add(key,key+'.mp4')
 for key,action in [('watch','watch'),('render','render')]:
  store.update(key,state='ready');store.schedule_editor(key,action)
 entered={k:threading.Event() for k in ['main','next','watch','render']}
 release={k:threading.Event() for k in entered}
 def process(job):
  entered[job['id']].set();release[job['id']].wait(5);store.update(job['id'],state='done')
 queue=VideoQueue(store,process);queue.start();thread=queue.thread
 try:
  assert entered['main'].wait(2) and entered['watch'].wait(2)
  assert not entered['next'].wait(.2) and not entered['render'].wait(.2)
  release['watch'].set();assert entered['render'].wait(2)
  assert not entered['next'].is_set()
  release['main'].set();assert entered['next'].wait(2)
 finally:
  queue.pause()
  for event in release.values():event.set()
  thread.join(5)


def test_render_resource_context_does_not_wait_for_heavy_stage(tmp_path,monkeypatch):
 import local_resources as resources
 monkeypatch.setenv('DUBLY_VIDEO_ID','watch');monkeypatch.setenv('DUBLY_INDEPENDENT_RENDER','1')
 monkeypatch.setattr(resources,'_try_lock',lambda handle:(_ for _ in ()).throw(AssertionError('Render must not acquire heavy lock')))
 with resources.local_resources():pass


def test_reset_resumes_waiting_queue_and_preserves_active_work_and_files(tmp_path):
 store=Store(tmp_path)
 for key in ['failed','active','waiting']:store.add(key,key+'.mp4')
 store.update('failed',state='error',error='saved failure')
 cached=tmp_path/'work/keep.txt';cached.write_text('keep')
 entered=threading.Event();release=threading.Event();calls=[]
 def process(job):
  calls.append(job['id'])
  if job['id']=='active':entered.set();release.wait(4)
  store.update(job['id'],state='done')
 queue=VideoQueue(store,process);queue.start();thread=queue.thread
 try:
  assert entered.wait(2)
  queue.pause();queue.failure_id='failed'
  queue.reset()
  assert queue.thread is thread and queue.active_jobs['active'].is_alive()
  assert not queue.stop_requested and queue.failure_id is None
  assert store.get('failed')['state']=='error' and cached.read_text()=='keep'
  release.set();thread.join(4)
  assert calls==['active','waiting'] and store.get('waiting')['state']=='done'
 finally:release.set();queue.stop();thread.join(4)


def test_reset_endpoint_resumes_queue(tmp_path):
 from unittest.mock import patch
 from fastapi.testclient import TestClient
 import app as api
 store=Store(tmp_path);queue=VideoQueue(store)
 with patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'reset') as reset,TestClient(api.app) as client:
  assert client.post('/api/queue/reset').status_code==200
  reset.assert_called_once_with()
