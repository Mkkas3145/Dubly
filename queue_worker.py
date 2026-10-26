from __future__ import annotations
import subprocess
import os
import signal
import sys
import threading
import time
import json


UTILITY_ACTIONS=('watch','render','mix')

class VideoQueue:
    """Process one video at a time, including external API waits."""
    def __init__(self, store, processor=None):
        self.store=store
        self.processor=processor or self.process
        self.lock=threading.RLock()
        self.thread=None
        self.stop_requested=False
        self.cancel_requested=False
        self.active_id=None
        self.process_handle=None
        self.editor_only=False
        self.active_jobs={}
        self.process_handles={}
        self.max_inflight=1
        self.cancelled_ids=set()
        self.failure_path=self.store.root/'work/queue-failure-pause.json'
        self.failure_id=json.loads(self.failure_path.read_text('utf8')).get('video_id') if self.failure_path.exists() else None

    @property
    def running(self):
        with self.lock:
            return bool(self.thread and self.thread.is_alive())

    def start(self,editor_only=False,*,automatic=False):
        with self.lock:
            if automatic and self.failure_id:return
            if self.failure_id:
                from core import atomic_json
                atomic_json(self.failure_path,{'video_id':None})
                self.failure_id=None;self.stop_requested=False
            if self.running:
                return
            self.stop_requested=False
            self.cancel_requested=False
            self.editor_only=editor_only
            self.thread=threading.Thread(target=self.run,daemon=True,name='video-queue')
            self.thread.start()

    def reset(self):
        """Resume queued work without cancelling active jobs or retrying failures."""
        with self.lock:
            # An older editor-only worker must also see the full waiting queue.
            self.editor_only=False
            self.start()

    def pause(self):
        with self.lock:
            self.stop_requested=True

    def stop(self):
        with self.lock:
            self.stop_requested=True
            self.cancel_requested=True
            processes=list(self.process_handles.values())
            if self.process_handle and self.process_handle not in processes:processes.append(self.process_handle)
            for process in processes:self._terminate(process)

    @staticmethod
    def _terminate(process):
        if process.poll() is not None:return
        if os.name=='nt':
            result=subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
            if result.returncode and process.poll() is None:raise RuntimeError('Failed to stop the job. Try again.')
        else:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:raise RuntimeError('Could not confirm that the job stopped. Try again.') from None

    def cancel(self,video_id):
        worker=None
        with self.lock:
            job=self.store.get(video_id)
            if job is None:raise KeyError(video_id)
            if job['state'] in {'done','ready','cancelled'}:return job['state']
            if job['state'] not in {'uploaded','queued','running','error'}:raise RuntimeError('The current job cannot be stopped.')
            worker=self.active_jobs.get(video_id)
            if worker:
                self.cancelled_ids.add(video_id)
                process=self.process_handles.get(video_id)
                if process:self._terminate(process)
            self.store.update(video_id,state='cancelled',phase='중단됨',detail='',error=None)
        if worker:worker.join(5)
        return self.store.get(video_id)['state']

    def delete(self,video_id):
        with self.lock:
            if video_id in self.active_jobs:raise RuntimeError('Stop the video processing job first.')
            self.store.delete(video_id)

    def retry(self,video_id):
        with self.lock:
            job=self.store.get(video_id)
            if job is None:raise KeyError(video_id)
            if job['state'] not in {'error','cancelled'} or video_id in self.active_jobs:
                raise RuntimeError('Only failed or cancelled videos can be retried.')
            from core import atomic_json
            editor=self.store.root/'work/videos'/video_id/'editor'
            if (editor/'draft.json').exists() and (editor/'context.json').exists():
                atomic_json(editor/'resume-completed.json',{'preserve_completed':True})
            self.store.update(video_id,state='queued',phase='재시도 대기',phase_progress=0,progress=0,detail='',error=None)

    def _finish(self,job):
        error=None
        try:self.processor(job)
        except Exception:error='처리 프로세스가 종료되었습니다. 재시도해 주세요.'
        finally:
            with self.lock:
                current=self.store.get(job['id'])
                if current and (self.cancel_requested or job['id'] in self.cancelled_ids) and current['state'] not in {'done','ready'}:
                    self.store.update(job['id'],state='cancelled',phase='중단됨',detail='',error=None)
                elif current and (error or current['state']=='running'):
                    self.store.update(job['id'],state='error',phase='실패',detail='',error=error or '처리 결과를 확인하지 못했습니다.')
                current=self.store.get(job['id'])
                if current and current['state']=='error':
                    from core import atomic_json
                    self.failure_id=job['id'];self.stop_requested=True
                    atomic_json(self.failure_path,{'video_id':job['id']})
                self.active_jobs.pop(job['id'],None);self.process_handles.pop(job['id'],None);self.cancelled_ids.discard(job['id'])
                if current:
                    from studio_concurrency import advance
                    advance(self.store,current)
                self.active_id=next(iter(self.active_jobs),None)
                self.process_handle=next(iter(self.process_handles.values()),None)

    def run(self):
        while True:
            with self.lock:
                if self.stop_requested and not self.active_jobs:self.thread=None;return
                all_active=[self.store.get(id) for id in self.active_jobs]
                active=[j for j in all_active if j and j.get('workflow_action') not in UTILITY_ACTIONS]
                utilities=[j for j in all_active if j and j.get('workflow_action') in UTILITY_ACTIONS]
                can_start=not active or (len(active)<self.max_inflight and all(j.get('resource_state')=='external' for j in active))
                job=None
                if not self.stop_requested:
                    if not utilities:job=self.store.claim(editor_only=self.editor_only,actions=UTILITY_ACTIONS)
                    if job is None and can_start:job=self.store.claim(editor_only=self.editor_only,exclude_actions=UTILITY_ACTIONS)
                if job:
                    worker=threading.Thread(target=self._finish,args=(job,),daemon=True,name='video-'+job['id'])
                    self.active_jobs[job['id']]=worker;self.active_id=self.active_id or job['id'];worker.start()
                elif not self.active_jobs:self.thread=None;return
            time.sleep(.1)

    def process(self, job):
        root=self.store.root
        (root/'logs').mkdir(exist_ok=True)
        with (root/'logs'/f"{job['id']}.log").open('a',encoding='utf-8') as log:
            with self.lock:
                if self.cancel_requested or job['id'] in self.cancelled_ids:return
                args=[sys.executable,'-u',str(root/'pipeline.py'),job['id']]
                if job.get('workflow_action','full')!='full':args.extend(['--action',job['workflow_action']])
                if job.get('editor_index') is not None:args.extend(['--segment',str(job['editor_index'])])
                process=subprocess.Popen(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'DUBLY_VIDEO_ID':job['id'],'DUBLY_INDEPENDENT_RENDER':'1' if job.get('workflow_action') in UTILITY_ACTIONS else '0'},
                    start_new_session=os.name!='nt')
                self.process_handles[job['id']]=process
                self.process_handle=self.process_handle or process
            code=process.wait()
        if code and not self.cancel_requested and job['id'] not in self.cancelled_ids and self.store.get(job['id'])['state'] not in {'error','cancelled'}:
            raise RuntimeError('Worker exited')
