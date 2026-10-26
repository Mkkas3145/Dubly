"""One bounded reusable local MFA process; stop it with its video pipeline."""
from __future__ import annotations
import atexit,json,os,queue,signal,subprocess,threading
from pathlib import Path
from core import ROOT


class WorkerError(RuntimeError):
    def __init__(self,message,fatal=False):super().__init__(message);self.fatal=fatal


class MfaClient:
    def __init__(self,args,env=None,log=None):
        self.messages=queue.Queue();self.lock=threading.Lock();self.sequence=0
        log=log or ROOT/'logs'/f'mfa-resident-{os.getpid()}.log'
        Path(log).parent.mkdir(parents=True,exist_ok=True);self.log=open(log,'a',encoding='utf-8')
        try:
            self.process=subprocess.Popen(args,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                encoding='utf-8',errors='replace',bufsize=1,start_new_session=os.name!='nt',
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except Exception:self.log.close();raise
        self.reader=threading.Thread(target=self._read,daemon=True,name='mfa-output');self.reader.start()
        atexit.register(self.close)

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith('MFA_RESULT '):
                    try:self.messages.put(json.loads(line[len('MFA_RESULT '):]))
                    except ValueError:self.messages.put(None)
        finally:self.messages.put(None)

    def request(self,audio,text,output,timeout=120):
        with self.lock:
            if self.process.poll() is not None:raise WorkerError('MFA worker exited',fatal=True)
            self.sequence+=1
            request={'id':self.sequence,'audio':str(audio),'text':text,'output':str(output)}
            try:
                self.process.stdin.write(json.dumps(request,ensure_ascii=True)+'\n');self.process.stdin.flush()
                result=self.messages.get(timeout=timeout)
            except (OSError,queue.Empty) as error:
                self.close(force=True);raise WorkerError('MFA worker timed out or disconnected',fatal=True) from error
            if result is None or result.get('id')!=self.sequence:
                self.close(force=True);raise WorkerError('MFA worker protocol failed',fatal=True)
            if not result.get('ok'):raise WorkerError(result.get('error','MFA alignment failed'),fatal=result.get('fatal',False))
            return result

    def _kill(self):
        if os.name=='nt':
            subprocess.run(['taskkill','/PID',str(self.process.pid),'/T','/F'],capture_output=True,timeout=5,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        else:
            try:os.killpg(self.process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        if self.process.poll() is None:self.process.kill()

    def close(self,force=False):
        if self.log.closed:return
        try:
            if self.process.poll() is None:
                if force:self._kill()
                self.process.stdin.close()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self._kill();self.process.wait(timeout=5)
            self.reader.join(timeout=1)
        finally:
            self.process.stdout.close()
            if not self.process.stdin.closed:self.process.stdin.close()
            self.log.close();atexit.unregister(self.close)
