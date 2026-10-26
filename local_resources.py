"""One heavy local video stage at a time; external API waits may overlap."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import os,threading,time
from core import ROOT

_condition=threading.Condition();_users=0;_acquiring=False;_handle=None

def _state(value):
    video_id=os.environ.get('DUBLY_VIDEO_ID')
    if video_id:
        from storage import Store
        Store().update(video_id,resource_state=value)

def _try_lock(handle):
    if os.name=='nt':
        import msvcrt
        try:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1);return True
        except OSError:return False
    import fcntl
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
    except BlockingIOError:return False

def _unlock(handle):
    if os.name=='nt':
        import msvcrt
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
    else:
        import fcntl
        fcntl.flock(handle,fcntl.LOCK_UN)

@contextmanager
def local_resources():
    global _users,_acquiring,_handle
    if not os.environ.get('DUBLY_VIDEO_ID') or os.environ.get('DUBLY_INDEPENDENT_RENDER')=='1':
        yield;return
    leader=False
    with _condition:
        while _acquiring:_condition.wait()
        if _users:_users+=1
        else:_acquiring=True;leader=True;_state('local_wait')
    if leader:
        handle=None
        try:
            path=ROOT/'work/local-video-resource.lock';path.parent.mkdir(parents=True,exist_ok=True)
            handle=path.open('a+b')
            if path.stat().st_size==0:handle.write(b'0');handle.flush()
            while not _try_lock(handle):time.sleep(.1)
            with _condition:
                _handle=handle;_users=1;_acquiring=False;_state('local');_condition.notify_all()
        except BaseException:
            if handle:handle.close()
            with _condition:_acquiring=False;_condition.notify_all()
            raise
    try:yield
    finally:
        with _condition:
            _users-=1
            if not _users:
                try:_unlock(_handle)
                finally:_handle.close();_handle=None;_state('external');_condition.notify_all()

def local_stage(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        with local_resources():return function(*args,**kwargs)
    return wrapped
