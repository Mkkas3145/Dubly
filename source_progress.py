"""Keep long provider waits visible without inventing percentage progress."""
from contextlib import contextmanager
import threading,time


class Progress:
    def __init__(self,callback,interval=2.):
        self.callback=callback;self.interval=interval;self.lock=threading.Lock()
        self.percent=0.;self.label='준비';self.started=time.monotonic();self.stop=threading.Event()

    def update(self,percent,label):
        with self.lock:self.percent=max(self.percent,float(percent));self.label=label
        self.emit()

    def emit(self):
        with self.lock:self.callback(self.percent,f'{self.label} · 경과 {int(time.monotonic()-self.started)}초')

    @contextmanager
    def running(self):
        def pulse():
            while not self.stop.wait(self.interval):self.emit()
        worker=threading.Thread(target=pulse,name='asr-progress',daemon=True);worker.start()
        try:yield self
        finally:self.stop.set();worker.join();self.emit()
