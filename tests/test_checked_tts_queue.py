from checked_tts_queue import run
from tts_output_guard import POLICY
import threading
def test_stable_check_batches_and_original_order():
    checked=[];updates=[]
    def checker(values):checked.append(list(values));return values
    results=run(list(range(35)),lambda i,s:i,lambda v:v,checker,lambda *v:updates.append(v))
    assert results==list(range(35))
    assert sorted(checked,key=lambda v:v[0])==[list(range(16)),list(range(16,32)),list(range(32,35))]
    assert updates[-1]==(35,0,0,0,0,0)

def test_local_waveform_checks_begin_individually_and_preserve_output_order():
    checked=[]
    result=run(list(range(23)),lambda i,s:i,lambda v:v,
        lambda parts:checked.append(parts) or parts,lambda *v:None,batch_size=1)
    assert result==list(range(23))
    assert sorted(checked)==[[i] for i in range(23)]

def test_generation_continues_while_scribe_checks_and_fitting_runs():
    checking=threading.Event();later_generation=threading.Event()
    def generator(i,s):
        if i>=32:later_generation.set()
        return i
    def checker(values):
        checking.set();assert later_generation.wait(5);return values
    assert run(list(range(50)),generator,lambda v:v,checker,lambda *v:None)==list(range(50))
