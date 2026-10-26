import numpy as np
import pytest
import soundfile as sf
from core import SR
from qwen_boundaries import correct_edges


@pytest.mark.parametrize('offset',[0.,2.71,11.23])
def test_voice_change_uses_intervening_pause_not_syllable_gap(tmp_path,offset):
    samples=np.zeros(round((offset+4)*SR),np.float32)
    def voiced(a,b):
        lo=round((offset+a)*SR);hi=round((offset+b)*SR)
        samples[lo:hi]=.2*np.sin(np.arange(hi-lo)*2*np.pi*170/SR)
    voiced(.3,1.6);voiced(2.05,2.35);voiced(2.40,2.9)
    source=tmp_path/'voices.wav';sf.write(source,samples,SR)
    rows=[{'id':0,'start':offset+.4,'end':offset+2.3,'speaker':'voice_a','text':'first'},
        {'id':1,'start':offset+2.4,'end':offset+3.,'speaker':'voice_b','text':'second'}]
    result,audit=correct_edges(source,rows)
    # A short intra-word pause is closer to the estimated boundary. It must
    # lose to the sustained pause that separates the two complete voices.
    assert offset+1.55<result[0]['end']<offset+1.7
    assert offset+1.95<result[1]['start']<offset+2.1
    assert rows[0]['end']==offset+2.3
    assert audit[0]['following_voice_boundary']['action']=='split_at_shared_pause'
    assert [r['speaker'] for r in result]==['voice_a','voice_b']


def test_no_quiet_gap_does_not_invent_an_acoustic_split(tmp_path):
    source=tmp_path/'continuous.wav'
    sf.write(source,.2*np.sin(np.arange(3*SR)*2*np.pi*180/SR),SR)
    rows=[{'start':.2,'end':1.4,'speaker':'a'},{'start':1.5,'end':2.8,'speaker':'b'}]
    result,audit=correct_edges(source,rows)
    assert result[0]['end']<=result[1]['start']
    assert audit[0]['following_voice_boundary']['action']=='no_complete_quiet_gap_keep_provisional'


@pytest.mark.parametrize('offset',[0.,7.37])
def test_late_onset_anchor_recovers_complete_body_and_weak_initial_sound(tmp_path,offset):
    values=np.zeros(round((offset+5)*SR),np.float32)
    def voiced(a,b,amplitude):
        lo=round((offset+a)*SR);hi=round((offset+b)*SR)
        values[lo:hi]=amplitude*np.sin(np.arange(hi-lo)*2*np.pi*190/SR)
    voiced(.9,.96,.008);voiced(1.06,2.,.2)
    path=tmp_path/'late.wav';sf.write(path,values,SR)
    rows=[{'start':offset+1.55,'end':offset+2.2,'speaker':'voice'}]
    result,_=correct_edges(path,rows)
    assert offset+.85<result[0]['start']<offset+.92
    assert offset+1.99<result[0]['end']<offset+2.08


def test_short_turn_with_anchor_entirely_in_silence_recovers_nearby_speech(tmp_path):
    values=np.zeros(4*SR,np.float32)
    lo,hi=round(1.4*SR),round(1.62*SR)
    values[lo:hi]=.2*np.sin(np.arange(hi-lo)*2*np.pi*180/SR)
    path=tmp_path/'short.wav';sf.write(path,values,SR)
    result,audit=correct_edges(path,[{'start':1.85,'end':2.05,'speaker':'voice'}])
    assert 1.35<result[0]['start']<1.42 and 1.61<result[0]['end']<1.7
    assert audit[0]['acoustic_body']['silent_anchor_recovered']


def test_tail_does_not_follow_faint_noise_or_separate_later_voice(tmp_path):
    values=np.zeros(5*SR,np.float32)
    lo,hi=round(1.1*SR),round(2.1*SR)
    values[lo:hi]=.2*np.sin(np.arange(hi-lo)*2*np.pi*190/SR)
    lo,hi=round(2.1*SR),round(3.1*SR)
    values[lo:hi]=.001*np.sin(np.arange(hi-lo)*2*np.pi*250/SR)
    lo,hi=round(3.5*SR),round(3.8*SR)
    values[lo:hi]=.15*np.sin(np.arange(hi-lo)*2*np.pi*280/SR)
    path=tmp_path/'tail.wav';sf.write(path,values,SR)
    result,_=correct_edges(path,[{'start':1.2,'end':3.1,'speaker':'voice'}])
    assert result[0]['end']<2.4
