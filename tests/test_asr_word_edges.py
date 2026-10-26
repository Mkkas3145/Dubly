import numpy as np
import soundfile as sf
from asr_word_edges import repair


def wave(tmp_path,spans,duration=8):
    rate=16000;values=np.zeros((duration*rate,2),np.float32)
    for a,b,volume in spans:
        lo,hi=round(a*rate),round(b*rate)
        values[lo:hi]=volume*np.sin(np.arange(hi-lo)*.1)[:,None]
    p=tmp_path/'voice.wav';sf.write(p,values,rate,subtype='FLOAT');return p


def test_large_false_prefix_is_removed_without_rewriting_transcript_or_interior_pause(tmp_path):
    clean=wave(tmp_path,[(2,2.01,.3),(6.3,6.8,.1)])
    word={'word':'Bonjour.','start':0.,'end':6.88,'speaker':'0'}
    result,audit=repair(clean,[word],detector=lambda *a:[(6.3,6.8)])
    assert 6.1<result[0]['start']<6.3 and result[0]['end']==6.88
    assert result[0]['word']==word['word'] and result[0]['speaker']=='0'
    assert word['start']==0 and audit[0]['original_start']==0


def test_genuine_early_speech_and_internal_pauses_are_preserved(tmp_path):
    clean=wave(tmp_path,[(.2,.4,.01),(3,3.5,.1),(6.3,6.8,.1)])
    result,_=repair(clean,[{'word':'long phrase','start':0.,'end':7.8}],detector=lambda *a:[(.2,.4),(3.,3.5),(6.3,6.8)])
    assert result[0]['start']==0 and 6.8<result[0]['end']<7


def test_weak_onset_before_detected_voice_is_protected(tmp_path):
    clean=wave(tmp_path,[(5.9,6.1,.0004),(6.1,6.6,.1)])
    result,_=repair(clean,[{'word':'quiet onset','start':0.,'end':7.9}],detector=lambda *a:[(6.1,6.6)])
    assert result[0]['start']<=5.95 and result[0]['end']>=6.6


def test_short_interjection_is_not_filtered_or_sent_to_edge_detector(tmp_path):
    word={'word':'え？','start':1.,'end':1.06,'speaker':'2'}
    def fail(*a):raise AssertionError('Do not retime short valid words')
    result,_=repair(wave(tmp_path,[(1,1.06,.1)]),[word],detector=fail)
    assert result==[word]


def test_uncertain_or_failed_vad_preserves_words(tmp_path):
    word={'word':'hello','start':0.,'end':7.}
    clean=wave(tmp_path,[(6,6.5,.1)])
    assert repair(clean,[word],detector=lambda *a:[])[0]==[word]
    def broken(*a):raise RuntimeError('VAD unavailable')
    result,audit=repair(clean,[word],detector=broken)
    assert result==[word] and audit[0]['method']=='preserve_asr_word_after_edge_detector_failure'
