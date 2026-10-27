import copy
import numpy as np
import soundfile as sf
from source_voice_release import recover_ends
from core import SR
from render import reference_audio


def speech(values,a,b,rate=SR):
    n=round((b-a)*rate);t=np.arange(n)/rate
    values[round(a*rate):round(a*rate)+n]=(.2*np.sin(2*np.pi*180*t))[:,None]


def row(start,end):return {'id':0,'start':start,'end':end,'aligned_words':[{'word':'voice','start':start,'end':end}]}


def test_active_last_vowel_is_extended_until_measured_quiet_not_fixed_padding(tmp_path):
    x=np.zeros((3*SR,2),np.float32);speech(x,.5,1.4)
    p=tmp_path/'speech.wav';sf.write(p,x,SR,subtype='FLOAT')
    original=[row(.5,.9)];before=copy.deepcopy(original)
    result,audit=recover_ends(p,original)
    assert 1.4<=result[0]['end']<=1.47 and result[0]['start']==.5
    assert result[0]['lexical_end']==.9 and audit['extended_count']==1
    assert original==before


def test_already_quiet_end_does_not_grow_to_later_sound(tmp_path):
    x=np.zeros((3*SR,2),np.float32);speech(x,.5,.9);speech(x,1.5,2.)
    p=tmp_path/'speech.wav';sf.write(p,x,SR,subtype='FLOAT')
    result,audit=recover_ends(p,[row(.5,1.)])
    assert result[0]['end']==1. and audit['extended_count']==0


def test_continuous_voice_never_consumes_the_next_speaker(tmp_path):
    x=np.zeros((3*SR,2),np.float32);speech(x,.5,2.5)
    p=tmp_path/'speech.wav';sf.write(p,x,SR,subtype='FLOAT')
    rows=[row(.5,.9),{**row(1.2,2.5),'id':1}]
    result,audit=recover_ends(p,rows)
    assert result[0]['end']<=1.2 and result[1]['start']==1.2
    assert not audit['checks'][0]['extended']


def test_reference_extraction_preserves_edges_and_adds_no_silence(tmp_path):
    x=np.full((SR,2),.5,np.float32);source=tmp_path/'voice.wav';dest=tmp_path/'reference.wav'
    sf.write(source,x,SR,subtype='FLOAT');reference_audio(source,{'lo':0,'hi':SR},dest)
    result,rate=sf.read(dest,dtype='float32')
    assert len(result)==SR and result[0]==.5 and result[-1]==.5
