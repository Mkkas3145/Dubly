from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR
from collapsed_source_onsets import repair


def test_collapsed_first_word_recovers_only_exact_verified_unclaimed_island(tmp_path):
    values=np.random.default_rng(42).normal(0,.0003,(4*SR,1)).astype(np.float32)
    t=np.arange(round(.2*SR))/SR
    values[SR:SR+len(t),0]+=.04*np.sin(2*np.pi*220*t)
    values[2*SR:3*SR,0]+=.2*np.sin(2*np.pi*220*np.arange(SR)/SR)
    path=tmp_path/'clean.wav';sf.write(path,values,SR)
    rows=[{'start':.1,'end':.6,'text':'previous'},{'start':2.,'end':3.,'text':'え？してないよ。'}]
    words=[{'word':'え','start':2.,'end':2.}]
    with patch('scribe.transcribe_words',return_value={'text':'え？','words':[{'word':'え','probability':.99}]}):
        result,audit=repair(path,rows,words,tmp_path/'check',{'key':'test'},'ja')
    assert .95<result[1]['start']<1.02 and audit[0]['verified']
    assert result[1]['text']==rows[1]['text'] and result[1]['end']==3.


def test_disagreeing_asr_does_not_automatically_reassign_unclaimed_audio(tmp_path):
    values=np.zeros((4*SR,1),np.float32)
    values[SR:round(1.2*SR)]=.04
    values[2*SR:3*SR]=.2
    path=tmp_path/'clean.wav';sf.write(path,values,SR)
    rows=[{'start':.1,'end':.6,'text':'previous'},{'start':2.,'end':3.,'text':'え？してないよ。'}]
    with patch('scribe.transcribe_words',return_value={'text':'other','words':[{'word':'other','probability':.99}]}):
        result,audit=repair(path,rows,[{'word':'え','start':2.,'end':2.}],tmp_path/'check',{'key':'test'},'ja')
    assert result==rows and not audit[0]['verified']
