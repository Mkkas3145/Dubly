import numpy as np
import pytest
from tts_forced_alignment import locations

def test_exact_script_recovers_recognizer_spelling_difference():
    values=np.ones((2000,2),dtype=np.float32)*.1
    words=[{'word':'아르스','start':.1,'end':.6},{'word':'님','start':.6,'end':.8}]
    heard=[{'word':'알스님','start':.1,'end':.8}]
    result=locations(['…','아르스 님.'],words,values,1000,heard)
    assert result[0]['reason']=='no_lexical_dubbing_text'
    assert result[1]['matched'] and result[1]['start']==.1

def test_forced_short_word_inside_different_recognized_word_is_rejected():
    result=locations(['네.'],[{'word':'네','start':.4,'end':.48}],np.ones((1000,2))*.1,1000,
        [{'word':'이건','start':.1,'end':.7}])
    assert not result[0]['matched']
    assert result[0]['reason']=='independent_asr_contradiction'

def test_silence_and_collapsed_timestamps_are_not_fabricated_speech():
    assert not locations(['hello'],[{'word':'hello','start':.1,'end':.5}],np.zeros((1000,2)),1000)[0]['matched']
    assert not locations(['hello'],[{'word':'hello','start':.1,'end':.1}],np.ones((1000,2)),1000)[0]['matched']

def test_script_mismatch_is_rejected():
    with pytest.raises(RuntimeError,match='token mismatch'):
        locations(['hello'],[{'word':'goodbye','start':.1,'end':.5}],np.ones((1000,2)),1000)
