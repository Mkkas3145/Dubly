import numpy as np
import soundfile as sf
import pytest
from mms_alignment_signal import contrast,profile
from mms_alignment_search import same_location


def test_contrast_distinguishes_new_voice_from_ongoing_quiet_residue():
    levels=np.full(500,.0002,np.float32)
    levels[90:120]=.0004
    levels[250:300]=.000005
    levels[300:330]=.0004
    assert contrast(levels,6.,6.6)>contrast(levels,1.8,2.4)+.5
    assert contrast(levels*.01,6.,6.6)==pytest.approx(contrast(levels,6.,6.6))


def test_profile_uses_original_clock_and_never_changes_source(tmp_path):
    rate=16000;path=tmp_path/'native.wav'
    x=np.zeros((rate,2),np.float32);x[8000:9600]=.1
    sf.write(path,x,rate,subtype='FLOAT');before=path.read_bytes()
    values=profile(path)
    assert len(values)==50 and np.all(values[:25]==0)
    assert values[25]==pytest.approx(.1)
    assert path.read_bytes()==before
    assert contrast(np.zeros(100),.5,1.)==0


def test_boundary_uncertainty_is_not_a_different_word_occurrence():
    assert same_location({'start':3.,'end':4.},{'start':3.4,'end':4.02})
    assert not same_location({'start':3.,'end':4.},{'start':7.,'end':8.})
    assert not same_location({'start':3.,'end':4.},{'start':3.9,'end':4.1})
