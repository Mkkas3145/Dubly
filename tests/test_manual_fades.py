import math
from unittest.mock import patch
import numpy as np
import soundfile as sf
import pytest
from audio_edges import apply_dialogue_fades,fade_lengths
from core import SR
from studio_workflow import edited_audio,clip_file,clip_info,digest,validate_edit,save,load
from test_studio_workflow import fixture,body


def test_only_explicit_fades_change_gain_and_never_change_length():
    x=np.ones((1000,2),np.float32)
    np.testing.assert_array_equal(apply_dialogue_fades(x,1000),x)
    y=apply_dialogue_fades(x,1000,.2,.3)
    assert y.shape==x.shape and y[0,0]==0 and y[-1,0]==0
    np.testing.assert_array_equal(y[200:700],1)
    assert y[100,0]==pytest.approx(math.sin(100/199*math.pi/2))
    np.testing.assert_array_equal(x,1)
    assert fade_lengths(.8,.8,1)==(.5,.5)


def test_manual_fades_are_saved_and_reopened_without_touching_paid_base(tmp_path):
    video,store,doc,ctx,_=fixture(tmp_path);base=clip_file(video,0,tmp_path);before=digest(base)
    payload=body(doc);payload['segments'][0]['audio_edit']={'fade_in':.3,'fade_out':.4}
    doc,_=save(video,payload,tmp_path);reopened,_=load(video,tmp_path)
    assert reopened['transcript']['segments'][0]['audio_edit']['fade_in']==.3
    assert reopened['transcript']['segments'][0]['audio_edit']['fade_out']==.4
    assert digest(base)==before


def test_trimmed_content_fades_exclude_explicit_silence_padding(tmp_path):
    video,store,doc,ctx,_=fixture(tmp_path);base=clip_file(video,0,tmp_path)
    sf.write(base,np.full((2*SR,2),.5,np.float32),SR,subtype='FLOAT');before=digest(base)
    doc['transcript']['segments'][0]['audio_edit']={'trim_start':.2,'trim_end':1.4,'pad_start':.1,'pad_end':.2,'fade_in':.2,'fade_out':.3}
    y,rate=sf.read(edited_audio(video,0,doc,ctx,tmp_path),dtype='float32',always_2d=True)
    assert len(y)==round(1.5*SR)
    np.testing.assert_array_equal(y[:round(.1*SR)],0)
    np.testing.assert_array_equal(y[round(1.3*SR):],0)
    assert y[round(.1*SR),0]==0 and y[round(1.3*SR)-1,0]==0
    np.testing.assert_array_equal(y[round(.3*SR):round(1.*SR)],.5)
    assert digest(base)==before


def test_fade_seconds_are_measured_after_speed_processing(tmp_path):
    video,store,doc,ctx,_=fixture(tmp_path);base=clip_file(video,0,tmp_path)
    sf.write(base,np.full((2*SR,2),.5,np.float32),SR,subtype='FLOAT')
    doc['transcript']['segments'][0]['audio_edit']={'speed':2,'fade_in':.2,'fade_out':.2}
    def stretch(source,destination,frames):sf.write(destination,np.full((frames,2),.5,np.float32),SR,subtype='FLOAT')
    with patch('speedy.stretch',side_effect=stretch):y,rate=sf.read(edited_audio(video,0,doc,ctx,tmp_path),dtype='float32',always_2d=True)
    assert len(y)==SR
    assert y[round(.1*SR),0]==pytest.approx(.5*math.sin(round(.1*SR)/(round(.2*SR)-1)*math.pi/2),abs=1e-6)
    assert y[round(.2*SR),0]==.5


@pytest.mark.parametrize('value',[-1,None,True,float('nan'),'1'])
def test_invalid_fade_values_fail_cleanly(value):
    info={'duration':2.,'sha256':'test','pauses':[]}
    with pytest.raises(ValueError):validate_edit({'fade_in':value},info,10.)


def test_overlapping_requests_normalize_to_selected_body_duration():
    info={'duration':2.,'sha256':'test','pauses':[]}
    result=validate_edit({'trim_start':.5,'trim_end':1.5,'speed':2,'fade_in':.8,'fade_out':.8},info,10.)
    assert result['fade_in']==pytest.approx(.25) and result['fade_out']==pytest.approx(.25)
