import copy
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR
from test_studio_workflow import fixture
from studio_workflow import edited_audio,clip_file,digest,base_edge_fades,restore_generation
from clip_split import split_clip


def test_trim_preserves_all_remaining_samples_without_fading_new_boundaries(tmp_path):
    video,store,doc,ctx,_=fixture(tmp_path)
    base=clip_file(video,0,tmp_path);values=np.full((2*SR,2),.5,dtype='float32');sf.write(base,values,SR,subtype='FLOAT');before=digest(base)
    doc['transcript']['segments'][0]['audio_edit']={'position':.2}
    moved,_=sf.read(edited_audio(video,0,doc,ctx,tmp_path),dtype='float32',always_2d=True)
    np.testing.assert_array_equal(moved,values)
    doc['transcript']['segments'][0]['audio_edit']={'trim_start':.2,'trim_end':1.4,'position':.2}
    out=edited_audio(video,0,doc,ctx,tmp_path);trimmed,rate=sf.read(out,dtype='float32',always_2d=True)
    assert len(trimmed)==round(1.2*SR)
    np.testing.assert_array_equal(trimmed,.5)
    np.testing.assert_array_equal(trimmed[round(.1*SR):round(1.1*SR)],.5)
    assert digest(base)==before and edited_audio(video,0,doc,ctx,tmp_path)==out
    doc['transcript']['segments'][0]['audio_edit'].update(pad_start=.2,pad_end=.3)
    padded,_=sf.read(edited_audio(video,0,doc,ctx,tmp_path),dtype='float32',always_2d=True)
    assert len(padded)==round(1.7*SR)
    np.testing.assert_array_equal(padded[round(.2*SR)],.5)
    np.testing.assert_array_equal(padded[round(.3*SR)-1],.5)


def test_split_preserves_playback_samples_and_restores_history_metadata(tmp_path):
    from studio_workflow import load
    video,store,doc,ctx,values=fixture(tmp_path);base=clip_file(video,0,tmp_path);before=digest(base)
    plan={'original':['はい','！'],'translated':['네','!'],'source_time':.4}
    with patch('clip_split.text_plan',return_value=plan):result=split_clip(video,0,{'revision':doc['revision'],'time':.75},tmp_path)
    split,ctx=load(video,tmp_path);left_sha=digest(base)
    assert base_edge_fades(split,0)=={'start':True,'end':False}
    assert base_edge_fades(split,2)=={'start':False,'end':True}
    left,_=sf.read(edited_audio(video,0,split,ctx,tmp_path),dtype='float32',always_2d=True)
    right,_=sf.read(edited_audio(video,2,split,ctx,tmp_path),dtype='float32',always_2d=True)
    raw_left,_=sf.read(base,dtype='float32',always_2d=True);raw_right,_=sf.read(clip_file(video,2,tmp_path),dtype='float32',always_2d=True)
    np.testing.assert_array_equal(left,raw_left);np.testing.assert_array_equal(right,raw_right)
    np.testing.assert_array_equal(np.concatenate([raw_left,raw_right]),values)
    restored,_=restore_generation(video,0,before,tmp_path);assert base_edge_fades(restored,0)=={'start':True,'end':True}
    redone,_=restore_generation(video,0,left_sha,tmp_path);assert base_edge_fades(redone,0)=={'start':True,'end':False}

def test_speed_edit_does_not_add_any_edge_gain_envelope(tmp_path):
    video,store,doc,ctx,_=fixture(tmp_path);base=clip_file(video,0,tmp_path)
    sf.write(base,np.full((2*SR,2),.5,dtype='float32'),SR,subtype='FLOAT')
    doc['transcript']['segments'][0]['audio_edit']={'speed':2.,'position':0}
    def stretch(source,destination,frames):
        sf.write(destination,np.full((frames,2),.5,dtype='float32'),SR,subtype='FLOAT')
    with patch('speedy.stretch',side_effect=stretch):out=edited_audio(video,0,doc,ctx,tmp_path)
    values,rate=sf.read(out,dtype='float32',always_2d=True)
    assert len(values)==SR
    np.testing.assert_array_equal(values,.5)
