from pathlib import Path
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import render
import studio_workflow as workflow
from core import SR
from mix_levels import TTS_GAIN,TTS_GAIN_DB,VOLUME_VERSION


def test_constant_gain_preserves_dynamics_background_and_kept_source(tmp_path):
    original=np.full((3*SR,2),.12,np.float32)
    clean=np.full_like(original,.10)
    bed=original-clean
    tts=np.zeros((SR,2),np.float32)
    tts[:SR//2]=.01;tts[SR//2:]=.10
    for name,audio in [('original',original),('clean',clean),('bed',bed),('0000-fitted',tts),('0001-fitted',tts[:SR//4])]:
        sf.write(tmp_path/(name+'.wav'),audio,SR,subtype='FLOAT')
    rows=[{'lo':SR,'hi':SR+3*SR//4},{'lo':SR+3*SR//4,'hi':2*SR,'keep_original':True}]
    for background in [None,tmp_path/'bed.wav']:
        render.mix(tmp_path/'original.wav',tmp_path/'clean.wav',rows,tmp_path,tmp_path/'mixed.wav',lambda *a:None,background=background)
        result,_=sf.read(tmp_path/'mixed.wav',dtype='float32',always_2d=True)
        np.testing.assert_allclose(result[SR:SR+SR//2]-bed[SR:SR+SR//2],tts[:SR//2]*TTS_GAIN,atol=2e-8)
        np.testing.assert_allclose(result[SR+SR//2:SR+3*SR//4]-bed[SR+SR//2:SR+3*SR//4],tts[SR//2:3*SR//4]*TTS_GAIN,atol=2e-8)
        np.testing.assert_array_equal(result[SR+3*SR//4:2*SR],original[SR+3*SR//4:2*SR])
        np.testing.assert_array_equal(result[:SR//2],original[:SR//2])
    unchanged,_=sf.read(tmp_path/'0000-fitted.wav',dtype='float32',always_2d=True)
    np.testing.assert_array_equal(unchanged,tts)


def test_audition_cache_adds_gain_once_and_invalidates_with_source(tmp_path,monkeypatch):
    source=tmp_path/'base.wav';audio=np.linspace(-.2,.2,1000,dtype=np.float32)[:,None]
    sf.write(source,audio,SR,subtype='FLOAT')
    monkeypatch.setattr(workflow,'edited_audio',lambda *a:source)
    monkeypatch.setattr(workflow,'paths',lambda *a:(tmp_path,None,None))
    before=workflow.digest(source)
    result=workflow.listening_audio('id',0,{}, {},tmp_path)
    repeated=workflow.listening_audio('id',0,{}, {},tmp_path)
    assert result==repeated and workflow.digest(source)==before
    y,_=sf.read(result,dtype='float32',always_2d=True)
    np.testing.assert_allclose(y,audio*TTS_GAIN,atol=1e-8)
    sf.write(source,audio*.5,SR,subtype='FLOAT')
    assert workflow.listening_audio('id',0,{}, {},tmp_path)!=result
    assert TTS_GAIN_DB==4 and VOLUME_VERSION==5
