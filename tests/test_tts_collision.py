import json
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR
from tts_collision import relax_collisions


def test_next_clip_silence_can_prevent_unnecessary_speedup(tmp_path):
    sf.write(tmp_path/'0001-fitted.wav',np.r_[np.zeros(SR//2),np.ones(SR//2)*.1],SR)
    (tmp_path/'0000-tts.mp3').write_bytes(b'raw')
    (tmp_path/'0000.json').write_text(json.dumps({'tempo':2.,'collision_budget_frames':SR}),encoding='utf8')
    (tmp_path/'0000-generation.json').write_text('{}',encoding='utf8')
    segments=[{'lo':0,'hi':SR,'start':0.,'end':1.},{'lo':SR,'hi':2*SR,'start':1.,'end':2.}]
    def fit(prepared,cfg):
        assert prepared['collision_budget']==SR+SR//2
        return {'index':0,'tempo':1.,'output_frames':SR+100}
    with patch('tts_collision.speech_hints',return_value=[(.5,1.)]),patch('render.reference_audio'),patch('render.fit_speech',side_effect=fit) as fitter:
        result=relax_collisions(segments,[{'index':0,'tempo':2.}],'clean',tmp_path,{})
    assert fitter.call_count==1 and result[0]['tempo']==1.
    assert result[0]['collision_deadline_method']=='next_tts_audible_onset'


def test_no_leading_silence_does_not_relax_a_real_collision(tmp_path):
    sf.write(tmp_path/'0001-fitted.wav',np.ones(SR)*.1,SR)
    (tmp_path/'0000-tts.mp3').write_bytes(b'raw')
    (tmp_path/'0000.json').write_text(json.dumps({'tempo':2.,'collision_budget_frames':SR}),encoding='utf8')
    (tmp_path/'0000-generation.json').write_text('{}',encoding='utf8')
    segments=[{'lo':0,'hi':SR},{'lo':SR,'hi':2*SR}]
    with patch('tts_collision.speech_hints',return_value=[(0.,1.)]),patch('render.fit_speech') as fitter:
        result=relax_collisions(segments,[{'index':0,'tempo':2.}],'clean',tmp_path,{})
    fitter.assert_not_called();assert result[0]['tempo']==2.
