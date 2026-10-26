"""Production wiring: verification must run before fitting; no cloud calls."""
import json
from unittest.mock import patch
import numpy as np
import soundfile as sf
import render
from core import SR,config
from tts_output_guard import POLICY

def test_real_generate_check_fit_path_uses_cropped_audio_and_then_reuses_cache(tmp_path):
    clean=tmp_path/'clean.wav';voice=.1*np.sin(np.arange(3*SR)*.04)
    sf.write(clean,np.repeat(voice[:,None],2,axis=1),SR,subtype='FLOAT')
    segments=[{'start':0.,'end':1.,'lo':0,'hi':SR,'original_text':'スパイ、','translated_text':'[嬉しい]스파이,',
        '_shared_reference_path':str(clean),'_shared_reference_text':'スパイ、ミッション、',
        '_guard_reference_texts':[{'original_text':'ミッション、','translated_text':'미션,'}]}]
    folder=tmp_path/'pieces';folder.mkdir()
    calls=[]
    def generate(reference,segment,language,key,destination):
        audio=np.sin(np.arange(2*SR)*.04).astype('float32')*.1;audio[round(.7*SR):SR]=0
        # SoundFile dispatches by extension, so the fixture is WAV bytes at the mp3 path.
        sf.write(destination,audio,SR,format='WAV',subtype='FLOAT')
        calls.append('tts');return {'generated_seconds':2.,'tts_policy':render.tts_policy('ko')}
    def asr(*args,**kwargs):
        calls.append('asr');return {'words':[{'word':'スパイ','start':.1,'end':.7},{'word':'ミッション','start':1.,'end':1.8}]}
    def fit(source,destination,frames,reference,text,language,**kwargs):
        assert text=='스파이,' # Emotion tags must not be pronounced/aligned as words.
        data,sr=sf.read(source,always_2d=True);assert .7<=len(data)/sr<=1.
        sf.write(destination,np.repeat(data,2,axis=1),sr,subtype='FLOAT');calls.append('fit')
        return {'fit_version':render.FIT_VERSION,'timing_method':render.TIMING_POLICY,'output_frames':len(data),
            'speech_window_frames':[0,SR],'collision_budget_frames':None}
    cfg={**config(),'key':'test','tts_output_guard':'scribe-v2'}
    with patch('render.request_audio',side_effect=generate),patch('tts_output_guard.transcribe_words',side_effect=asr), \
         patch('render.bounded_timing_audio',side_effect=fit):
        result=render.synthesize_all(segments,clean,folder,cfg,lambda *v:None)
    assert calls==['tts','asr','fit']
    assert result[0]['tts_output_check']['action']=='cropped' and result[0]['tts_output_guard_policy']==POLICY
    with patch('render.request_audio',side_effect=AssertionError('cached TTS')), \
         patch('tts_output_guard.transcribe_words',side_effect=AssertionError('cached check')), \
         patch('render.bounded_timing_audio',side_effect=AssertionError('cached fitting')):
        reused=render.synthesize_all(segments,clean,folder,cfg,lambda *v:None)
    assert reused==result

def test_scribe_failure_preserves_generated_audio_with_explicit_review(tmp_path):
    from tts_output_guard import check_batch
    path=tmp_path/'tts.wav';sf.write(path,.1*np.sin(np.arange(SR)*.05),SR)
    prepared={'index':0,'generated':path,'folder':tmp_path,'generation':{},'segment':{'original_text':'はい','translated_text':'네'}}
    with patch('tts_output_guard.transcribe_words',side_effect=RuntimeError('HTTP 500')):
        checked=check_batch([prepared],{'key':'unused'},tmp_path/'checks')[0]
    assert '_guard_error' not in checked
    assert checked['generation']['tts_output_check']['action']=='keep_uncertain'
    assert checked['generated']==path
