import base64,json
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from core import config,SR,GEMINI
from speech import transcribe
from gemini_source_pipeline import parse,AUTHORITY


def answer():
    return {'source_language':'ja','speakers':[{'id':0,'nickname':'친구'},{'id':1,'nickname':'동생'}],
        'segments':[{'start':.1,'end':.45,'speaker':0,'text':'[soft, hesitant]本当？'},
                    {'start':.6,'end':.9,'speaker':1,'text':'[bright, quick reply]はい！'}]}


def test_native_audio_pipeline_no_external_asr_and_cache_reuse(tmp_path):
    clean=tmp_path/'voice.wav';sf.write(clean,np.zeros((SR,2),np.float32),SR)
    calls=[]
    def api(payload,key,**kwargs):
        assert payload['model']==GEMINI
        content=payload['messages'][1]['content'];calls.append(content)
        if isinstance(content,list):
            assert base64.b64decode(content[1]['input_audio']['data'])==clean.read_bytes()
            assert 'segments' not in json.loads(content[0]['text'])
            assert payload['response_format']['json_schema']['strict']
            return kwargs['parser'](json.dumps(answer())),{'cost':.001}
        context=json.loads(content)
        assert [r['speaker'] for r in context['regions']]==['0','1']
        return kwargs['parser'](json.dumps({'0':'[soft, hesitant]정말?','1':'[bright, quick reply]네!'})),{}
    cfg={**config(),'key':'test','language':'ko'}
    assert cfg['diarizer']=='gemini-native-audio'
    with patch('scribe.prepare_input',return_value=(clean,'wav',1.)), \
         patch('speech.http_json',side_effect=api), \
         patch('scribe.transcribe_words',side_effect=AssertionError('No ASR')), \
         patch('source_asr.transcribe_words',side_effect=AssertionError('No Fish ASR')), \
         patch('qwen_performance_turns.review',side_effect=AssertionError('No Qwen')):
        result=transcribe(clean,tmp_path/'analysis',tmp_path/'result.json',cfg,lambda *a:None)
        reused=transcribe(clean,tmp_path/'analysis',tmp_path/'result.json',cfg,lambda *a:None)
    assert len(calls)==2  # one audio analysis and one text translation, both cached
    assert result['speaker_names']=={'0':'친구','1':'동생'}
    assert result['boundary_authority']==AUTHORITY
    assert result['asr_used'] is False and result['llm_timestamps'] is True
    assert result['request_counts']['source_asr_api']==0
    assert reused['request_counts']['source_analysis']==0
    assert result['segments'][0]['original_text']=='[soft, hesitant]本当？'
    assert result['segments'][0]['translated_text']=='[soft, hesitant]정말?'
    from reference_timing import refine_references
    from render import prepare_segments
    assert refine_references(clean,result,tmp_path/'reference')[0] is result
    assert not any(row['natural_reference_margin'] for row in prepare_segments(result,SR))


@pytest.mark.parametrize('change', ['out_of_range','unknown_speaker','invalid_name'])
def test_invalid_audio_analysis_does_not_enter_tts(change):
    data=answer()
    if change=='out_of_range':data['segments'][0]['end']=2.
    if change=='unknown_speaker':data['segments'][0]['speaker']=99
    if change=='invalid_name':data['speakers'][0]['nickname']=''
    with pytest.raises(ValueError):parse(json.dumps(data),1.)


def test_fenced_response_preserves_unknown_voice_and_inline_cue_changes():
    data=answer();data['segments'][0]['speaker']=None
    data['segments'][0]['text']='[soft]本当？ [hesitant, falling pitch]あの…'
    result=parse('```json\n'+json.dumps(data)+'\n```',1.)
    assert result['segments'][0]['speaker']=='unknown'
    assert result['segments'][0]['text']==data['segments'][0]['text']
