import base64,json
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from core import config,SR,GEMINI
from gemini_source_pipeline import transcribe
from gemini_source_pipeline import parse,AUTHORITY


def answer():
    return {'source_language':'ja','speakers':[{'id':0,'nickname':'친구'},{'id':1,'nickname':'동생'}],
        'segments':[{'speaker':0,'original_text':'[soft, hesitant]本当？','translated_text':'[부드럽게, 망설이며]정말?'},
                    {'speaker':1,'original_text':'[bright, quick reply]はい！','translated_text':'[bright, quick reply]네!'}]}


def test_native_audio_pipeline_no_external_asr_and_cache_reuse(tmp_path):
    clean=tmp_path/'voice.wav';sf.write(clean,np.zeros((SR,2),np.float32),SR)
    calls=[]
    def api(payload,key,**kwargs):
        assert payload['model']==GEMINI
        assert payload['max_tokens']==524288
        content=payload['messages'][1]['content'];calls.append(content)
        assert isinstance(content,list), 'Never send a transcript back for a second translation call'
        assert base64.b64decode(content[1]['input_audio']['data'])==clean.read_bytes()
        assert 'segments' not in json.loads(content[0]['text'])
        schema=payload['response_format']['json_schema']
        assert schema['strict']
        assert set(schema['schema']['properties']['segments']['items']['properties'])=={'speaker','original_text','translated_text'}
        return kwargs['parser'](json.dumps(answer())),{'cost':.001}
    cfg={**config(),'key':'test','language':'ko'}
    assert cfg['diarizer']=='gemini-forced'
    with patch('scribe.prepare_input',return_value=(clean,'wav',1.)), \
         patch('speech.http_json',side_effect=api), \
         patch('source_forced_alignment.align_rows',side_effect=lambda audio,rows,*args,**kwargs:([{**r,'start':a,'end':b} for r,(a,b) in zip(rows,[(.1,.45),(.6,.9)])],[])), \
         patch('scribe.transcribe_words',side_effect=AssertionError('No ASR')), \
         patch('source_asr.transcribe_words',side_effect=AssertionError('No Fish ASR')), \
         patch('qwen_performance_turns.review',side_effect=AssertionError('No Qwen')), \
         patch('mai_pipeline.translate',side_effect=AssertionError('No separate translation')), \
         patch('fish_pipeline.fixed_translations',side_effect=AssertionError('No fixed Fish ID contract')):
        result=transcribe(clean,tmp_path/'analysis',tmp_path/'result.json',cfg,lambda *a:None)
        reused=transcribe(clean,tmp_path/'analysis',tmp_path/'result.json',cfg,lambda *a:None)
    assert len(calls)==1  # one audio request returns everything; second execution reuses it
    assert result['speaker_names']=={'0':'친구','1':'동생'}
    assert result['boundary_authority']==AUTHORITY
    assert result['asr_used'] is False and result['llm_timestamps'] is False
    assert result['request_counts']['source_asr_api']==0
    assert reused['request_counts']['combined_audio']==0
    assert result['request_counts']['separate_translation']==0
    assert result['segments'][0]['original_text']=='本当？'
    assert result['segments'][0]['translated_text']=='[부드럽게, 망설이며]정말?'
    from reference_timing import refine_references
    from render import prepare_segments
    assert refine_references(clean,result,tmp_path/'reference')[0] is result
    assert not any(row['natural_reference_margin'] for row in prepare_segments(result,SR))


@pytest.mark.parametrize('change', ['missing_source','unknown_speaker','invalid_name','missing_translation'])
def test_invalid_audio_analysis_does_not_enter_tts(change):
    data=answer()
    if change=='missing_source':data['segments'][0]['original_text']=''
    if change=='unknown_speaker':data['segments'][0]['speaker']=99
    if change=='invalid_name':data['speakers'][0]['nickname']=''
    if change=='missing_translation':data['segments'][0].pop('translated_text')
    with pytest.raises(ValueError):parse(json.dumps(data),1.)


def test_fenced_response_preserves_unknown_voice_and_inline_cue_changes():
    data=answer();data['segments'][0]['speaker']=None
    data['segments'][0]['original_text']='[soft]本当？ [hesitant, falling pitch]あの…'
    result=parse('```json\n'+json.dumps(data)+'\n```',1.)
    assert result['segments'][0]['speaker']=='unknown'
    assert result['segments'][0]['text']=='本当？ あの…'


def test_untimed_streamed_dialogue_count_updates_before_completion(tmp_path):
    import io
    import speech
    from gemini_source_pipeline import source
    clean=tmp_path/'voice.wav';sf.write(clean,np.zeros((SR,2),np.float32),SR)
    data=answer()
    first,second=[json.dumps(row,ensure_ascii=False) for row in data['segments']]
    fragments=['{"source_language":"ja","speakers":'+json.dumps(data['speakers'])+',"segments":[',
        first[:-1],'},',second,']}']
    chunks=[]
    for fragment in fragments:
        chunks.append('data: '+json.dumps({'choices':[{'delta':{'content':fragment}}]})+'\n\n')
    chunks.append('data: '+json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}]})+'\n\n')
    chunks.append('data: [DONE]\n\n')
    updates=[]
    with patch('scribe.prepare_input',return_value=(clean,'wav',1.)), \
         patch('speech.urllib.request.urlopen',return_value=io.BytesIO(''.join(chunks).encode())), \
         patch('speech.time.monotonic',return_value=10.):
        result=source(clean,tmp_path/'source',{**config(),'key':'test'},lambda p,d:updates.append((p,d)))
    one=[p for p,d in updates if '1개 대사 수신' in d]
    two=[p for p,d in updates if '2개 대사 수신' in d]
    assert one and two and 0<one[0]==two[0]<100
    assert updates[-1][0]==100 and len(result['segments'])==2
    assert all(' / ' not in d for _,d in updates)
    counter=speech.SegmentStreamCounter()
    assert counter.feed('{"segments":['+first[:-1])==0
    assert counter.completed_audio_end==0
    assert counter.feed('}')==1
    assert counter.completed_audio_end==0


def test_production_dispatch_uses_gemini_and_mms_instead_of_external_asr(tmp_path):
    from speech import transcribe as production
    cfg=config()
    with patch('gemini_source_pipeline.transcribe',return_value={'model':'test'}) as native, \
         patch('scribe_pipeline.transcribe',side_effect=AssertionError('No external ASR')):
        assert production('audio',tmp_path,tmp_path/'out.json',cfg,lambda *a:None)=={'model':'test'}
    assert native.call_count==1
    from gemini_source_pipeline import POLICY,PROMPT
    assert POLICY['alignment_model']=='Meta/MMS-FA'
    assert POLICY['cue_language']=='target_language' and POLICY['output_cues_only']
    assert 'ONLY in translated_text' in PROMPT

def test_untimed_source_preserves_audio_order_and_never_requests_timestamps():
    from gemini_source_pipeline import response_format,PROMPT
    schema=response_format(10800)['json_schema']['schema']
    segment=schema['properties']['segments']['items']
    assert set(segment['properties'])==set(segment['required'])=={'speaker','original_text','translated_text'}
    data=answer();parsed=parse(data,10800)
    assert all('start' not in r and 'end' not in r for r in parsed['segments'])
    assert [r['text'] for r in parsed['segments']]==['本当？','はい！']
    assert 'Do NOT output start/end timestamps' in PROMPT
