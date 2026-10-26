import json
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from core import config,SR,QWEN,GEMINI
import fish_qwen_pipeline as f


def test_speaker_instruction_explicitly_prohibits_echoing_input_fields():
    assert 'OUTPUT ONLY one JSON object' in f.SPEAKER_PROMPT
    assert 'NEVER echo input region objects, start/end times, transcript text' in f.SPEAKER_PROMPT
    assert 'NEVER use dialogue meaning, conversational context' in f.SPEAKER_PROMPT
    assert 'There is NO limit on the number of speakers.' in f.SPEAKER_PROMPT
    assert f.parse_map('{"0":37,"1":108}',[0,1],'speakers')=={'0':37,'1':108}


def test_sentence_and_voice_boundaries_are_not_merged_by_wrong_fish_identity():
    raw={'words':[{'start':.1,'end':.4,'word':'はい！','speaker':0},
        {'start':.5,'end':.8,'word':'あ、','speaker':0},
        {'start':.85,'end':1.2,'word':'本当？','speaker':0},
        {'start':1.3,'end':1.6,'word':'ええ。','speaker':1}]}
    rows=f.regions(raw,2.)
    assert [r['text'] for r in rows]==['はい！','あ、本当？','ええ。']
    assert [r['id'] for r in rows]==[0,1,2]


def test_translation_omission_keeps_original_and_speakers_only_allow_integer_ids():
    assert f.parse_map('{"0":"대사"}',[0,1],'translations')=={'0':'대사','1':None}
    assert f.parse_map('{"0":2}',[0,1],'speakers')=={'0':2,'1':None}
    assert f.parse_map('[{"id":1,"speaker":2},{"id":0,"speaker":null}]',[0,1],'speakers')=={'0':None,'1':2}
    assert f.parse_map('[{"id":0,"start":99,"end":100,"speaker":4}]',[0],'speakers')=={'0':4}
    assert f.parse_map('[{"id":0,"start":"unused","end":null,"speaker":4}]',[0],'speakers')=={'0':4}
    assert f.parse_map('[{"region_id":0,"speaker_id":4}]',[0],'speakers')=={'0':4}
    with pytest.raises(ValueError):f.parse_map('[{"id":0,"region_id":1,"speaker":4}]',[0,1],'speakers')
    with pytest.raises(ValueError):f.parse_map('[{"id":0,"speaker":4,"text":"extra"}]',[0],'speakers')
    for response in ['{"0":{"speaker":2,"end":1}}','{"0":true}','{"0":"speaker_2"}','{"0":1,"0":2}','{"3":1}']:
        with pytest.raises(ValueError):f.parse_map(response,[0,1],'speakers')
    with pytest.raises(ValueError):f.parse_map('[{"id":0,"speaker":1},{"id":0,"speaker":2}]',[0,1],'speakers')


def test_dispatch_three_roles_separate_caches_and_no_extra_source_asr(tmp_path):
    from speech import transcribe
    from reference_timing import refine_references
    from render import prepare_segments
    audio=tmp_path/'dialogue.wav';sf.write(audio,np.zeros(3*SR,np.float32),SR)
    encoded=tmp_path/'input.mp3';encoded.write_bytes(b'audio')
    raw={'signature':{'model':f.ASR,'source':'synthetic'},'language':'ja','words':[
        {'start':.2,'end':.7,'word':'[surprised]本当？'},
        {'start':1.,'end':1.5,'word':'はい！'},
        {'start':2.,'end':2.4,'word':'うん。'}]}
    cfg={**config(),'key':'test','diarizer':'fish-qwen-gemini'}
    calls=[]
    def chat(payload,key,folder,ids,kind,progress):
        calls.append(kind)
        schema=payload['response_format']['json_schema']['schema']
        assert schema['additionalProperties']['type']==(['integer','null'] if kind=='speakers' else ['string','null'])
        if kind=='speakers':
            assert ids==[0,1,2]
            assert payload['model']==QWEN and payload['reasoning']=={'enabled':False}
            assert payload['messages'][0]=={'role':'system','content':f.SPEAKER_PROMPT}
            content=payload['messages'][1]['content'];assert len(content)==3 and content[1]['type']=='input_audio'
            assert 'Do NOT repeat the supplied region data' in content[-1]['text']
            assert content[1]['input_audio']['data']=='YXVkaW8='
            assert '"speaker"' not in content[0]['text']
            return {'0':0,'1':1,'2':1},{'usage':{}}
        assert payload['model']==GEMINI
        assert ids==[0,1]
        assert all(isinstance(m['content'],str) for m in payload['messages'])
        context=json.loads(payload['messages'][1]['content'])
        assert all(set(r)=={'id','speaker','text'} for r in context)
        assert [r['speaker'] for r in context]==['qwen_0','qwen_1']
        assert 'input_audio' not in json.dumps(payload)
        return f.parse_map('{"0":"[surprised]정말?","1":null}',ids,kind),{'usage':{}}
    with patch.object(f,'transcribe_words',return_value=raw) as asr, \
         patch.object(f,'encode_audio',return_value=(encoded,{'format':'mp3'})), \
         patch.object(f,'chat',side_effect=chat):
        result=transcribe(audio,tmp_path/'work',tmp_path/'segments.json',cfg,lambda *a:None)
        cached=transcribe(audio,tmp_path/'work',tmp_path/'segments.json',cfg,lambda *a:None)
        refined,_=refine_references(audio,result,tmp_path/'references')
        prepared=prepare_segments(refined,3*SR)
    assert calls==['speakers','translations']
    assert asr.call_count==2 # The source loader handles its own audio-hash cache.
    assert result==cached
    assert len(result['segments'])==1 and len(result['excluded_intervals'])==1
    assert result['source_draft_region_count']==3 and len(result['source_regions'])==2
    assert result['segments'][0]['translated_text']=='[surprised]정말?'
    assert result['source_asr_model']==f.ASR and result['speaker_authority']==QWEN
    assert not result['llm_timestamps'] and result['llm_audio_input'] is None
    assert not prepared[0]['natural_reference_margin']


def test_saved_response_recovery_requires_exact_input_identity(tmp_path):
    from core import atomic_json
    folder=tmp_path/'qwen-speaker-responses';folder.mkdir()
    atomic_json(tmp_path/'qwen-speaker-input.json',{'signature':{'source':'one'}})
    atomic_json(folder/'response.json',{'model':QWEN,'usage':{},'choices':[{'finish_reason':'stop',
        'message':{'content':'[{"id":0,"speaker":2}]'}}]})
    payload={'model':QWEN,'response_format':f.schema('speakers')}
    with patch.object(f,'chat',side_effect=AssertionError('Unexpected paid request')):
        result,receipt=f.stage_chat(payload,'test',tmp_path,{'source':'one'},[0],'speakers',lambda *a:None)
    assert result=={'0':2} and receipt['recovered_from_saved_response']
    with patch.object(f,'chat',return_value=({'0':1},{'usage':{}})) as call:
        result,_=f.stage_chat(payload,'test',tmp_path,{'source':'different'},[0],'speakers',lambda *a:None)
    assert call.call_count==1 and result=={'0':1}


def test_audio_budget_diagnostics_do_not_invalidate_identical_audio():
    old={'source':'one','audio':{'sha256':'abc','format':'mp3'}}
    new={'source':'one','audio':{'sha256':'abc','format':'mp3','base64_bytes':100,'base64_budget_bytes':8000000}}
    assert f.same_signature(old,new)
    assert not f.same_signature(old,{**new,'audio':{**new['audio'],'sha256':'different'}})


def test_same_input_retry_does_not_hide_saved_response_behind_new_manifest_time(tmp_path):
    from core import atomic_json
    folder=tmp_path/'qwen-speaker-responses';folder.mkdir()
    manifest=tmp_path/'qwen-speaker-input.json'
    atomic_json(manifest,{'signature':{'source':'one'}})
    atomic_json(folder/'response.json',{'model':QWEN,'choices':[{'finish_reason':'stop',
        'message':{'content':'[{"region_id":0,"speaker_id":4}]'}}]})
    epoch=manifest.stat().st_mtime_ns
    payload={'model':QWEN,'response_format':f.schema('speakers')}
    with patch.object(f,'parse_map',side_effect=ValueError('old parser')), \
         patch.object(f,'chat',return_value=({},{})):
        f.stage_chat(payload,'test',tmp_path,{'source':'one'},[0],'speakers',lambda *a:None)
    assert manifest.stat().st_mtime_ns==epoch
    with patch.object(f,'chat',side_effect=AssertionError('Do not repeat completed request')):
        result,_=f.stage_chat(payload,'test',tmp_path,{'source':'one'},[0],'speakers',lambda *a:None)
    assert result=={'0':4}


def test_post_qwen_merge_ignores_sentence_ends_but_respects_voice_unknown_and_gap():
    rows=[{'id':i,'start':a,'end':b,'speaker':s,'text':t} for i,(a,b,s,t) in enumerate([
        (0.,1.,'qwen_0','First!'),(1.8,2.,'qwen_0','Second?'),(2.1,3.,'qwen_1','Different.'),
        (3.9,4.,'qwen_1','Long gap.'),(4.1,4.5,'unknown','Unknown.'),(4.6,5.,'unknown','Another.')])]
    result,audit,changes=f.merge_voice_turns(rows,[{} for _ in rows])
    assert len(result)==5 and len(changes)==1
    assert result[0]['text']=='First! Second?' and result[0]['source_region_ids']==[0,1]
    assert result[0]['start']==0. and result[0]['end']==2.
    assert [r['id'] for r in result]==list(range(5))
    assert len(audit)==len(result)
