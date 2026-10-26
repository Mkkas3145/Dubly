import io,json,uuid
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
import app as api
import render
from core import SR,video_languages
from storage import Store
from tts_presets import PRESETS,options

def test_upload_persists_per_video_preset_and_rejects_unknown(tmp_path):
    store=Store(tmp_path);(tmp_path/'work/videos').mkdir()
    info={'format':{'duration':'1'},'streams':[{'codec_type':'audio'},{'codec_type':'video'}]}
    with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'probe',return_value=info),TestClient(api.app) as client:
        for code in PRESETS:
            r=client.post('/api/videos',data={'source_language':'ja','target_language':'ko','tts_preset':code},files={'file':('sample.mp4',b'video','video/mp4')})
            assert r.status_code==201
            job=store.get(r.json()['id']);assert job['tts_preset']==code
            assert video_languages({'language':'en'},job)['tts_provider_options']==options(code)
        bad=client.post('/api/videos',data={'tts_preset':'invalid'},files={'file':('sample.mp4',b'video','video/mp4')})
        assert bad.status_code==422 and len(store.list())==3

def test_sampling_reaches_actual_speech_request_and_invalidates_generation_cache(tmp_path):
    clean=tmp_path/'clean.wav';sf.write(clean,np.ones((SR,2),np.float32)*.1,SR)
    segment={'start':0.,'end':1.,'lo':0,'hi':SR,'original_text':'はい。','translated_text':'네.','speaker_id':'0'}
    payloads=[]
    class Reply(io.BytesIO):headers={}
    def reply(req,**kwargs):payloads.append(json.loads(req.data));return Reply(b'fake encoded speech')
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'1'}}):
        a=render.generate_speech(0,segment,clean,tmp_path,{'language':'ko','key':'test','tts_provider_options':options('balanced')})
        b=render.generate_speech(0,segment,clean,tmp_path,{'language':'ko','key':'test','tts_provider_options':options('consistent')})
    assert len(payloads)==2 and a['policy']!=b['policy']
    for payload,code in zip(payloads,['balanced','consistent']):
        assert payload['instructions']==render.tts_instructions('ko')
        assert payload['instructions']==('원본 언어에서 한국어로 자연스럽게 더빙하세요, 음성 및 발성, 말하기와 리듬, 감정 및 연기, 역동성, 현지화, 싱크 및 기술적 완성도 등의 조건들을 모두 충족해야 합니다.')
        assert payload['provider']['options']['fish-audio']=={'latency':'normal','condition_on_previous_chunks':True,**options(code)}
    assert b['generation']['tts_provider_options']=={'latency':'normal','condition_on_previous_chunks':True,**options('consistent')}


def test_per_video_sampling_preserves_zero_after_storage_and_reaches_request(tmp_path):
    from tts_presets import configure
    store=Store(tmp_path)
    video_id=str(uuid.uuid4());store.add(video_id,'test.mp4',tts_preset='consistent')
    with store.connection() as db:
        db.execute('UPDATE videos SET tts_sampling_options=? WHERE id=?',
                   (json.dumps({'temperature':0.,'top_p':0.}),video_id))
    cfg=configure({'language':'ko','key':'test'},store.get(video_id))
    assert cfg['tts_provider_options']=={'temperature':0.,'top_p':0.}
    reference=tmp_path/'reference.wav';reference.write_bytes(b'audio')
    payloads=[]
    class Reply(io.BytesIO):headers={}
    def reply(req,**kwargs):payloads.append(json.loads(req.data));return Reply(b'audio')
    row={'original_text':'hello','translated_text':'hello','_provider_options':cfg['tts_provider_options']}
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'1'}}):
        render.request_audio(reference,row,'ko','test',tmp_path/'result.mp3')
    assert payloads[0]['provider']['options']['fish-audio']=={'latency':'normal','temperature':0.,'top_p':0.,'condition_on_previous_chunks':True}
    assert configure({}, {'tts_preset':'consistent'})['tts_provider_options']==options('consistent')


@pytest.mark.parametrize('override',[{'temperature':-1,'top_p':0},{'temperature':0,'top_p':float('nan')},{'temperature':0},{'temperature':True,'top_p':0}])
def test_invalid_per_video_sampling_is_rejected(override):
    from tts_presets import configure
    with pytest.raises(ValueError):configure({}, {'tts_sampling_options':override})


def test_previous_chunk_context_boolean_changes_reach_provider_and_invalidate_policy(tmp_path):
    reference=tmp_path/'reference.wav';reference.write_bytes(b'audio');payloads=[]
    class Reply(io.BytesIO):headers={}
    def reply(req,**kwargs):payloads.append(json.loads(req.data));return Reply(b'audio')
    receipts=[]
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'1'}}):
        for value in (True,False):
            row={'original_text':'はい。','translated_text':'네.','_provider_options':{'condition_on_previous_chunks':value}}
            receipts.append(render.request_audio(reference,row,'ko','test',tmp_path/f'{value}.mp3'))
    assert [p['provider']['options']['fish-audio']['condition_on_previous_chunks'] for p in payloads]==[True,False]
    assert receipts[0]['tts_policy']!=receipts[1]['tts_policy']


def test_reference_and_output_inline_cues_are_preserved_without_repeating_injected_cue(tmp_path):
    reference=tmp_path/'reference.wav';reference.write_bytes(b'audio');payloads=[]
    class Reply(io.BytesIO):headers={}
    def reply(req,**kwargs):payloads.append(json.loads(req.data));return Reply(b'audio')
    row={'original_text':'[soft] source one\n[hesitant] source two','translated_text':'[soft] first [hesitant] phrase\n[acting] second','_delivery_cue':'acting','_omit_instructions':True}
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'1'}}):
        render.request_audio(reference,row,'ko','test',tmp_path/'sample.mp3')
    payload=payloads[0]
    assert payload['input']=='[acting] [soft] first [hesitant] phrase\n[acting] second'
    assert payload['input_references'][1]['text']=='[soft] source one\n[hesitant] source two'
    assert 'instructions' not in payload
    assert payload['provider']['options']['fish-audio']['condition_on_previous_chunks'] is True


@pytest.mark.parametrize('code,name',[('ko','Korean'),('ja','Japanese'),('en','English'),('ar','Arabic')])
def test_korean_instruction_uses_selected_language_pair(code,name):
    from core import LANGUAGE_CATALOG
    text=render.tts_instructions(code,source_language='ja')
    target=LANGUAGE_CATALOG['languages'][code]['labels']['ko']
    assert text==f'일본어에서 {target}로 자연스럽게 더빙하세요, 음성 및 발성, 말하기와 리듬, 감정 및 연기, 역동성, 현지화, 싱크 및 기술적 완성도 등의 조건들을 모두 충족해야 합니다.'
