import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
import app as api
from storage import Store
from queue_worker import VideoQueue
from core import config,video_languages,SR


def test_upload_persists_additional_translation_instructions(tmp_path):
    store=Store(tmp_path);queue=VideoQueue(store,tmp_path)
    (tmp_path/'work/videos').mkdir(parents=True)
    info={'streams':[{'codec_type':'audio'},{'codec_type':'video'}],'format':{'duration':'10'}}
    instruction='인물 이름은 원어 발음대로 표기하고 존댓말을 유지해 주세요.'
    with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start'),patch.object(api,'probe',return_value=info),patch('audio_tracks.initialize',return_value={'selection_required':False}),TestClient(api.app) as client:
        response=client.post('/api/videos',data={'source_language':'ja','target_language':'ko','translation_prompt':'  '+instruction+'  '},files={'file':('test.mp4',b'mp4','video/mp4')})
    assert response.status_code==201,response.text
    job=store.get(response.json()['id'])
    assert job['translation_prompt']==instruction
    assert video_languages(config(),job)['translation_prompt']==instruction


def test_custom_translation_context_and_cache_identity(tmp_path):
    from gemini_source_pipeline import source
    from test_gemini_source_pipeline import answer
    audio=tmp_path/'voice.wav';sf.write(audio,np.zeros(SR,dtype='float32'),SR)
    contexts=[]
    def request(payload,key,**kwargs):
        contexts.append(json.loads(payload['messages'][1]['content'][0]['text']))
        return kwargs['parser'](json.dumps(answer())),{}
    cfg={**config(),'key':'test','language':'ko','translation_prompt':'존댓말을 유지하세요.'}
    with patch('scribe.prepare_input',return_value=(audio,'wav',1.)),patch('speech.http_json',side_effect=request):
        first=source(audio,tmp_path/'stage',cfg,lambda *a:None)
        source(audio,tmp_path/'stage',cfg,lambda *a:None)
        second=source(audio,tmp_path/'stage',{**cfg,'translation_prompt':'반말을 사용하세요.'},lambda *a:None)
        source(audio,tmp_path/'empty',{**cfg,'translation_prompt':''},lambda *a:None)
    assert len(contexts)==3
    assert contexts[0]['ADDITIONAL_TRANSLATION_INSTRUCTIONS']=='존댓말을 유지하세요.'
    assert contexts[1]['ADDITIONAL_TRANSLATION_INSTRUCTIONS']=='반말을 사용하세요.'
    assert 'ADDITIONAL_TRANSLATION_INSTRUCTIONS' not in contexts[2]
    assert first['signature']!=second['signature']
