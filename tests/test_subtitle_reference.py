import json,uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
import app as api
from storage import Store
from queue_worker import VideoQueue
from subtitle_reference import parse_srt,save_for_parts,load_for_video,MAX_BYTES

SRT='1\n00:00:01,000 --> 00:00:03,000\n<i>안녕하세요.</i>\n\n2\n00:00:07,000 --> 00:00:09,000\n반갑습니다.\n'


def test_decode_and_parse_preserve_multilingual_and_numeric_subtitle_text():
    for raw in [SRT.encode('utf8'),b'\xef\xbb\xbf'+SRT.encode('utf8'),SRT.encode('utf-16')]:
        ref=parse_srt(raw,r'C:\fakepath\reference.SRT')
        assert ref['name']=='reference.SRT'
        assert ref['cues']==[{'start':1.,'end':3.,'text':'안녕하세요.'},{'start':7.,'end':9.,'text':'반갑습니다.'}]
    ref=parse_srt(b'1\n00:00:01,000 --> 00:00:03,000\n100\n','numbers.srt')
    assert ref['cues'][0]['text']=='100'
    with pytest.raises(ValueError):parse_srt(SRT.encode(),'wrong.txt')
    with pytest.raises(ValueError):parse_srt(b'not subtitles','wrong.srt')
    with pytest.raises(ValueError):parse_srt(b'x'*(MAX_BYTES+1),'large.srt')


def test_split_references_are_cropped_and_rebased_to_each_uploaded_part(tmp_path):
    parts=[]
    for n,(lo,hi) in enumerate([(0,2),(2,6),(6,10)]):
        folder=tmp_path/str(n);folder.mkdir();(folder/'split-origin.json').write_text(json.dumps({'start':lo,'end':hi}),'utf8')
        parts.append({'folder':folder,'info':{'format':{'duration':hi-lo}}})
    save_for_parts(parse_srt(SRT.encode(),'reference.srt'),parts)
    saved=[json.loads((p['folder']/'translation-reference.json').read_text('utf8')) for p in parts]
    assert saved[0]['cues']==[{'start':1.,'end':2,'text':'안녕하세요.'}]
    assert saved[1]['cues']==[{'start':0.,'end':1.,'text':'안녕하세요.'}]
    assert saved[2]['cues']==[{'start':1.,'end':3.,'text':'반갑습니다.'}]


def test_upload_saves_reference_and_video_config_forwards_it_to_gemini(tmp_path):
    import core
    from gemini_source_pipeline import source
    from test_gemini_source_pipeline import answer
    store=Store(tmp_path);queue=VideoQueue(store,tmp_path);(tmp_path/'work/videos').mkdir()
    info={'streams':[{'codec_type':'audio'},{'codec_type':'video'}],'format':{'duration':'10'}}
    with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start'),patch.object(api,'probe',return_value=info),patch('audio_tracks.initialize',return_value={'selection_required':False}),TestClient(api.app) as client:
        response=client.post('/api/videos',data={'source_language':'ja','target_language':'ko'},files={'file':('video.mp4',b'mp4','video/mp4'),'translation_srt':('reference.srt',SRT.encode(),'application/x-subrip')})
        assert response.status_code==201,response.text
        bad=client.post('/api/videos',data={'source_language':'ja','target_language':'ko'},files={'file':('video.mp4',b'mp4','video/mp4'),'translation_srt':('bad.srt',b'not subtitles','text/plain')})
        assert bad.status_code==422 and len(store.list())==1
    job=store.get(response.json()['id'])
    with patch.object(core,'ROOT',tmp_path):cfg=core.video_languages({**core.config(),'key':'test'},job)
    assert cfg['translation_reference']['name']=='reference.srt'
    assert len(load_for_video(job['id'],tmp_path)['cues'])==2
    audio=tmp_path/'voice.wav';sf.write(audio,np.zeros(1000,dtype='float32'),1000)
    contexts=[]
    def request(payload,key,**kwargs):
        contexts.append(json.loads(payload['messages'][1]['content'][0]['text']))
        assert 'never replace the audio transcription' in payload['messages'][0]['content']
        return kwargs['parser'](json.dumps(answer())),{}
    with patch('scribe.prepare_input',return_value=(audio,'wav',1.)),patch('speech.http_json',side_effect=request):
        source(audio,tmp_path/'analysis',cfg,lambda *a:None)
        source(audio,tmp_path/'analysis',cfg,lambda *a:None)
        source(audio,tmp_path/'analysis',{**cfg,'translation_reference':{**cfg['translation_reference'],'cues':[{'start':0,'end':1,'text':'다른 참고 문장'}]}},lambda *a:None)
    assert len(contexts)==2
    assert contexts[0]['REFERENCE_SUBTITLES']['cues'][0]['text']=='안녕하세요.'
