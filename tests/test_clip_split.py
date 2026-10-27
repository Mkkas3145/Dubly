import copy
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from test_studio_workflow import fixture,body
from studio_workflow import load,save,clip_file,paths,restore_generation,digest
from clip_split import split_clip,split_tagged,cut_from_audio


def test_inline_cue_stays_with_its_words_and_active_cue_carries_right():
    assert split_tagged('[낮게] 안녕 반가워',3)==('[낮게] 안녕','[낮게] 반가워')
    assert split_tagged('[낮게] 안녕 [큰 소리로] 반가워',3)==('[낮게] 안녕','[큰 소리로] 반가워')
    assert split_tagged('[낮게] 안녕',0)==('','[낮게] 안녕')
    assert split_tagged('こんにちは。さようなら。',6)==('こんにちは。','さようなら。')


def test_text_split_follows_aligned_words_not_character_ratio():
    text='[soft] 안녕 반가워'
    words=[{'word':'안녕','start':0.,'end':.4},{'word':'반가워','start':2.,'end':3.}]
    assert cut_from_audio(text,words,1.5)==3


def test_split_audio_is_exact_and_can_restore_full_original(tmp_path):
    video,store,doc,ctx,values=fixture(tmp_path)
    before_sha=digest(clip_file(video,0,tmp_path))
    plan={'original':['はい','！'],'translated':['[낮게] 네','[낮게] !'],'source_time':.4}
    with patch('clip_split.text_plan',return_value=plan):
        result=split_clip(video,0,{'revision':doc['revision'],'time':.75},tmp_path)
    assert result['left']==0 and result['right']==2
    left,sr=sf.read(clip_file(video,0,tmp_path),dtype='float32',always_2d=True)
    right,_=sf.read(clip_file(video,2,tmp_path),dtype='float32',always_2d=True)
    np.testing.assert_array_equal(np.concatenate([left,right]),values)
    split,ctx=load(video,tmp_path)
    assert split['transcript']['segments'][2]['audio_edit']['position']==.75
    assert result['document']['segments'][0]['audio_stale'] is False
    assert result['document']['segments'][2]['audio_stale'] is False
    undo=body(split)
    for i in [0,1]:
        undo['segments'][i].update({k:v for k,v in doc['transcript']['segments'][i].items() if not k.startswith('_')})
    undo['segments'][2]['deleted']=True
    saved,_=save(video,undo,tmp_path)
    restored,_=restore_generation(video,0,before_sha,tmp_path)
    assert restored['transcript']['segments'][0]['end']==1
    assert restored['transcript']['segments'][2]['deleted']
    actual,_=sf.read(clip_file(video,0,tmp_path),dtype='float32',always_2d=True)
    np.testing.assert_array_equal(actual,values)


def test_conflicting_split_revision_cannot_replace_audio(tmp_path):
    video,store,doc,ctx,values=fixture(tmp_path)
    before=digest(clip_file(video,0,tmp_path))
    with pytest.raises(RuntimeError):split_clip(video,0,{'revision':doc['revision']-1,'time':.5},tmp_path)
    assert digest(clip_file(video,0,tmp_path))==before


def test_split_http_endpoint_and_empty_speech_fragment_save(tmp_path):
    import app as api
    from fastapi.testclient import TestClient
    from queue_worker import VideoQueue
    video,store,doc,ctx,values=fixture(tmp_path);queue=VideoQueue(store,tmp_path)
    plan={'original':['','はい！'],'translated':['','[낮게] 네!'],'source_time':.1}
    with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start'),patch('clip_split.text_plan',return_value=plan),TestClient(api.app) as client:
        response=client.post(f'/api/studio/{video}/segments/0/split',json={'revision':doc['revision'],'time':.5})
    assert response.status_code==200,response.text
    current,_=load(video,tmp_path)
    saved,_=save(video,body(current),tmp_path)
    assert saved['transcript']['segments'][0]['keep_original'] is False
    assert saved['generated']['0']['processing_policy']
