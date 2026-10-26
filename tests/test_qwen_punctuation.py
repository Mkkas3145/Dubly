from unittest.mock import patch
import numpy as np
import soundfile as sf
import pytest
from qwen_punctuation import validate,repair_source

def test_echoed_clock_hints_are_ignored_and_cannot_retime_source():
    from qwen_punctuation import normalize_punctuation_response
    assert normalize_punctuation_response([{'id':0,'start':99,'end':100,'text':'はい！'}])=={'segments':[{'id':0,'text':'はい！'}]}


@pytest.mark.parametrize('old,new',[
    ('嘘だ。','嘘だ！'),('嘘だ。。','嘘だ...'),('本当。','本当？！'),
    ('안녕. 정말.','안녕!! 정말?!'),('No. Wait.','No... Wait?!'),
    ('Wait.','Wait'+'!'*150),('100.','100?!'),('안녕.','안녕~'),('やあ。','やあ〜'),
    ('こんにちは。','こんにちは～'),('안녕.','“안녕”—'),('Wait.','Wait……')])
def test_expression_punctuation_and_repetition_are_allowed(old,new):
    assert validate(new,old) is None


@pytest.mark.parametrize('old,new',[
    ('嘘だ。','嘘だよ！'),('Hello.','[shouting] Hello!'),('1.5.','15!'),
    ('100.','1.00!'),('100.','1?!00!'),('hello.','hel!lo!'),
    ('Hello world.','Helloworld!'),('I can’t.','I cant!'),
    ('[laughing] Hello.','[laughing!] Hello!'),('100.','100%'),('100%.','100!')])
def test_words_numbers_whitespace_and_cues_are_immutable(old,new):
    assert validate(new,old) is not None


def test_audio_request_cached_and_only_valid_punctuation_applied_per_region(tmp_path):
    clean=tmp_path/'a.wav';sf.write(clean,np.zeros(32000),16000)
    rows=[{'id':0,'start':0.,'end':.8,'speaker':'0','text':'嘘だ。'},
          {'id':1,'start':1.,'end':1.9,'speaker':'1','text':'本当。'}]
    response={'segments':[{'id':0,'text':'嘘だ！！'},{'id':1,'text':'[surprised] 本当？！'}]}
    retry={'segments':[{'id':1,'text':'本当？！'}]}
    with patch('qwen_punctuation.request',side_effect=[(response,{'requests':1}),(retry,{'requests':1})]) as request:
        result,receipt=repair_source(clean,rows,tmp_path,{'key':'test'})
        cached,receipt2=repair_source(clean,rows,tmp_path,{'key':'test'})
    assert request.call_count==2 and receipt2['requests']==0
    assert result==cached
    assert result[0]['text']=='嘘だ！！' and result[1]['text']=='本当？！'
    assert receipt['corrected_regions']==2 and receipt['reviews']==[]
    assert receipt['validation_retries'][0]['ids']==[1]
    assert [(r['start'],r['end'],r['speaker']) for r in result]==[(r['start'],r['end'],r['speaker']) for r in rows]
    payload=request.call_args_list[0].args[0]
    assert request.call_args_list[0].kwargs['operation']=='문장부호 보정'
    assert payload['messages'][1]['content'][1]['type']=='input_audio'
    assert 'Do not transcribe again' in payload['messages'][0]['content']


def test_changed_region_ids_keep_originals_without_applying_cross_line_output(tmp_path):
    clean=tmp_path/'a.wav';sf.write(clean,np.zeros(16000),16000)
    rows=[{'id':0,'start':0.,'end':.8,'speaker':'0','text':'はい。'}]
    with patch('qwen_punctuation.request',return_value=({'segments':[{'id':2,'text':'はい！'}]},{})):
        result,receipt=repair_source(clean,rows,tmp_path,{'key':'test'})
    assert result==rows and receipt['error']=='region_ids_changed'


def test_repeated_word_changes_stop_after_bounded_retries_and_preserve_original(tmp_path):
    clean=tmp_path/'a.wav';sf.write(clean,np.zeros(16000),16000)
    rows=[{'id':0,'start':0.,'end':.8,'speaker':'0','text':'知らない。'}]
    response={'segments':[{'id':0,'text':'でも、知らない！'}]}
    with patch('qwen_punctuation.request',return_value=(response,{'requests':1})) as request:
        result,receipt=repair_source(clean,rows,tmp_path,{'key':'test'})
    assert request.call_count==3 and result==rows
    assert len(receipt['reviews'])==1 and len(receipt['validation_retries'])==2


@pytest.mark.parametrize('old,new',[
    ('知らない人だよ。','し、知らない人だよ。'),
    ('俺が知りたいよ。','お、俺が知りたいよ。'),
    ('아무것도.','아, 아무것도!'),('Hello.','H, Hello!'),
    ('知らない。','し、し、知らない！')])
def test_audible_opening_sound_repetitions_can_be_restored(old,new):
    assert validate(new,old) is None


@pytest.mark.parametrize('old,new',[
    ('知らない。','でも、知らない！'),('俺が。','彼が！'),
    ('아무것도.','정말 아무것도!'),('100.','1、100!'),
    ('知らない。','知ってる！')])
def test_stutter_exception_does_not_allow_semantic_rewrites(old,new):
    assert validate(new,old) is not None
