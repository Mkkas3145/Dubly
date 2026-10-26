import json
from unittest.mock import patch
import pytest
from tts_alignment_review import review

def missing():return {'matched':False,'word_indices':[],'words':[]}

def test_audio_review_handles_different_spelling_without_number_dictionary_and_caches(tmp_path):
    audio=tmp_path/'generated.mp3';audio.write_bytes(b'audio')
    words=[{'word':'백','start':.2,'end':.6},{'word':'끝','start':1.,'end':1.4}]
    fixed={'matched':True,'word_indices':[1],'start':1.,'end':1.4}
    def call(payload,key,**kwargs):
        assert payload['messages'][1]['content'][1]['type']=='input_audio'
        return kwargs['parser']('{"0":[0,0]}'),{}
    with patch('speech.http_json',side_effect=call) as request:
        locations,audit=review(audio,['100','끝'],words,[missing(),fixed],tmp_path/'review',{'key':'test'})
        cached,_=review(audio,['100','끝'],words,[missing(),fixed],tmp_path/'review',{'key':'test'})
    assert request.call_count==1 and cached==locations
    assert locations[0]['start']==.2 and locations[0]['end']==.6
    assert locations[1]==fixed and audit[0]['method']=='generated_audio_script_review'

def test_audio_review_cannot_reuse_words_owned_by_a_fixed_neighbor(tmp_path):
    audio=tmp_path/'generated.mp3';audio.write_bytes(b'audio')
    words=[{'word':'fixed','start':.1,'end':.5},{'word':'uncertain','start':1.,'end':1.4}]
    fixed={'matched':True,'word_indices':[0],'start':.1,'end':.5}
    with patch('speech.http_json',side_effect=lambda *a,**k:(k['parser']('{"1":[0,1]}'),{})):
        locations,audit=review(audio,['fixed','missing'],words,[fixed,missing()],tmp_path/'review',{'key':'test'})
    assert not locations[1]['matched'] and not audit

def test_audio_review_must_use_real_word_indices(tmp_path):
    audio=tmp_path/'generated.mp3';audio.write_bytes(b'audio')
    with patch('speech.http_json',side_effect=lambda *a,**k:(k['parser']('{"0":[0,99]}'),{})):
        with pytest.raises(ValueError):review(audio,['line'],[{'word':'line','start':0.,'end':1.}],[missing()],tmp_path/'review',{'key':'test'})
