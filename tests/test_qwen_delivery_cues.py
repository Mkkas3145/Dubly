import json
from unittest.mock import patch
import numpy as np
import soundfile as sf
from qwen_delivery_cues import validate


def test_cues_cannot_change_spoken_words_punctuation_or_pairing():
    assert validate(['[sad] 안녕하세요 [laughing] 반갑습니다.','[sad] Hello [laughing] nice to meet you.'],
        '안녕하세요 반갑습니다.','Hello nice to meet you.') is None
    assert validate(['[surprised]本当？','[surprised]정말?'],'本当？','정말?') is None
    assert validate(['本当？！','정말?'],'本当？','정말?')=='spoken_text_or_punctuation_changed'
    assert validate(['[surprised]本当？','정말?'],'本当？','정말?')=='paired_cues_differ'
    assert validate(['本当？[surprised]','정말?[surprised]'],'本当？','정말?')=='cue_after_final_spoken_word'
    assert validate(['[驚き]本当？','[驚き]정말?'],'本当？','정말?')=='invalid_or_non_english_cue'



def test_source_cues_preserve_voice_time_and_fallback_per_region(tmp_path):
    from qwen_delivery_cues import annotate_source
    clean=tmp_path/'a.wav';sf.write(clean,np.zeros(32000),16000)
    rows=[{'id':0,'start':0.,'end':.8,'speaker':'a','text':'本当？'},
          {'id':1,'start':1.,'end':1.9,'speaker':'b','text':'はい！'}]
    with patch('qwen_delivery_cues.request',return_value=({'segments':[{'id':0,'text':'[surprised] 本当？'},{'id':1,'text':'別の台詞'}]},{'requests':1})):
        result,report=annotate_source(clean,rows,tmp_path,{'key':'test'})
    assert result[0]['text']=='[surprised] 本当？' and result[1]['text']=='はい！'
    assert all((r['speaker'],r['start'],r['end'])==(o['speaker'],o['start'],o['end']) for r,o in zip(result,rows))
    assert len(report['reviews'])==1 and report['annotated_regions']==1


def test_equivalent_response_wrappers_still_require_typed_unique_entries():
    import pytest
    from qwen_delivery_cues import normalize_response
    rows=[{'id':1,'text':'[soft] はい。'}]
    assert normalize_response(rows)==normalize_response({'segments':rows})=={'segments':rows}
    for invalid in (rows+rows,[{'id':True,'text':'text'}],[{'id':1,'text':'text','speaker':2}]):
        with pytest.raises(ValueError):normalize_response(invalid)


def test_annotation_only_refresh_preserves_locked_words_and_places_internal_cue():
    from qwen_delivery_cues import expand_cue_response,cue_units
    rows=[{'id':7,'text':'Hello, nice to meet you.'}]
    units=cue_units(rows[0]['text']);nice=next(i for i,(_,text) in enumerate(units) if text=='nice')
    result=expand_cue_response({'segments':[{'id':7,'c':'soft low voice','m':[{'i':nice,'c':'light laugh'}]}]},rows)
    assert result['segments'][0]['text']=='[soft low voice] Hello, [light laugh] nice to meet you.'
    assert validate([result['segments'][0]['text']]*2,rows[0]['text'],rows[0]['text']) is None


def test_annotation_refresh_rejects_missing_lines_and_unknown_positions():
    import pytest
    from qwen_delivery_cues import expand_cue_response
    rows=[{'id':7,'text':'Hello.'}]
    with pytest.raises(ValueError):expand_cue_response({'segments':[]},rows)
    with pytest.raises(ValueError):expand_cue_response({'segments':[{'id':7,'c':'low voice','m':[{'i':99,'c':'hesitant'}]}]},rows)
