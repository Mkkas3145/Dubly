import json
import pytest
from qwen_response import normalize,parse

EXPECTED={'segments':[{'id':0,'text':'嘘だ！'},{'id':1,'text':'本当？！'}]}

@pytest.mark.parametrize('value',[
    EXPECTED,EXPECTED['segments'],
    {'0':'嘘だ！','1':'本当？！'},
    {'results':{'0':'嘘だ！','1':'本当？！'}},
    {'data':{'corrections':[{'region_id':'0','corrected_text':'嘘だ！'},{'segment_id':'1','punctuated_text':'本当？！'}]}},
    [['0','嘘だ！'],[1,'本当？！']],['嘘だ！','本当？！'],
    {'segments':[{'id':'1','text':'本当？！','start':999,'end':1000},{'id':'0','text':'嘘だ！','speaker_id':'fake'}]},
    {'0':{'original_text':'嘘だ。','corrected_text':'嘘だ！'},'1':{'text':'本当？！'}},
])
def test_common_shapes_keep_supplied_identity_and_ignore_echoed_metadata(value):
    assert normalize(value,[0,1])==EXPECTED
    assert parse(json.dumps(value,ensure_ascii=False),[0,1])==EXPECTED

@pytest.mark.parametrize('content',[
    '```json\n'+json.dumps(EXPECTED)+'\n```',
    'Here is the corrected result:\n'+json.dumps(EXPECTED)+'\nDone.',
    '結果です:\n```json\n'+json.dumps(EXPECTED)+'\n```\n以上です。',
    "{'segments': [{'id': 0, 'text': '嘘だ！'}, {'id': 1, 'text': '本当？！'},],}",
    "```json\n{'0': '嘘だ！', '1': '本当？！',}\n```",
    '0: 嘘だ！\n1: 本当？！',
    '- [0]：嘘だ！\n- [1]：本当？！',
])
def test_common_text_wrappers_and_literal_formats_are_supported(content):
    assert parse(content,[0,1])==EXPECTED

@pytest.mark.parametrize('content',[
    '{"0":"first","0":"second","1":"other"}',
    '{"segments":[{"id":0,"text":"first"},{"id":"0","text":"second"}]}',
    '{"0":"text"}', '{"0":"text","1":"other","2":"extra"}',
    '{"segments":[{"id":false,"text":"text"},{"id":1,"text":"other"}]}',
    '{"segments":[{"id":0,"index":1,"text":"text"},{"id":1,"text":"other"}]}',
    '["only one"]',
    '```json\n{"0":"first","1":"other"}\n```\n```json\n{"0":"different","1":"other"}\n```',
    "__import__('os').system('should never run')",
])
def test_ambiguous_missing_or_extra_ids_never_get_guessed(content):
    with pytest.raises(ValueError):parse(content,[0,1])
