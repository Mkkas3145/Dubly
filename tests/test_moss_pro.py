import json
import pytest
from moss_pro import parse_events,keyterms


def test_stream_has_exact_speaker_times_and_progress(tmp_path):
    events=[{'type':'task.created','task_id':'a'},
      {'type':'transcript.segment.done','start':.12,'end':1.17,'speaker':'S17','text':'声'},
      {'type':'transcript.text.done','text':'声','usage':{'total_tokens':4}}]
    seen=[]
    result=parse_events([('data: '+json.dumps(e)).encode() for e in events],tmp_path/'raw.sse',lambda *args:seen.append(args))
    assert result['segments']==[{'start':.12,'end':1.17,'speaker':'S17','text':'声'}]
    assert result['usage']['total_tokens']==4 and seen


def test_truncated_stream_never_succeeds(tmp_path):
    with pytest.raises(RuntimeError,match='완료 전에'):
        parse_events(['data: {"type":"transcript.text.delta","delta":"a"}'],tmp_path/'raw.sse',lambda *a:None)


def test_hotwords_are_explicit_terms_not_freeform_instructions():
    assert keyterms('["アーニャ","ロイド","アーニャ"]')==['アーニャ','ロイド']
    for value in ('a',{'a':1},['a']*21,['a'*31]):
        with pytest.raises(RuntimeError):keyterms(value)


def test_pro_selection_reaches_translation_without_local_inference(tmp_path):
    from unittest.mock import patch
    import numpy as np
    import soundfile as sf
    import moss_pipeline
    clean=tmp_path/'audio.wav';sf.write(clean,np.zeros(32000),16000)
    raw={'signature':{'policy':{'model':'moss-transcribe-diarize-pro'}},'segments':[
        {'start':.1,'end':1.1,'speaker':'S19','text':'台詞'}]}
    with patch('moss_pro.source',return_value=raw) as api, \
         patch('moss_pipeline.source',side_effect=AssertionError('local must not run')), \
         patch('moss_fish_regions.transcribe_regions',side_effect=AssertionError('Fish ASR must not run')), \
         patch('fish_qwen_pipeline.stage_chat',return_value=({'0':'대사'},{'usage':{}})):
        result=moss_pipeline.transcribe(clean,tmp_path,tmp_path/'result.json',
            {'key':'test','language':'ko','moss_api_key':'private','moss_keyterms':'["固有名"]'},lambda *a:None)
    assert api.call_args.args[3]=='["固有名"]'
    assert result['speaker_authority']=='moss-transcribe-diarize-pro'
    assert result['segments'][0]['speaker_id']=='S19'
    assert result['request_counts']['source_asr_api']==1
    assert result['request_counts']['speakers_api']==0
    assert result['fish_asr_requests']==0
    assert result['models']['source_asr']=='moss-transcribe-diarize-pro'
