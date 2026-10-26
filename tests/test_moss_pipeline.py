import json
from unittest.mock import patch
import numpy as np
import soundfile as sf
import pytest
import moss_pipeline as moss
import render,speech
from core import config,SR
from reference_timing import refine_references


def test_moss_native_times_and_speakers_survive_translation_and_reference_preparation(tmp_path):
    clean=tmp_path/'dialogue.wav';sf.write(clean,np.zeros((3*SR,2)),SR)
    raw={'signature':{'test':'native'},'text':'raw','segments':[
        {'start':.17,'end':.91,'speaker':'S01','text':'[laughs]一'},
        {'start':1.12,'end':1.44,'speaker':'S02','text':'二'},
        {'start':1.48,'end':2.7,'speaker':'S01','text':'三'}]}
    captured=[]
    def translate(payload,key,work,signature,ids,kind,progress):
        captured.append(payload)
        return {'0':'[laughs]하나','1':None,'2':'셋'},{'usage':{}}
    def fish(clean,rows,*args):return rows,{'api_region_requests':len(rows),'fallback_count':0}
    with patch.object(moss,'source',return_value=raw),patch('moss_fish_regions.transcribe_regions',side_effect=fish),patch('fish_qwen_pipeline.stage_chat',side_effect=translate):
        result=speech.transcribe(clean,tmp_path,tmp_path/'segments.json',{'diarizer':'moss-gemini','key':'test','language':'ko'},lambda *a:None)
    context=json.loads(captured[0]['messages'][1]['content'])
    assert 'Preserve expressive punctuation' in captured[0]['messages'][0]['content']
    assert 'Preserve every supplied [] cue verbatim in its original language' in captured[0]['messages'][0]['content']
    assert 'using English inside brackets' not in captured[0]['messages'][0]['content']
    assert all(set(r)=={'id','speaker','text'} for r in context)
    assert [r['speaker'] for r in context]==['S01','S02','S01']
    assert not any('audio' in json.dumps(m) for m in captured[0]['messages'])
    assert result['models']['source_asr']==moss.MODEL and result['models']['speakers']==moss.MODEL
    assert result['fish_asr_requests']==0
    assert result['boundary_authority']==moss.AUTHORITY
    with patch('reference_timing.acoustic_boundaries',side_effect=AssertionError('No boundary correction')):
        same,_=refine_references(clean,result,tmp_path/'timing')
    segments=render.prepare_segments(same,3*SR,{})
    assert [(s['start'],s['end'],s['speaker_id']) for s in segments]==[(.17,.91,'S01'),(1.48,2.7,'S01')]
    assert all(not s['natural_reference_margin'] for s in segments)
    assert [(s['reference_lo'],s['reference_hi']) for s in segments]==[(round(.17*SR),round(.91*SR)),(round(1.48*SR),round(2.7*SR))]
    assert result['excluded_intervals'][0]['start']==1.12
    # A second translation call must reuse the input-identical cache.
    with patch.object(moss,'source',return_value=raw),patch('moss_fish_regions.transcribe_regions',side_effect=fish),patch('fish_qwen_pipeline.stage_chat',side_effect=AssertionError('cached')):
        moss.transcribe(clean,tmp_path,tmp_path/'segments.json',{'key':'test','language':'ko'},lambda *a:None)


def test_overlap_keeps_original_without_shifting_or_merging_moss_times():
    raw={'segments':[{'start':0.,'end':1.,'speaker':'S01','text':'a'},
        {'start':.8,'end':1.2,'speaker':'S02','text':'b'},
        {'start':1.5,'end':2.,'speaker':'S01','text':'c'}]}
    rows=moss.regions(raw,3)
    assert [(r['start'],r['end']) for r in rows]==[(0.,1.),(.8,1.2),(1.5,2.)]
    assert [r.get('keep_original',False) for r in rows]==[True,True,False]


def test_current_pipeline_default():
    assert config()['diarizer']=='scribe-v2'
    assert config()['tts_reference_mode']=='individual'


def test_source_identity_round_trips_json(tmp_path):
    path=tmp_path/'input.wav';path.write_bytes(b'input')
    identity=moss.source_signature(path)
    assert identity==json.loads(json.dumps(identity))
    assert identity['prompt_sha256']


def test_moss_acting_prompt_keeps_native_format_and_asks_for_audible_english_cues():
    from moss_prompt import transcription_prompt
    result=transcription_prompt('native timestamps and speaker format')
    assert result.endswith('\nnative timestamps and speaker format')
    assert '英文[...]' in result and 'TTS' in result
    assert '结束时间戳' in result and '不省略原文' in result
