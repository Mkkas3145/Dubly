import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR,config
from mai_pipeline import AUTHORITY,MODEL,normalize,correct_edges,translate

def test_default_and_tts_options():
    from render import TTS_PROVIDER_OPTIONS
    assert config()['diarizer']=='mai-transcribe-2'
    assert TTS_PROVIDER_OPTIONS['temperature']==TTS_PROVIDER_OPTIONS['top_p']==1.

def test_overlap_not_shared_as_single_voice():
    rows=normalize({'segments':[{'start':0,'end':1,'text':'a','speaker':0},
        {'start':.8,'end':2,'text':'b','speaker':1}]},2)
    assert len(rows)==1 and rows[0]['speaker']=='unknown'

def test_acoustic_edges_keep_onset_and_neighbor_separate(tmp_path):
    t=np.arange(SR*3)/SR;v=np.zeros((len(t),2),np.float32)
    for a,b in [(.4,1.),(1.35,2.2)]:
        lo,hi=round(a*SR),round(b*SR);v[lo:hi]=(.2*np.sin(2*np.pi*220*t[lo:hi]))[:,None]
    source=tmp_path/'audio.wav';sf.write(source,v,SR,subtype='FLOAT')
    rows=[{'start':.5,'end':.9,'text':'a','speaker':'0'}, {'start':1.4,'end':2.1,'text':'b','speaker':'1'}]
    corrected,audit=correct_edges(source,rows)
    assert corrected[0]['start']<=.4 and corrected[0]['end']>=1.
    assert corrected[1]['start']<=1.35 and corrected[1]['end']>=2.2
    assert corrected[0]['end']<=corrected[1]['start']
    assert len(audit)==2

def test_text_only_translation_after_edges_and_cached(tmp_path):
    from render import prepare_segments
    from shared_reference import plan
    source=tmp_path/'audio.wav';sf.write(source,np.zeros((SR*3,2),np.float32),SR)
    raw={'language':'ja','segments':[{'start':.4,'end':1.,'text':'はい','speaker':0},
        {'start':1.5,'end':2.,'text':'どうぞ','speaker':1}]}
    requests=[]
    def api(payload,key,**kwargs):
        requests.append(payload);context=json.loads(payload['messages'][1]['content'])
        assert context['regions']==[{'id':0,'speaker':'0','text':'はい'},{'id':1,'speaker':'1','text':'どうぞ'}]
        assert 'input_audio' not in json.dumps(payload)
        assert (tmp_path/'translation/mai-boundaries.json').exists()
        return kwargs['parser']('{"0":"네!","1":"이쪽으로."}'),{}
    with patch('mai_pipeline.transcribe_source',return_value=(raw,{'hash':'test'})),patch('speech.http_json',side_effect=api):
        cfg={'key':'test','language':'ko'}
        result=translate(source,tmp_path/'translation',tmp_path/'result.json',cfg,lambda *a:None)
        again=translate(source,tmp_path/'translation',tmp_path/'result.json',cfg,lambda *a:None)
    assert len(requests)==1 and result==again
    assert result['source_asr_model']==MODEL and result['boundary_authority']==AUTHORITY
    segments=prepare_segments(result,SR*3)
    assert all(not r['natural_reference_margin'] for r in segments)
    assert len(plan(segments,result['speaker_timeline']))==2
    from reference_timing import refine_references
    with patch('native_timeline.references',side_effect=AssertionError('legacy refinement invoked')):
        same,_=refine_references(source,result,tmp_path/'refine')
    assert same is result

def test_source_request_contract_and_raw_cache(tmp_path):
    from mai_pipeline import transcribe_source
    source=tmp_path/'audio.wav';sf.write(source,np.zeros((SR,2),np.float32),SR)
    raw={'segments':[{'start':0,'end':1,'speaker':2,'text':'test'}]}
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(raw).encode()
    def call(request,**kwargs):
        payload=json.loads(request.data)
        assert payload['model']==MODEL
        assert payload['provider']['options']['azure']['diarization']['enabled'] is True
        assert payload['timestamp_granularities']==['segment','word']
        assert payload['input_audio']['format']=='flac'
        return Response()
    with patch('mai_pipeline.urllib.request.urlopen',side_effect=call) as api:
        transcribe_source(source,tmp_path/'asr','test',lambda *a:None)
        transcribe_source(source,tmp_path/'asr','test',lambda *a:None)
    assert api.call_count==1
    assert json.loads((tmp_path/'asr/raw.json').read_text('utf8'))==raw

def test_optional_speaker_corrections_are_strict():
    import pytest
    from mai_pipeline import parse_translations
    rows=[{'id':0,'speaker':'0'},{'id':1,'speaker':'1'},{'id':2,'speaker':'unknown'}]
    parsed=parse_translations('{"0":"첫 대사","1":["다음 대사","0"],"2":null}',rows)
    assert parsed=={'values':['첫 대사','다음 대사',None],'speaker_corrections':{'1':'0'}}
    for bad in ['{"0":"a","1":["b","99"],"2":null}',
        '{"0":"a","1":"b","2":["c","0"]}',
        '{"0":"a","0":"b","1":"c","2":null}']:
        with pytest.raises(ValueError):parse_translations(bad,rows)

def test_context_correction_changes_actual_reference_bank_and_survives_cache(tmp_path):
    from shared_reference import plan
    source=tmp_path/'source.wav';sf.write(source,np.zeros((SR*4,2),np.float32),SR)
    raw={'segments':[{'start':.2,'end':.8,'text':'one','speaker':0},
        {'start':1.2,'end':1.8,'text':'two','speaker':1},
        {'start':2.2,'end':2.8,'text':'three','speaker':1}]}
    def api(payload,key,**kwargs):
        return kwargs['parser']('{"0":"하나","1":["둘","0"],"2":"셋"}'),{}
    cfg={'key':'test','language':'ko'}
    with patch('mai_pipeline.transcribe_source',return_value=(raw,{'hash':'test'})),patch('speech.http_json',side_effect=api) as request:
        result=translate(source,tmp_path/'translation',tmp_path/'result.json',cfg,lambda *a:None)
        cached=translate(source,tmp_path/'translation',tmp_path/'result.json',cfg,lambda *a:None)
    assert request.call_count==1 and cached==result
    assert [s['speaker_id'] for s in result['segments']]==['0','0','1']
    groups=plan(result['segments'],result['speaker_timeline'])
    assert [(g['speaker'],g['indices']) for g in groups]==[('0',[0,1]),('1',[2])]
    assert result['source_regions'][1]['mai_speaker']=='1'
    assert result['speaker_corrections'][0]['source_region_id']==1

def test_word_gap_and_speaker_change_split_without_losing_punctuation():
    from mai_pipeline import word_regions
    raw={'segments':[{'start':0,'end':5,'text':'あ、はい。いいえ！次。','speaker':0}],
        'words':[{'start':0,'end':.2,'word':'あ','speaker':0},
        {'start':1.,'end':1.2,'word':'はい','speaker':0},
        {'start':1.3,'end':2.,'word':'いいえ','speaker':1},
        {'start':2.81,'end':3.,'word':'次','speaker':1}]}
    rows=word_regions(raw,5.)
    assert [(r['text'],r['speaker']) for r in rows]==[('あ、はい。','0'),('いいえ！','1'),('次。','1')]
    assert [(r['start'],r['end']) for r in rows]==[(0.,1.2),(1.3,2.),(2.81,3.)]

def test_wordless_region_is_preserved():
    from mai_pipeline import word_regions
    raw={'segments':[{'start':0,'end':1,'text':'はい。','speaker':0},
        {'start':2,'end':3,'text':'え？','speaker':1}],
        'words':[{'start':.1,'end':.9,'word':'はい','speaker':0}]}
    assert [r['text'] for r in word_regions(raw,3)]==['はい。','え？']


def test_late_asr_start_preserves_leading_syllable_across_short_consonant_gap(tmp_path):
    x=np.zeros((3*SR,2),np.float32);x[round(1.1*SR):round(1.26*SR)]=.1;x[round(1.32*SR):round(1.7*SR)]=.1
    source=tmp_path/'onset.wav';sf.write(source,x,SR)
    result,_=correct_edges(source,[{'start':1.48,'end':1.7,'speaker':'0'}])
    assert result[0]['start']<=1.1
    x[round(1.1*SR):round(1.26*SR)]=0;x[round(.9*SR):round(1.04*SR)]=.1
    sf.write(source,x,SR)
    result,_=correct_edges(source,[{'start':1.48,'end':1.7,'speaker':'0'}])
    assert result[0]['start']>1.04
