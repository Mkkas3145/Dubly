import json
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR,config
from speech import transcribe
from mai_pipeline import correct_edges


def test_active_mai_one_asr_gemini_text_only_qwen_source_cues_before_translation(tmp_path):
    clean=tmp_path/'clean.wav';sf.write(clean,np.zeros((SR*3,2)),SR)
    raw={'segments':[{'start':.2,'end':.9,'text':'本当？','speaker':4},
                     {'start':1.3,'end':1.9,'text':'はい！','speaker':7}]}
    def translate(payload,key,work,signature,ids,kind,progress):
        content=payload['messages'][1]['content']
        assert isinstance(content,str) and 'input_audio' not in json.dumps(payload)
        context=json.loads(content)
        assert all(set(r)=={'id','speaker','text'} for r in context)
        assert [r['speaker'] for r in context]==['4','7']
        assert 'fixed by MAI' in payload['messages'][0]['content']
        assert 'Preserve every supplied English [] performance cue verbatim' in payload['messages'][0]['content']
        assert context[0]['text']=='[surprised] 本当？'
        return {'0':'[surprised] 정말?','1':'네!'},{'usage':{}}
    def cues(payload,key,folder,ids,progress):
        assert payload['reasoning']=={'enabled':False}
        assert payload['model']=='qwen/qwen3.8-omni-flash'
        assert payload['messages'][1]['content'][1]['type']=='input_audio'
        return {'segments':[{'id':0,'text':'[surprised] 本当？'},{'id':1,'text':'はい！'}]},{'requests':1}
    with patch('mai_native_pipeline.transcribe_source',return_value=(raw,{'test':'mai'})) as mai, \
         patch('moss_pro.source',side_effect=AssertionError('No MOSS')), \
         patch('moss_fish_regions.transcribe_regions',side_effect=AssertionError('No Fish ASR')), \
         patch('fish_qwen_pipeline.stage_chat',side_effect=translate), \
         patch('qwen_delivery_cues.request',side_effect=cues) as qwen:
        result=transcribe(clean,tmp_path/'translation',tmp_path/'result.json',{**config(),'key':'test','diarizer':'mai-transcribe-2'},lambda *a:None)
    assert mai.call_count==1
    assert result['models']['source_asr']==result['models']['speakers']=='microsoft/mai-transcribe-2'
    assert result['boundary_authority']=='mai_transcribe_2' and result['fish_asr_requests']==0
    assert result['llm_audio_input'] is None
    assert result['delivery_cue_receipt']['audio']['bitrate_kbps']==64
    assert qwen.call_count==1 and result['request_counts']['delivery_cues']==1
    assert result['segments'][0]['original_text']=='[surprised] 本当？'
    assert result['segments'][0]['translated_text']=='[surprised] 정말?'
    assert result['request_counts']['translation']==1
    # Complete same-input retries reuse both the compressed audio and translation.
    with patch('mai_native_pipeline.transcribe_source',return_value=(raw,{'test':'mai'})), \
         patch('fish_qwen_pipeline.stage_chat',side_effect=AssertionError('reuse translation')), \
         patch('qwen_delivery_cues.request',side_effect=AssertionError('reuse cues')):
        cached=transcribe(clean,tmp_path/'translation',tmp_path/'result.json',{**config(),'key':'test','diarizer':'mai-transcribe-2'},lambda *a:None)
    assert cached['request_counts']['translation']==0
    assert cached['request_counts']['delivery_cues']==0
    assert len(result['boundary_measurements'])==2
    assert [r['speaker_id'] for r in result['segments']]==['4','7']
    assert [(w['start'],w['end']) for w in result['reference_windows']]==[(s['start'],s['end']) for s in result['segments']]


def test_active_waveform_inside_asr_edges_recovers_body_without_crossing_neighbor(tmp_path):
    clean=tmp_path/'clean.wav';v=np.zeros((SR*3,2),np.float32)
    for a,b in ((.35,.9),(1.3,1.8)):
        t=np.arange(round((b-a)*SR))/SR
        v[round(a*SR):round(a*SR)+len(t)]=(.2*np.sin(2*np.pi*180*t))[:,None]
    sf.write(clean,v,SR)
    rows=[{'id':0,'start':.5,'end':.75,'speaker':'4','text':'first'},
          {'id':1,'start':1.4,'end':1.65,'speaker':'7','text':'second'}]
    corrected,audit=correct_edges(clean,rows)
    assert corrected[0]['start']<=.36 and corrected[0]['end']>=.89
    assert corrected[1]['start']<=1.31 and corrected[1]['end']>=1.79
    assert corrected[0]['end']<=corrected[1]['start']
    assert [r['speaker'] for r in corrected]==['4','7']
