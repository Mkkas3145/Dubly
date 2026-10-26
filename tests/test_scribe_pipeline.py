import io,json
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import config,SR
from scribe_pipeline import transcribe
import scribe_pipeline as s
import pytest


def test_scribe_words_go_through_performance_review_before_translation(tmp_path):
    clean=tmp_path/'voice.wav';sf.write(clean,np.zeros((3*SR,2)),SR)
    raw={'text':'Hello, nice to meet you. Yes!','language_code':'en','words':[
        {'text':'Hello,','start':.1,'end':.4,'speaker_id':'speaker_7','type':'word'},
        {'text':'nice to meet you.','start':.6,'end':1.3,'speaker_id':'speaker_7','type':'word'},
        {'text':'(music)','start':1.4,'end':1.5,'type':'audio_event'},
        {'text':'Yes!','start':2.,'end':2.4,'speaker_id':'speaker_2','type':'word'}]}
    def translate(payload,key,work,identity,ids,kind,progress):
        context=json.loads(payload['messages'][1]['content'])
        assert context[0]['text']=='Hello, nice to meet you.'
        assert context[1]['text']=='Yes!'
        assert 'Scribe ASR' in payload['messages'][0]['content']
        return {'0':'안녕하세요, 반갑습니다.','1':'네!'},{'usage':{}}
    cfg={**config(),'key':'test','language':'ko','diarizer':'scribe-v2'}
    assert cfg['diarizer']=='scribe-v2' and not cfg['speaker_embeddings']
    with patch('scribe.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(raw).encode())) as api, \
         patch.object(s,'correct_fish_edges',side_effect=lambda clean,rows,progress:(rows,[])), \
         patch.object(s,'review_performance',side_effect=lambda clean,rows,*a:(rows,{'requests':0,'changes':[]})), \
         patch('qwen_punctuation.repair_source',side_effect=AssertionError('Qwen correction must not run')), \
         patch('fish_qwen_pipeline.stage_chat',side_effect=translate) as gemini:
        first=transcribe(clean,tmp_path/'work',tmp_path/'out.json',cfg,lambda *a:None)
        second=transcribe(clean,tmp_path/'work',tmp_path/'out.json',cfg,lambda *a:None)
    assert api.call_count==gemini.call_count==1
    payload=json.loads(api.call_args.args[0].data)
    assert payload['model']=='elevenlabs/scribe-v2'
    assert payload['provider']['options']['elevenlabs']=={'diarize':True,'tag_audio_events':False}
    assert first['source_languages']==['en']
    assert first['models']['delivery_cues']==s.QWEN and first['delivery_cue_receipt']['requests']==0
    assert first['models']['punctuation']==s.QWEN and 'punctuation_receipt' not in first
    assert first['segments'][0]['original_text']=='Hello, nice to meet you.'
    assert 'qwen_emotion_turns_speakers' in first['transcription_mode']
    assert first['speaker_authority']=='qwen/qwen3.8-omni-flash'
    assert first['native_speaker_mapping']==[{'native_id':'speaker_7','speaker_id':'0'},{'native_id':'speaker_2','speaker_id':'1'}]
    assert first['request_counts']['source_asr_api']==1
    assert second['request_counts']['source_asr_api']==0
    assert first['boundary_authority']=='scribe_v2' and not first['llm_timestamps']
    from reference_timing import refine_references
    with patch('reference_timing.source_evidence',side_effect=AssertionError('No second ASR')):
        same,_=refine_references(clean,first,tmp_path/'references')
    assert same is first


def test_diarization_options_invalidate_plain_asr_cache(tmp_path):
    from scribe import transcribe_words
    clean=tmp_path/'voice.wav';sf.write(clean,np.zeros(SR),SR)
    raw={'language_code':'ja','words':[{'text':'はい','start':.1,'end':.4,'speaker_id':'speaker_1'}]}
    def response(*args,**kwargs):return io.BytesIO(json.dumps(raw).encode())
    with patch('scribe.urllib.request.urlopen',side_effect=response) as api:
        transcribe_words(clean,tmp_path/'asr','test')
        result=transcribe_words(clean,tmp_path/'asr','test',provider_options=s.OPTIONS)
        transcribe_words(clean,tmp_path/'asr','test',provider_options=s.OPTIONS)
    assert api.call_count==2 and result['language']=='ja'


def test_corrected_gap_merge_happens_before_translation(tmp_path):
    clean=tmp_path/'voice.wav';sf.write(clean,np.zeros((4*SR,2)),SR)
    raw={'text':'First. Second.','language_code':'en','words':[
        {'text':'First.','start':.2,'end':.5,'speaker_id':'speaker_0'},
        {'text':'Second.','start':2.,'end':2.5,'speaker_id':'speaker_0'}]}
    def correct(clean,rows,progress):
        assert len(rows)==2
        return [{**rows[0],'start':.1,'end':1.3},{**rows[1],'start':1.4,'end':2.6}],[]
    def translate(payload,key,work,identity,ids,kind,progress):
        context=json.loads(payload['messages'][1]['content'])
        assert ids==[0] and len(context)==1
        assert 'First.' in context[0]['text'] and 'Second.' in context[0]['text']
        return {'0':'Combined translation.'},{'usage':{}}
    with patch('scribe.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(raw).encode())), \
         patch.object(s,'correct_fish_edges',side_effect=correct), \
         patch('collapsed_source_onsets.repair',side_effect=lambda clean,rows,*a,**kw:(rows,[])), \
         patch.object(s,'review_performance',side_effect=lambda clean,rows,*a:(rows,{'requests':0,'changes':[]})), \
         patch('fish_qwen_pipeline.stage_chat',side_effect=translate):
        result=s.transcribe(clean,tmp_path/'work',tmp_path/'out.json',
            {**config(),'key':'test','language':'ko'},lambda *a:None)
    assert len(result['segments'])==1
    assert result['reference_windows'][0]['start']==.1
    assert result['reference_windows'][0]['end']==2.6
    assert result['corrected_source_gap_merges']


def test_emotion_cuts_new_voices_and_nicknames_reach_translation_and_studio(tmp_path):
    import qwen_performance_turns as performance
    clean=tmp_path/'voice.wav';sf.write(clean,np.ones((2*SR,2))*.1,SR)
    raw={'text':'Hello. No!','language_code':'en','words':[
        {'text':'Hello.','start':.2,'end':.5,'speaker_id':'speaker_0'},
        {'text':'No!','start':.7,'end':1.,'speaker_id':'speaker_0'}]}
    def review(clean,rows,*args):
        assert len(rows)==1
        fixed=performance.fixed_rows(rows)
        revised,changes=performance.apply(fixed,{'segments':[{'id':0,'parts':[
            [0,0,0,'calm'],[1,1,27,'angry']]}]})
        return revised,{'requests':1,'changes':changes,'speaker_names':{'0':'차분한 목소리','27':'거친 목소리'}}
    def translate(payload,key,work,identity,ids,kind,progress):
        context=json.loads(payload['messages'][1]['content'])
        assert len(context)==2 and [r['speaker'] for r in context]==['0','27']
        assert [r['text'] for r in context]==['Hello.','No!']
        return {'0':'안녕.','1':'안 돼!'},{'usage':{}}
    with patch('scribe.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(raw).encode())), \
         patch.object(s,'correct_fish_edges',side_effect=lambda clean,rows,*a:(rows,[])), \
         patch('collapsed_source_onsets.repair',side_effect=lambda clean,rows,*a,**kw:(rows,[])), \
         patch.object(s,'review_performance',side_effect=review), \
         patch('fish_qwen_pipeline.stage_chat',side_effect=translate):
        result=s.transcribe(clean,tmp_path/'work',tmp_path/'out.json',{**config(),'key':'test','language':'ko'},lambda *a:None)
    assert len(result['segments'])==2
    assert [r['speaker_id'] for r in result['segments']]==['0','27']
    assert result['speaker_names']=={'0':'차분한 목소리','27':'거친 목소리'}
    assert [r['start'] for r in result['reference_windows']]==[.2,.7]
    assert result['request_counts']['performance_review']==1
    assert not result['llm_timestamps']


def test_sentence_and_internal_cues_reach_translation_without_timing_or_speaker_changes(tmp_path,monkeypatch):
 from moss_pipeline import transcribe as translate
 clean=tmp_path/'voice.wav';sf.write(clean,np.ones((2*SR,2))*.1,SR)
 raw={'model':s.MODEL,'language':'en','signature':{'policy':{}},'segments':[{'id':0,'start':.2,'end':1.5,'speaker':'0','text':'Hello. No!'}], 'native_speaker_mapping':[], 'source_word_boundary_repairs':[],'words':[]}
 monkeypatch.setattr(s,'source',lambda *a:raw)
 monkeypatch.setattr(s,'correct_fish_edges',lambda clean,rows,*a:(rows,[]))
 monkeypatch.setattr(s,'review_performance',lambda clean,rows,*a:(rows,{'requests':0,'changes':[]}))
 monkeypatch.setattr('collapsed_source_onsets.repair',lambda clean,rows,*a,**kw:(rows,[]))
 def review_with_cues(clean,rows,*a):return [{**rows[0],'text':'[warm] Hello. [firm] No!'}],{'requests':1,'changes':[]}
 monkeypatch.setattr(s,'review_performance',review_with_cues)
 def response(payload,*a):
  assert json.loads(payload['messages'][1]['content'])[0]['text']=='[warm] Hello. [firm] No!'
  return {'0':'[warm] 안녕. [firm] 안 돼!'},{'usage':{}}
 monkeypatch.setattr('fish_qwen_pipeline.stage_chat',response)
 result=s.transcribe(clean,tmp_path/'work',tmp_path/'out.json',{**config(),'key':'test','language':'ko'},lambda *a:None)
 row=result['segments'][0]
 assert row['original_text']=='[warm] Hello. [firm] No!'
 assert row['translated_text']=='[warm] 안녕. [firm] 안 돼!'
 assert (row['start'],row['end'],row['speaker_id'])==(.2,1.5,'0')
 assert result['request_counts']['delivery_cues']==0
 assert result['delivery_cue_receipt']['shared_with']=='performance_review'
 assert result['request_counts']['performance_review']==1
