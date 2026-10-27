import io,json
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import config,SR,GEMINI
from scribe_pipeline import transcribe
import scribe_pipeline as s


def test_legacy_scribe_transcription_with_conservative_audio_tags(tmp_path):
    clean=tmp_path/'voice.wav';sf.write(clean,np.zeros((3*SR,2)),SR)
    raw={'text':'Hello. Yes!','language_code':'en','words':[
        {'text':'Hello.','start':.1,'end':.6,'speaker_id':'speaker_7'},
        {'text':'Yes!','start':2.,'end':2.4,'speaker_id':'speaker_2'}]}
    def translate(payload,key,**kwargs):
        assert payload['model']==GEMINI=='google/gemini-4-argon'
        content=payload['messages'][1]['content']
        assert isinstance(content,list) and content[1]['type']=='input_audio'
        assert content[1]['input_audio']['data']
        context=json.loads(content[0]['text'])
        assert [row['text'] for row in context['regions']]==['Hello.','Yes!']
        assert [row['speaker'] for row in context['regions']]==['0','1']
        schema=payload['response_format']['json_schema']['schema']
        assert schema['required']==['0','1'] and schema['additionalProperties'] is False
        return kwargs['parser']('{"0":{"t":"안녕.","c":["shouting"]},"1":{"t":"네!","c":[]}}'),{}
    cfg={**config(),'key':'test','language':'ko'}
    assert cfg['diarizer']=='gemini-forced'
    with patch('scribe.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(raw).encode())) as api, \
         patch.object(s,'correct_fish_edges',side_effect=lambda clean,rows,*a:(rows,[])), \
         patch('procedural_delivery.annotate',side_effect=AssertionError('No caption stage')) as captions, \
         patch('speech.http_json',side_effect=translate) as gemini, \
         patch('qwen_performance_turns.review',side_effect=AssertionError('Qwen must not be called')) as review, \
         patch('gemini_source_pipeline.transcribe',side_effect=AssertionError('No Gemini audio transcription')):
        result=transcribe(clean,tmp_path/'work',tmp_path/'out.json',cfg,lambda *a:None)
        reused=transcribe(clean,tmp_path/'work',tmp_path/'out.json',cfg,lambda *a:None)
    assert api.call_count==gemini.call_count==1
    payload=json.loads(api.call_args.args[0].data)
    assert payload['model']=='elevenlabs/scribe-v2'
    assert payload['provider']['options']['elevenlabs']=={'diarize':True,'tag_audio_events':False}
    assert result['speaker_authority']=='elevenlabs/scribe-v2'
    assert result['llm_audio_input']['original_clock'] is True and result['llm_timestamps'] is False
    assert result['request_counts']['performance_review']==0
    review.assert_not_called()
    captions.assert_not_called()
    assert result['request_counts']['delivery_cues']==0
    assert reused['request_counts']['source_asr_api']==0
    assert result['segments'][0]['original_text']=='[shouting] Hello.'
    assert result['segments'][0]['translated_text']=='[shouting] 안녕.'
    assert result['segments'][1]['original_text']=='Yes!'
    assert result['segments'][1]['translated_text']=='네!'
    from reference_timing import refine_references
    assert refine_references(clean,result,tmp_path/'references')[0] is result


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



def test_corrected_gap_merge_preserves_words_and_scribe_speakers(tmp_path):
    clean=tmp_path/'voice.wav';sf.write(clean,np.zeros((4*SR,2)),SR)
    raw={'text':'First. Second.','language_code':'en','words':[
        {'text':'First.','start':.2,'end':.5,'speaker_id':'speaker_0'},
        {'text':'Second.','start':2.,'end':2.5,'speaker_id':'speaker_0'}]}
    def correct(clean,rows,progress):
        assert len(rows)==2
        return [{**rows[0],'start':.1,'end':1.3},{**rows[1],'start':1.4,'end':2.6}],[]
    def translate(payload,key,**kwargs):
        context=json.loads(payload['messages'][1]['content'][0]['text'])
        assert len(context['regions'])==1 and 'First.' in context['regions'][0]['text'] and 'Second.' in context['regions'][0]['text']
        return kwargs['parser']('{"0":"합쳐진 번역."}'),{}
    with patch('scribe.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(raw).encode())), \
         patch.object(s,'correct_fish_edges',side_effect=correct), \
         patch('procedural_delivery.annotate',side_effect=AssertionError('No caption stage')) as captions, \
         patch('speech.http_json',side_effect=translate), \
         patch('qwen_performance_turns.review',side_effect=AssertionError('Qwen must not be called')):
        result=s.transcribe(clean,tmp_path/'work',tmp_path/'out.json',{**config(),'key':'test','language':'ko'},lambda *a:None)
    assert len(result['segments'])==1
    assert result['reference_windows'][0]['start']==.1 and result['reference_windows'][0]['end']==2.6
    assert result['corrected_source_gap_merges']


def test_annotations_are_shared_and_neutral_lines_stay_untagged():
    rows=[{'id':0,'text':'Hello.','start':.1,'end':.5,'speaker':'4'},
          {'id':1,'text':'Yes.','start':1.,'end':1.5,'speaker':'6'}]
    parsed=s.translations('{"0":{"t":"안녕.","c":["Shouting","shouting"]},"1":{"t":"네.","c":[]}}',rows)
    revised,changes=s.tagged_source_rows(rows,parsed)
    assert revised[0]['text']=='[shouting] Hello.'
    assert parsed['values']==['[shouting] 안녕.','네.'] and revised[1]['text']=='Yes.'
    assert not changes and all(revised[i]['start']==r['start'] and revised[i]['end']==r['end'] and revised[i]['speaker']==r['speaker'] for i,r in enumerate(rows))
    assert rows[0]['text']=='Hello.'


def test_invalid_descriptive_tags_do_not_fail_translation_or_change_dialogue():
    rows=[{'id':0,'text':'Hello.'}]
    parsed=s.translations('{"0":{"t":"안녕.","c":["<forceful>","長い説明",null]}}',rows)
    assert parsed['values']==['안녕.'] and parsed['delivery_cues']=={'0':[]}
    assert parsed['rejected_delivery_tags']


def test_detailed_sentence_cues_and_audible_target_cutoffs_keep_source_words():
    row={'id':0,'text':'Wait. Stop!','speaker':'4','start':1.,'end':3.}
    first='speaking through a held breath, tightly pressed onset, releasing the words with a strained groan'
    second='shouting, voice breaking'
    raw={'0':{'t':f'[{first}] 잠ㄲ… [{second}] 멈춰!',
        'c':[{'i':0,'c':first},{'i':2,'c':second}]}}
    parsed=s.translations(json.dumps(raw,ensure_ascii=False),[row])
    revised,changes=s.tagged_source_rows([row],parsed)
    assert revised[0]['text']==f'[{first}] Wait. [{second}] Stop!'
    assert parsed['values'][0]==raw['0']['t']
    assert row['text']=='Wait. Stop!' and not changes
    assert revised[0]['speaker']=='4' and revised[0]['start']==1. and revised[0]['end']==3.
    cached=s.translations(json.dumps(parsed['response_entries'],ensure_ascii=False),[row])
    assert cached['values']==parsed['values']
    assert cached['delivery_cues']==parsed['delivery_cues']
