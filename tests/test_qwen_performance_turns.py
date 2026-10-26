import copy,json
from unittest.mock import patch
import pytest
import numpy as np
import soundfile as sf
import qwen_performance_turns as q


def rows():
    return q.fixed_rows([
        {'id':0,'start':.1,'end':2.,'speaker':'0','text':'Hello, no!', 'words':[
            {'word':'Hello,','start':.2,'end':.6,'space_after':True},
            {'word':'no!','start':1.,'end':1.6,'space_after':False}]},
        {'id':1,'start':3.,'end':4.,'speaker':'0','text':'Hello.', 'words':[
            {'word':'Hello.','start':3.1,'end':3.7,'space_after':False}]}])


def canonical():
    return {'speaker_names':{'0':'차분한 목소리','27':'거친 목소리'},'segments':[
        {'id':0,'parts':[{'first_word_id':0,'last_word_id':0,'speaker':0,'emotion':'calm'},
                         {'first_word_id':1,'last_word_id':1,'speaker':27,'emotion':'angry'}]},
        {'id':1,'parts':[{'first_word_id':2,'last_word_id':2,'speaker':27,'emotion':'calm'}]}]}


def test_mistyped_ranges_recover_only_from_exact_full_text_and_keep_voice_and_times():
 source=rows()[:1];before=copy.deepcopy(source)
 value={'segments':[{'id':0,'parts':[
  {'w':[0,0],'s':0,'t':'[low, soft] Hello,'},
  {'w':[0,0],'s':27,'t':'[hesitant] no!'}]}]}
 revised,audit=q.apply(source,value)
 assert [r['speaker'] for r in revised]==['0','27']
 assert [w['word_id'] for r in revised for w in r['words']]==[0,1]
 assert [(r['start'],r['end']) for r in revised]==[(.1,.6),(1.,2.)]
 assert audit[0]['word_ranges_recovered_from_exact_text']
 assert source==before


@pytest.mark.parametrize('texts',[['Hello,'],['Hel','lo, no!'],['Hello,','yes!']])
def test_range_recovery_cannot_complete_missing_words_or_cut_inside_word(texts):
 source=rows()[:1]
 parts=[{'first_word_id':0,'last_word_id':0,'speaker':0,'text':t} for t in texts]
 assert q.ranges_from_locked_text(source[0]['words'],parts) is None
 with pytest.raises(ValueError):q.apply(source,{'segments':[{'id':0,'parts':parts}]})


def test_range_recovery_preserves_numeric_lexical_punctuation():
 words=[{'word_id':0,'word':'1.5'},{'word_id':1,'word':'percent.'}]
 part={'first_word_id':0,'last_word_id':0,'speaker':0,'text':'[steady] 1.5 percent?!'}
 fixed=q.ranges_from_locked_text(words,[part]);assert fixed[0]['last_word_id']==1
 assert q.ranges_from_locked_text(words,[dict(part,text='[steady] 15 percent')]) is None


def test_annotations_only_constructs_locked_speech_and_inline_cues_without_echo():
 source=rows()[:1];before=copy.deepcopy(source)
 value={'segments':[{'id':0,'parts':[{'e':None,'s':27,'c':'soft low voice',
        'm':[{'i':1,'c':'hesitant pause'}],'p':[{'i':1,'a':'?!'}]}]}]}
 normalized=q.normalize(value,source)
 assert 'text' not in normalized['segments'][0]['parts'][0]
 revised,_=q.apply(source,normalized)
 assert revised[0]['speaker']=='27' and revised[0]['text']=='[soft low voice] Hello, [hesitant pause] no?!'
 assert (revised[0]['start'],revised[0]['end'])==(.1,2.)
 assert revised[0]['words']==[{**w,'speaker':'27'} for w in source[0]['words']]
 assert source==before


def test_annotation_partition_implicit_starts_and_final_remainder_preserve_every_word():
 source=rows()[:1]
 value={'segments':[{'id':0,'parts':[{'e':0,'s':0,'c':'steady quiet voice'},
                                  {'e':None,'s':27,'c':'strained raised voice'}]}]}
 revised,_=q.apply(source,value)
 assert [r['text'] for r in revised]==['[steady quiet voice] Hello,','[strained raised voice] no!']
 assert [w['word_id'] for r in revised for w in r['words']]==[0,1]


def test_annotation_cannot_add_spoken_words_through_punctuation():
 value={'segments':[{'id':0,'parts':[{'e':None,'s':0,'c':'low steady voice','p':[{'i':1,'a':'invented word'}]}]}]}
 revised,_=q.apply(rows()[:1],value)
 assert revised[0]['text']=='[low steady voice] Hello, no!'


def test_annotation_provider_wrappers_preserve_identity_and_cues():
 value={'segments':[{'id':0,'parts':[{'e':[None],'s':[27],'c':['soft ','low voice'],
        'm':{'i':1,'c':'hesitant'},'p':{'i':1,'a':'?'}}]}]}
 revised,_=q.apply(rows()[:1],value)
 assert revised[0]['speaker']=='27' and revised[0]['text']=='[soft low voice] Hello, [hesitant] no?'
 revised,_=q.apply(rows()[:1],{'segments':[{'id':0,'parts':[[None,27,'low voice']]}]})
 assert revised[0]['speaker']=='27' and revised[0]['text']=='[low voice] Hello, no!'


@pytest.mark.parametrize('value',[
    lambda v:v,
    lambda v:'Here is the result:\n```json\n'+json.dumps(v)+'\n```\nDone.',
    lambda v:"```python\n"+repr(v)+"\n```",
    lambda v:'Result follows:\n'+repr(v)+'\nReview complete.',
    lambda v:{'output':{'results':list(reversed(v['segments']))}},
    lambda v:{str(r['id']):r['parts'] for r in v['segments']},
    lambda v:[[r['id'],[[p['first_word_id'],p['last_word_id'],p['speaker'],p['emotion']] for p in r['parts']]] for r in v['segments']],
    lambda v:{'turns':[[p['first_word_id'],p['last_word_id'],p['speaker'],p['emotion']] for r in v['segments'] for p in r['parts']]},
    lambda v:{'segments':[{'id':r['id'],**p} for r in v['segments'] for p in r['parts']]},
    lambda v:{'segments':[{'region_id':str(r['id']),'splits':[{'w':[p['first_word_id'],p['last_word_id']],
        'speaker_id':f'speaker_{p["speaker"]}','delivery':[p['emotion']],'irrelevant_note':'ignored'} for p in reversed(r['parts'])]} for r in v['segments']]},
])
def test_format_variants_keep_new_speaker_identity_and_emotion_cuts(value):
    source=rows();before=copy.deepcopy(source)
    result,audit=q.apply(source,value(canonical()))
    assert [r['speaker'] for r in result]==['0','27','27']
    assert [(r['start'],r['end']) for r in result]==[(.1,.6),(1.,2.),(3.,4.)]
    assert [w['word_id'] for r in result for w in r['words']]==[0,1,2]
    assert [r['audible_emotion'] for r in result]==['calm','angry','calm']
    assert source==before and len(audit)==2


@pytest.mark.parametrize('mutate',[
    lambda v:v['segments'].pop(),
    lambda v:v['segments'][0]['parts'].pop(),
    lambda v:v['segments'][0]['parts'][1].update(first_word_id=0),
    lambda v:v['segments'][0]['parts'][1].update(last_word_id=2),
    lambda v:v['segments'][0]['parts'][0].update(first_word_id=999),
    lambda v:v['segments'].append(copy.deepcopy(v['segments'][0])),
])
def test_invalid_word_coverage_is_rejected_even_in_well_formed_json(mutate):
    value=canonical();mutate(value)
    with pytest.raises(ValueError):q.apply(rows(),value)


def test_same_voice_emotion_change_does_not_need_a_new_identity():
    value=canonical()
    for r in value['segments']:
        for p in r['parts']:p['speaker']=0
    revised,_=q.apply(rows(),value)
    assert len(revised)==3 and all(r['speaker']=='0' for r in revised)


def test_omitted_punctuation_point_tokens_do_not_force_another_model_request():
    source=[{'id':0,'start':0.,'end':2.,'speaker':'0','text':'Hello. No!','words':[
        {'word':'Hello','start':.1,'end':.4,'word_id':0,'space_after':False},
        {'word':'.','start':.4,'end':.4,'word_id':1,'space_after':True},
        {'word':'No','start':1.,'end':1.4,'word_id':2,'space_after':False},
        {'word':'!','start':1.4,'end':1.4,'word_id':3,'space_after':False}]}]
    revised,audit=q.apply(source,{'turns':[[0,0,0,'calm'],[2,2,27,'angry']]})
    assert [r['speaker'] for r in revised]==['0','27']
    assert [w['word_id'] for r in revised for w in r['words']]==[0,1,2,3]
    assert audit[0]['punctuation_word_ids_restored']==[1,3]


def test_unknown_voice_retained_without_borrowing_identity():
    value=canonical();value['segments'][0]['parts'][1]['speaker']=None
    revised,_=q.apply(rows(),value)
    assert revised[1]['speaker']=='unknown'


def test_review_disables_reasoning_preserves_audio_and_caches_validated_response(tmp_path):
    clean=tmp_path/'clean.wav';sf.write(clean,np.ones(4*16000)*.1,16000)
    audio=tmp_path/'audio.mp3';audio.write_bytes(b'audio')
    manifest={'sha256':'audio','duration_seconds':4.,'format':'mp3','silence_removed':False}
    seen=[]
    def request(payload,key,folder,ids,progress,**kwargs):
        seen.append(payload)
        assert payload['reasoning']=={'enabled':False}
        part_schema=payload['response_format']['json_schema']['schema']['properties']['segments']['items']['properties']['parts']['items']
        assert 'emotion' not in part_schema['properties'] and 'emotion' not in part_schema['required']
        assert 'ANY new nonnegative integer' in payload['messages'][0]['content']
        assert 'NO 8/16-speaker cap' in payload['messages'][0]['content']
        assert 'Do NOT trust ASR speaker IDs, even as reference evidence' in payload['messages'][0]['content']
        assert 'preserving ASR numbering is not a goal' in payload['messages'][0]['content']
        assert 'Reuse one output speaker ID whenever the same voice returns' in payload['messages'][0]['content']
        assert 'Do not force different speaker IDs solely because' in payload['messages'][0]['content']
        assert 'Character names, inferred roles and guessed nicknames must never override' in payload['messages'][0]['content']
        assert 'Different story characters/persons MUST' not in payload['messages'][0]['content']
        assert 'Prefer keeping a correct existing ID' not in payload['messages'][0]['content']
        assert 'dialogue content, names, forms of address, relationships' in payload['messages'][0]['content']
        assert 'NICKNAME_LANGUAGE' not in payload['messages'][0]['content']
        assert 'Prioritize complete sentences and coherent semantic units' in payload['messages'][0]['content']
        assert 'NEED FOR INDEPENDENT LOCAL REFERENCE AUDIO' in payload['messages'][0]['content']
        assert 'not by whether the change is abrupt' in payload['messages'][0]['content']
        assert 'When there is no clear need for independent references, prefer ONE part and use inline cues' in payload['messages'][0]['content']
        assert 'Topic, scene or background-music changes alone never justify a split' in payload['messages'][0]['content']
        assert 'dependent construction from its continuation' in payload['messages'][0]['content']
        assert 'An actual change of speaker must remain separated' in payload['messages'][0]['content']
        return canonical(),{'usage':{},'requests':1}
    with patch('qwen_audio_pipeline.encode_audio',return_value=(audio,manifest)),patch.object(q,'request',side_effect=request):
        first,receipt=q.review(clean,rows(),tmp_path,{'key':'test','language':'ko'})
        second,cached=q.review(clean,rows(),tmp_path,{'key':'test','language':'ko'})
    assert first==second and len(seen)==1
    assert receipt['requests']==1 and cached['requests']==0
    assert receipt['speaker_names']=={'0':'차분한 목소리','27':'거친 목소리'}


def test_invalid_partition_is_retried_without_losing_asr_words(tmp_path):
    clean=tmp_path/'clean.wav';sf.write(clean,np.ones(4*16000)*.1,16000)
    audio=tmp_path/'audio.mp3';audio.write_bytes(b'audio')
    invalid=canonical();invalid['segments'].pop()
    with patch('qwen_audio_pipeline.encode_audio',return_value=(audio,{'sha256':'audio'})), \
         patch.object(q,'request',side_effect=[(invalid,{}),(canonical(),{})]) as api:
        result,receipt=q.review(clean,rows(),tmp_path,{'key':'test','language':'ko'})
    assert api.call_count==2 and receipt['requests']==2
    assert len(result)==3


@pytest.mark.parametrize('names',[
    {'speakers':[{'id':0,'nickname':'차분한 목소리'},{'speaker_id':27,'name':'거친 목소리'}]},
    {'speaker_names':{'0':'차분한 목소리','27':'거친 목소리'}},
    {'nicknames':{'0':{'alias':'차분한 목소리'},'27':{'label':'거친 목소리'}}},
    {'speakers':[[0,'차분한 목소리'],[27,'거친 목소리']]},
])
def test_nickname_format_variants(names):
    value={'segments':canonical()['segments'],**names}
    assert q.normalize(value)['speaker_names']=={'0':'차분한 목소리','27':'거친 목소리'}

def test_single_part_with_collapsed_word_times_keeps_corrected_acoustic_region():
 source=q.fixed_rows([{'id':0,'speaker':'3','start':266.439,'end':266.679,'text':'えっ',
  'words':[{'word':'えっ','start':266.459,'end':266.459},{'word':'','start':266.459,'end':266.459}]}])
 value={'segments':[{'id':0,'parts':[{'first_word_id':0,'last_word_id':1,'speaker':3,'emotion':'gasping'}]}]}
 revised,_=q.apply(source,value)
 assert (revised[0]['start'],revised[0]['end'])==(266.439,266.679)
 assert revised[0]['words']==[{**w,'speaker':'3'} for w in source[0]['words']]
 assert revised[0]['text']=='えっ'
 value['segments'][0]['parts']=[{'first_word_id':0,'last_word_id':0,'speaker':3},{'first_word_id':1,'last_word_id':1,'speaker':3}]
 revised,audit=q.apply(source,value)
 assert len(revised)==1 and revised[0]['text']==source[0]['text']
 assert audit[0]['unapplied_splits']


def test_missing_nicknames_are_repaired_separately_without_audio_or_repartitioning(tmp_path):
 value=canonical();value.pop('speaker_names');revised,_=q.apply(rows(),value);calls=[]
 def request(payload,key,folder,ids,progress,**kwargs):
  calls.append(payload)
  assert payload['reasoning']=={'enabled':False}
  assert all(isinstance(m['content'],str) for m in payload['messages'])
  return kwargs['normalizer']({'speakers':[{'id':0,'nickname':'주인공'},{'id':27,'nickname':'친구'}]}),{}
 with patch.object(q,'request',side_effect=request):
  result,report=q.complete_nicknames(value,revised,{'key':'test','language':'ko'},tmp_path,lambda *a:None)
 assert result['speaker_names']=={'0':'주인공','27':'친구'}
 assert result['segments']==value['segments'] and report['requests']==1 and not report['missing_speaker_ids']
 assert len(calls)==1


def test_nickname_provider_failure_preserves_validated_turns(tmp_path):
 value=canonical();value.pop('speaker_names');revised,_=q.apply(rows(),value)
 with patch.object(q,'request',side_effect=RuntimeError('unavailable')):
  result,report=q.complete_nicknames(value,revised,{'key':'test','language':'ko'},tmp_path,lambda *a:None)
 assert result['segments']==value['segments'] and report['missing_speaker_ids']==['0','27']
 assert report['requests']==2

@pytest.mark.parametrize('text',['Hello?! no~','Hello... no!!','Hello, no?','Hello, no……'])
def test_punctuation_only_changes_are_applied_without_moving_asr_words(text):
 source=rows();value={'segments':[{'id':0,'parts':[{'first_word_id':0,'last_word_id':1,'speaker':0,'emotion':'surprised','text':text}]}]}
 before=copy.deepcopy(source[0]['words'])
 revised,audit=q.apply(source[:1],value)
 assert revised[0]['text']==text and revised[0]['words']==[{**w,'speaker':'0'} for w in before]
 assert (revised[0]['start'],revised[0]['end'])==(.1,2.)
 assert audit[0]['punctuation_edits'][0]['after']==text


@pytest.mark.parametrize('text',['Hello, yes!','Hello, no! [angry]','H, Hello, no!','Hello no 2!'])
def test_punctuation_field_cannot_rewrite_words_or_add_cues(text):
 value={'segments':[{'id':0,'parts':[{'first_word_id':0,'last_word_id':1,'speaker':0,'text':text}]}]}
 revised,audit=q.apply(rows()[:1],value)
 assert revised[0]['text']==rows()[0]['text']
 assert audit[0]['rejected_punctuation'][0]['rejected']==text


def test_numeric_punctuation_is_not_editable():
 source=q.fixed_rows([{'id':0,'start':0.,'end':1.,'speaker':'0','text':'1.5!', 'words':[{'word':'1.5','start':.1,'end':.9}]}])
 for text in ['15!','1,5!']:
  revised,audit=q.apply(source,{'segments':[{'id':0,'parts':[{'first_word_id':0,'last_word_id':0,'speaker':0,'text':text}]}]})
  assert revised[0]['text']=='1.5!' and audit[0]['rejected_punctuation']
 revised,_=q.apply(source,{'segments':[{'id':0,'parts':[{'first_word_id':0,'last_word_id':0,'speaker':0,'text':'1.5?!'}]}]})
 assert revised[0]['text']=='1.5?!'


@pytest.mark.parametrize('other_voice',[9,12])
def test_stale_word_clock_cannot_abort_corrected_region_or_merge_distinct_voices(other_voice):
 source=q.fixed_rows([{'id':0,'speaker':'9','start':10.,'end':12.,'text':'One two three',
  'words':[{'word':'One','start':1.,'end':1.3,'space_after':True},
           {'word':'two','start':10.2,'end':10.2,'space_after':True},
           {'word':'three','start':11.,'end':11.}]}])
 before=copy.deepcopy(source)
 value={'segments':[{'id':0,'parts':[{'first_word_id':0,'last_word_id':1,'speaker':9,'emotion':'calm'},
  {'first_word_id':2,'last_word_id':2,'speaker':other_voice,'emotion':'hesitant'}]}]}
 revised,audit=q.apply(source,value)
 assert source==before
 assert len(revised)==1 and (revised[0]['start'],revised[0]['end'])==(10.,12.)
 assert revised[0]['text']==source[0]['text']
 assert revised[0]['speaker']==('9' if other_voice==9 else 'unknown')
 assert [(w['start'],w['end']) for w in revised[0]['words']]==[(w['start'],w['end']) for w in source[0]['words']]
 assert audit[0]['unapplied_splits']
 context={'words':[[w['word_id'],w['start'],w['end'],w['word']] for w in source[0]['words']]}
 supplied=q.review_region_context(context,source[0])
 assert supplied['split_before_word_ids']==[]
 assert all(w[1:3]==[None,None] for w in supplied['words'])


def test_corrected_clock_exposes_only_usable_split_anchors():
 source=rows()[0]
 context={'words':[[w['word_id'],w['start'],w['end'],w['word']] for w in source['words']]}
 supplied=q.review_region_context(context,source)
 assert supplied['split_before_word_ids']==[1]
 assert supplied['words']==context['words']




def test_foreign_nicknames_are_kept_without_a_localization_request(tmp_path):
 value=canonical();value['speaker_names']={'0':'レナ','27':'圭一'}
 revised,_=q.apply(rows(),value)
 with patch.object(q,'request',side_effect=AssertionError('No extra localization request')) as request:
  result,report=q.complete_nicknames(value,revised,{'key':'test','language':'ko'},tmp_path,lambda *a:None)
 assert result['speaker_names']=={'0':'レナ','27':'圭一'}
 assert result['segments']==value['segments'] and report['requests']==0
 request.assert_not_called()


def test_nickname_language_is_instructed_in_initial_request():
 from qwen_nickname_language import instruction
 prompt=instruction('ko')
 assert 'Korean' in prompt and 'dubbing OUTPUT language' in prompt
 assert 'transliterate personal names' in prompt


def test_same_review_preserves_inline_english_cues_without_changing_voice_or_words():
 source=rows();value=canonical()
 value['segments'][0]['parts'][0]['text']='[warm] Hello,'
 value['segments'][0]['parts'][1]['text']='[firm] no!'
 value['segments'][1]['parts'][0]['text']='[hesitant] Hello.'
 revised,_=q.apply(source,value)
 assert [r['text'] for r in revised]==['[warm] Hello,','[firm] no!','[hesitant] Hello.']
 assert [r['speaker'] for r in revised]==['0','27','27']
 assert q.annotated_text_error('[laughing] Hello, no!','Hello, no!') is None
 assert q.annotated_text_error('[笑い] Hello, no!','Hello, no!')=='invalid_english_cue'
 assert q.annotated_text_error('[laughing] Hello, yes!','Hello, no!')=='changed_spoken_text'
 assert 'for FOUR tasks' in q.PROMPT


def test_inline_cues_and_splits_work_without_separate_emotion_labels():
 value=canonical()
 for entry in value['segments']:
  for part in entry['parts']:part.pop('emotion',None)
 value['segments'][0]['parts'][0]['text']='[warm] Hello,'
 revised,_=q.apply(rows(),value)
 assert len(revised)==3 and revised[0]['text']=='[warm] Hello,'
 assert [r['speaker'] for r in revised]==['0','27','27']
 assert '"emotion":' not in q.PROMPT


def test_acting_cues_keep_both_intent_and_actual_audible_delivery():
 from qwen_delivery_cues import SOURCE_PROMPT
 for prompt in [q.PROMPT,SOURCE_PROMPT]:
  assert 'BOTH communicative intent and audible delivery' in prompt
  assert 'quiet, gentle and low-voiced' in prompt
  assert 'merely because of its meaning or punctuation' in prompt
  assert 'do not erase its intent just because the voice is soft' in prompt


def test_each_line_and_split_part_requires_its_own_english_cue_in_prompt():
 from qwen_delivery_cues import SOURCE_PROMPT
 for prompt in [q.PROMPT,SOURCE_PROMPT]:
  assert 'EVERY output dialogue line and EVERY split part MUST begin' in prompt
  assert 'No exceptions for short replies' in prompt
  assert 'A cue on a preceding line or part does not count' in prompt
  assert 'rather than omitting the cue or inventing an emotion' in prompt
