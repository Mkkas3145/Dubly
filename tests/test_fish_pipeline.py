import json
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR,config
from fish_pipeline import regions,translate,AUTHORITY,MODEL
import pytest

@pytest.mark.parametrize('tail,expected',[(2.,2),(.4,1)])
def test_long_silent_word_tail_does_not_bridge_pause_threshold(tmp_path,tail,expected):
    clean=tmp_path/'clean.wav';second=1.+tail+.2
    values=np.zeros((round((second+1)*SR),2),np.float32)
    values[:SR]=.2;values[round(second*SR):]=.1
    sf.write(clean,values,SR,subtype='FLOAT')
    raw={'text':'<|speaker:0|>[怒鳴る]嘘だ！[平然]次です。','words':[
        {'word':'嘘','start':0.,'end':.2},
        {'word':'だ','start':.2,'end':1.+tail},
        {'word':'次です','start':second,'end':second+1}]}
    result=regions(raw,second+1,clean=clean)
    assert len(result)==expected
    assert raw['words'][1]['end']==1.+tail
    if expected==2:
        assert result[0]['end']==pytest.approx(1.04)
        assert result[1]['start']==second
        assert result[0]['words'][-1]['asr_end']==3.
        assert '[怒鳴る]' in result[0]['text'] and '[平然]' in result[1]['text']


def test_sustained_voice_is_not_shortened_just_because_word_is_long(tmp_path):
    clean=tmp_path/'clean.wav';sf.write(clean,np.full((4*SR,2),.2,np.float32),SR)
    raw={'words':[{'word':'あああ','start':0.,'end':3.},
                  {'word':'次','start':3.2,'end':4.}]}
    result=regions(raw,4.,clean=clean)
    assert len(result)==1 and result[0]['words'][0]['end']==3.


@pytest.mark.parametrize('punctuation,intervening_voice,restored',[
    ('、',False,True),('。',False,False),('、',True,False)])
def test_collapsed_continuation_requires_unfinished_phrase_and_no_separate_voice(tmp_path,punctuation,intervening_voice,restored):
    from fish_pipeline import correct_fish_edges
    clean=tmp_path/'clean.wav';values=np.zeros((4*SR,2),np.float32)
    values[round(.2*SR):SR]=.2;values[3*SR:round(3.5*SR)]=.15
    if intervening_voice:values[round(1.6*SR):round(2.1*SR)]=.1
    sf.write(clean,values,SR,subtype='FLOAT')
    raw={'text':f'<|speaker:1|>してないよ{punctuation}全然。<|speaker:0|>嘘だろ。',
         'words':[{'word':'してない','start':.2,'end':.6},
                  {'word':'よ','start':.6,'end':1.},
                  {'word':'全然','start':3.,'end':3.},
                  {'word':'嘘だろ','start':3.,'end':3.5}]}
    rows=regions(raw,4.,clean=clean);fixed,audit=correct_fish_edges(clean,rows)
    assert raw['words'][2]['start']==3. and raw['words'][2]['end']==3.
    if restored:
        assert len(rows)==2 and rows[0]['text']=='してないよ、全然。'
        assert rows[0]['end']==1. and rows[1]['speaker']=='0'
        assert rows[0]['words'][-1]['start']==rows[0]['words'][-1]['end']==1.
        assert audit[0]['word_timing_repairs'][0]['asr_start']==3.
    else:
        assert len(rows)==3 and rows[1]['text']=='全然。'
        assert fixed[1]['keep_original']
        assert fixed[1]['start']==fixed[1]['end']==3.
        assert audit[1]['method']=='unresolved_point_at_other_voice_boundary_keep_original'


def test_collapsed_continuation_cannot_join_previous_different_speaker(tmp_path):
    from fish_pipeline import correct_fish_edges
    clean=tmp_path/'clean.wav';sf.write(clean,np.full((4*SR,2),.2,np.float32),SR)
    raw={'text':'<|speaker:0|>前、<|speaker:1|>全然。<|speaker:0|>次。','words':[
        {'word':'前','start':.2,'end':1.}, {'word':'全然','start':3.,'end':3.},
        {'word':'次','start':3.,'end':3.5}]}
    rows=regions(raw,4.,clean=clean);fixed,audit=correct_fish_edges(clean,rows)
    assert len(rows)==3 and rows[1]['speaker']=='1' and fixed[1]['keep_original']


@pytest.mark.parametrize('second_island',[False,True])
def test_point_reply_uses_unique_island_not_previous_voice_decay(tmp_path,second_island):
    from fish_pipeline import correct_fish_edges
    clean=tmp_path/'clean.wav';values=np.zeros((4*SR,2),np.float32)
    values[round(.3*SR):round(1.13*SR)]=.1
    values[round(1.7*SR):round(2.*SR)]=.02
    if second_island:values[round(2.5*SR):round(2.7*SR)]=.02
    values[3*SR:round(3.5*SR)]=.1
    sf.write(clean,values,SR,subtype='FLOAT')
    raw={'text':'<|speaker:1|>前。<|speaker:0|>うん、次。','words':[
        {'word':'前','start':.3,'end':1.}, {'word':'うん','start':1.,'end':1.},
        {'word':'次','start':3.,'end':3.5}]}
    rows=regions(raw,4.,clean=clean);fixed,audit=correct_fish_edges(clean,rows)
    if second_island:
        assert not fixed[1].get('point_recovery')
    else:
        assert fixed[1]['start']==pytest.approx(1.67)
        assert fixed[1]['end']==pytest.approx(2.05)
        assert audit[1]['method']=='unique_isolated_reply_after_other_voice_tail'
        assert audit[1]['original_start']==1.


def test_one_source_number_is_not_split_by_fragmented_digit_timestamps():
    raw={'text':'<|speaker:1|>100。<|speaker:0|>100って高いの？','words':[
        {'word':'1','start':1.,'end':1.08}, {'word':'0','start':2.,'end':2.},
        {'word':'0','start':2.,'end':2.}, {'word':'1','start':2.3,'end':2.5},
        {'word':'0','start':2.5,'end':2.5}, {'word':'0','start':2.5,'end':2.5},
        {'word':'って','start':2.5,'end':2.5}, {'word':'高い','start':2.5,'end':2.9},
        {'word':'の','start':2.9,'end':3.}]}
    rows=regions(raw,4.)
    assert len(rows)==2 and rows[0]['text']=='100。'
    assert rows[0]['start']==1. and rows[0]['end']==2.
    assert rows[1]['words'][0]['word']=='100'


def test_separate_source_numbers_are_not_joined_as_one_literal():
    raw={'text':'<|speaker:0|>1、0、0。','words':[
        {'word':'1','start':1.,'end':1.1}, {'word':'0','start':2.2,'end':2.3},
        {'word':'0','start':3.4,'end':3.5}]}
    assert len(regions(raw,4.))==3


def rich_answer():
    return {'signature':{'model':MODEL,'source_sha256':'test'},'text':'<|speaker:0|>[驚き]あ！本当？<|speaker:1|>[笑い]はい。',
        'words':[{'word':'あ','start':.2,'end':.4},
        {'word':'本当','start':.7,'end':1.}, {'word':'はい','start':1.1,'end':1.5}]}

def test_inline_speaker_tags_and_emotion_survive_word_grouping():
    rows=regions(rich_answer(),2.)
    assert len(rows)==2
    assert [r['speaker'] for r in rows]==['0','1']
    assert '[驚き]' in rows[0]['text'] and '[笑い]' in rows[1]['text']
    assert all('<|speaker:' not in r['text'] for r in rows)
    assert rows[0]['start']==.2 and rows[0]['end']==1.

def test_whole_fish_to_translation_and_shared_reference_without_mai(tmp_path):
    from render import prepare_segments
    from shared_reference import plan
    from reference_timing import refine_references
    clean=tmp_path/'clean.wav';sf.write(clean,np.zeros((SR*2,2),np.float32),SR)
    calls=[]
    def api(payload,key,**kwargs):
        content=payload['messages'][1]['content'];context=json.loads(content);calls.append(context)
        assert isinstance(content,str) and 'input_audio' not in json.dumps(payload)
        assert '[驚き]' in context['regions'][0]['text']
        assert all(set(r)=={'id','speaker','text'} for r in context['regions'])
        return kwargs['parser']('{"0":"[驚き]아! 정말?","1":"[笑い]네."}'),{}
    with patch('fish_pipeline.transcribe_words',return_value=rich_answer()) as asr, \
         patch('mai_pipeline.transcribe_source',side_effect=AssertionError('MAI must not run')), \
         patch('speech.http_json',side_effect=api):
        result=translate(clean,tmp_path/'translation',tmp_path/'result.json',{'key':'test','language':'ko'},lambda *a:None)
    assert asr.call_count==1 and len(calls)==1
    assert asr.call_args.args[0]==clean
    assert result['source_asr_model']==MODEL and result['boundary_authority']==AUTHORITY
    assert result['speaker_authority']==MODEL and result['preserve_source_speaker_ids']
    prepared=prepare_segments(result,SR*2)
    assert not any(s['natural_reference_margin'] for s in prepared)
    assert len(plan(prepared,result['speaker_timeline']))==2
    same,_=refine_references(clean,result,tmp_path/'reference')
    assert same is result

def test_explicit_legacy_fish_dispatch_uses_fish_only(tmp_path):
    from speech import transcribe
    with patch('fish_pipeline.translate',return_value={'ok':True}) as fish, \
         patch('native_timeline.translate',side_effect=AssertionError('legacy called')):
        assert transcribe('audio',tmp_path,tmp_path/'result',{**config(),'diarizer':'fish-transcribe-1-pro'},lambda *a:None)=={'ok':True}
    assert fish.call_count==1


def test_zero_duration_anchor_is_measured_without_fabricated_duration(tmp_path):
    from mai_pipeline import correct_edges
    raw={'text':'<|speaker:0|>あ！','words':[{'word':'あ','start':.5,'end':.5}]}
    rows=regions(raw,1.)
    assert rows[0]['start']==rows[0]['end']==.5
    values=np.zeros((SR,2),np.float32);a,b=round(.51*SR),round(.65*SR)
    values[a:b]=(.1*np.sin(np.arange(b-a)*.08))[:,None]
    path=tmp_path/'audio.wav';sf.write(path,values,SR,subtype='FLOAT')
    corrected,audit=correct_edges(path,rows)
    assert .49<=corrected[0]['start']<=.52 and .64<=corrected[0]['end']<=.67
    assert audit[0]['method']=='zero_duration_token_nearest_bounded_acoustic_run'

def test_zero_duration_token_does_not_enter_neighbor_voice():
    raw={'text':'<|speaker:0|>あ！<|speaker:1|>はい。','words':[
        {'word':'あ','start':.5,'end':.5}, {'word':'はい','start':.5,'end':.8}]}
    rows=regions(raw,1.)
    assert len(rows)==2
    assert [[w['word_id'] for w in r['words']] for r in rows]==[[0],[1]]


def test_default_individual_fish_pipeline_and_frozen_translation_contract():
    from fish_pipeline import fixed_translations
    cfg=config()
    assert cfg['diarizer']=='scribe-v2'
    assert cfg['tts_mode']=='same_speaker_batches' and cfg['tts_reference_mode']=='same_speaker_batch'
    assert cfg['speaker_embeddings'] is False
    rows=[{'id':0}]
    for response in ['{}','{"0":["대사","1"]}','{"0":"대사","0":"다른 대사"}']:
        with pytest.raises(ValueError):fixed_translations(response,rows)


def test_original_cues_cannot_be_translated_omitted_or_invented():
    from fish_pipeline import fixed_translations
    rows=[{'id':0,'text':'[足音]こんにちは。[驚き]本当？'}]
    value='[足音]안녕. [驚き]정말?'
    assert fixed_translations(json.dumps({'0':value}),rows)['values']==[value]
    for value in ['[footsteps]안녕. [surprised]정말?','안녕. 정말?','[足音]안녕. [驚き]정말? [laughing]']:
        with pytest.raises(ValueError,match='verbatim'):fixed_translations(json.dumps({'0':value}),rows)


def test_fish_language_hint_uses_iso_code_and_auto_detection_recognizes_new_languages(tmp_path):
    from fish_pipeline import source
    from mai_native_pipeline import detected_languages,record_auto_language
    with patch('fish_pipeline.transcribe_words',return_value=rich_answer()) as asr:
        source('clean',tmp_path,'test',lambda *a:None,language='fil')
        assert asr.call_args.kwargs['language']=='tl'
        source('clean',tmp_path,'test',lambda *a:None,language=None)
        assert asr.call_args.kwargs['language'] is None
    for raw,expected in [({'language':'Armenian'},'hy'),({'language_code':'as','language':'Assamese'},'as'),({'language':'tl'},'fil')]:
        assert detected_languages(raw)[0]==expected
        record_auto_language(raw,{'source_language_auto':True,'language':'ko'})
    with pytest.raises(RuntimeError,match='같습니다'):
        record_auto_language({'language_code':'hy'},{'source_language_auto':True,'language':'hy'})


@pytest.mark.parametrize('islands,expected',[(1,True),(2,False),(0,False)])
def test_point_reply_at_next_voice_is_relocated_only_to_unique_prior_island(tmp_path,islands,expected):
    from fish_pipeline import correct_fish_edges
    clean=tmp_path/'clean.wav';values=np.zeros((4*SR,2),np.float32)
    values[round(.2*SR):round(.7*SR)]=.2
    values[round(2.7*SR):round(3.5*SR)]=.2
    if islands>=1:values[round(1.6*SR):round(1.8*SR)]=.15
    if islands>=2:values[round(1.1*SR):round(1.25*SR)]=.15
    sf.write(clean,values,SR,subtype='FLOAT')
    rows=[{'id':0,'start':.2,'end':.7,'speaker':'0','text':'Previous.'},
          {'id':1,'start':2.6,'end':2.6,'speaker':'2','text':'Six.'},
          {'id':2,'start':2.7,'end':3.5,'speaker':'1','text':'Next.'}]
    fixed,audit=correct_fish_edges(clean,rows)
    recovered=fixed[1].get('point_recovery',{}).get('method')=='unique_isolated_reply_before_next_voice_anchor'
    assert recovered==expected
    if expected:
        assert fixed[1]['start']==pytest.approx(1.58,abs=.02)
        assert fixed[1]['end']==pytest.approx(1.84,abs=.02)
        assert fixed[1]['end']<rows[1]['start']
        assert rows[1]['start']==2.6


def test_late_punctuation_cannot_cover_a_following_speakers_first_sound():
    raw={'text':'ん？６。お前。','words':[
        {'word':'ん','start':44.64,'end':44.68,'speaker':'0'},
        {'word':'？','start':45.12,'end':45.12,'speaker':'0'},
        {'word':'６。','start':45.98,'end':45.98,'speaker':'2'},
        {'word':'お前','start':46.08,'end':46.68,'speaker':'1'},
        {'word':'。','start':47.,'end':47.,'speaker':'1'}]}
    rows=regions(raw,48.)
    assert rows[0]['text']=='ん？' and rows[0]['end']==44.68
    assert rows[2]['end']==46.68
    assert rows[0]['words'][-1]['start']==45.12
