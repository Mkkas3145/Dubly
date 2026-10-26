from unittest.mock import patch
import json
import numpy as np
import soundfile as sf
from tts_output_guard import identify,inspect_output,check_batch,MODEL

def w(text,a,b):return {'word':text,'start':a,'end':b}
def test_cognates_are_protected_without_japanese_script_rejection():
    assert identify([w('ス',0,.2),w('パ',.2,.4),w('イ',.4,.6)],{'original_text':'スパイ、','translated_text':'[嬉しい]스파이,'})==[(0,2)]
    assert identify([w('ミッション',0,.6)],{'original_text':'ミッション、','translated_text':'미션,'})==[(0,0)]
    assert identify([w('パパ',0,.6)],{'original_text':'父！','translated_text':'파파!'})==[(0,0)]

def test_keep_translated_korean_rendered_in_katakana(tmp_path):
    result=inspect_output(None,[w('ドクンドクン',0,1)],{'original_text':'わくわく！','translated_text':'두근두근!'},tmp_path/'out.wav')
    assert result['action']=='keep' and not result['trimmed']

def test_crop_verified_suffix_at_quiet_gap_preserves_target_samples(tmp_path):
    rate=44100;t=np.arange(rate*2)/rate;audio=np.sin(t*900).astype('float32')*.1
    audio[round(.7*rate):rate]=0
    sf.write(tmp_path/'in.wav',audio,rate,subtype='FLOAT')
    words=[w('スパイ',.1,.7),w('ミッション',1.,1.8)]
    segment={'original_text':'スパイ、','translated_text':'스파이,','_guard_reference_texts':[{'original_text':'ミッション、','translated_text':'미션,'}]}
    r=inspect_output(tmp_path/'in.wav',words,segment,tmp_path/'out.wav')
    assert r['action']=='cropped' and .7<=r['keep_end']<=1.
    saved,rate=sf.read(tmp_path/'out.wav');assert np.max(np.abs(saved-audio[:len(saved)]))<2**-22

def test_unexplained_extra_is_not_blindly_deleted(tmp_path):
    r=inspect_output(None,[w('스파이',0,.5),w('unknown',.8,1)],{'original_text':'スパイ','translated_text':'스파이'},tmp_path/'x.wav')
    assert r['action']=='keep_uncertain' and not (tmp_path/'x.wav').exists()

def test_repeated_target_is_ambiguous_and_not_cut(tmp_path):
    r=inspect_output(None,[w('미션',0,.5),w('미션',.8,1.2)],{'original_text':'ミッション','translated_text':'미션'},tmp_path/'x.wav')
    assert r['action']=='keep_uncertain'

def test_no_quiet_boundary_is_not_cut(tmp_path):
    sf.write(tmp_path/'x.wav',np.full(44100,.2),44100)
    r=inspect_output(tmp_path/'x.wav',[w('미션',0,.4),w('스파이',.4,.9)],{'original_text':'ミッション','translated_text':'미션',
        '_guard_reference_texts':[{'original_text':'スパイ','translated_text':'스파이'}]},tmp_path/'out.wav')
    assert r['action']=='reject'

def test_one_scribe_request_per_batch_and_restore_each_original_output_clock(tmp_path):
    prepared=[]
    for i in range(2):
        path=tmp_path/f'{i}.wav';sf.write(path,.1*np.sin(np.arange(44100)*.05),44100)
        prepared.append({'index':i,'generated':path,'folder':tmp_path,'generation':{},
            'segment':{'original_text':'はい','translated_text':'네'}})
    answer={'words':[w('네',.1,.5),w('네',2.1,2.5)]}
    with patch('tts_output_guard.transcribe_words',return_value=answer) as call:
        result=check_batch(prepared,{'key':'test'},tmp_path/'check')
    assert call.call_count==1 and call.call_args.kwargs['model']==MODEL=='elevenlabs/scribe-v2'
    assert call.call_args.kwargs['language'] is None
    assert all(p['generation']['tts_output_check']['action']=='keep' for p in result)

def test_scribe_provider_failure_marks_only_that_batch(tmp_path):
    path=tmp_path/'x.wav';sf.write(path,.1*np.sin(np.arange(44100)*.05),44100)
    p={'index':0,'generated':path,'folder':tmp_path,'generation':{},'segment':{'original_text':'はい','translated_text':'네'}}
    with patch('tts_output_guard.transcribe_words',side_effect=RuntimeError('HTTP 500')):
        result=check_batch([p],{'key':'test'},tmp_path/'check')
    assert '_guard_error' not in result[0]
    assert result[0]['generation']['tts_output_check']['reason']=='scribe_output_check_failed'

def test_remove_distinct_original_language_prefix_before_verified_korean(tmp_path):
    rate=44100;audio=np.sin(np.arange(rate*2)*.04).astype('float32')*.1
    audio[round(.5*rate):round(.8*rate)]=0;source=tmp_path/'in.wav';sf.write(source,audio,rate)
    result=inspect_output(source,[w('こんにちは',.1,.5),w('안녕하세요',.8,1.6)],
        {'original_text':'こんにちは','translated_text':'안녕하세요'},tmp_path/'out.wav')
    assert result['action']=='cropped' and .5<=result['keep_start']<=.8

def test_repeated_cognate_is_never_deleted_as_foreign_language(tmp_path):
    result=inspect_output(None,[w('スパイ',.1,.6),w('스파이',.8,1.3)],
        {'original_text':'スパイ','translated_text':'스파이'},tmp_path/'out.wav')
    assert result['action']=='keep_uncertain' and not (tmp_path/'out.wav').exists()

def test_missing_short_reply_uses_verified_neighbor_slots_without_inventing_words(tmp_path):
    prepared=[]
    for i,text in enumerate(['앞 대사','네','뒤 대사']):
        path=tmp_path/f'{i}.wav';values=np.zeros(44100,dtype='float32')
        values[5000:13000]=np.sin(np.arange(8000)*.05)*.1;sf.write(path,values,44100)
        prepared.append({'index':i,'generated':path,'reference':path,'folder':tmp_path,'generation':{},
            'segment':{'original_text':'原文','translated_text':text}})
    answer={'words':[w('앞',.1,.2),w('대사',.2,.4),w('뒤',4.1,4.2),w('대사',4.2,4.4)]}
    with patch('tts_output_guard.transcribe_words',return_value=answer):result=check_batch(prepared,{'key':'test'},tmp_path/'check')
    report=result[1]['generation']['tts_output_check']
    assert report['action']=='keep' and report['reason']=='inferred_short_vocalization_from_verified_neighbors'
    assert report['recognized_text']=='' and not report['neighbor_inference']['content_verified']
    assert '_guard_error' not in result[1]

def test_missing_reply_does_not_get_inferred_without_both_verified_neighbors(tmp_path):
    path=tmp_path/'voice.wav';sf.write(path,np.sin(np.arange(44100)*.05)*.1,44100)
    p={'index':0,'generated':path,'reference':path,'folder':tmp_path,'generation':{},'segment':{'original_text':'原文','translated_text':'네'}}
    with patch('tts_output_guard.transcribe_words',return_value={'words':[]}):result=check_batch([p],{'key':'test'},tmp_path/'check')
    assert result[0]['generation']['tts_output_check']['action']=='keep_uncertain'

def test_old_policy_scribe_cache_reused_by_audio_hash_without_new_paid_request(tmp_path):
    import hashlib
    from core import atomic_json
    from scribe import VERSION
    root=tmp_path/'checks';legacy=root/'previous-policy';legacy.mkdir(parents=True)
    source=tmp_path/'tts.wav';sf.write(source,.1*np.sin(np.arange(44100)*.05),44100,subtype='PCM_16')
    with sf.SoundFile(legacy/'generated.wav','w',samplerate=44100,channels=1,subtype='PCM_16') as out:
        out.write(.1*np.sin(np.arange(44100)*.05));out.write(np.zeros(44100,dtype='float32'))
    sha=hashlib.sha256((legacy/'generated.wav').read_bytes()).hexdigest()
    signature={'model':MODEL,'version':VERSION,'source_sha256':sha,'language':None,
        'response_format':'verbose_json','timestamp_granularities':['word']}
    raw={'text':'네','words':[w('네',.1,.5)]}
    atomic_json(legacy/'scribe/scribe.json',{'signature':signature,'duration':2.,'words':raw['words'],'text':'네'})
    atomic_json(legacy/'scribe/scribe-raw.json',raw)
    p={'index':0,'generated':source,'reference':source,'folder':tmp_path,'generation':{},'segment':{'original_text':'はい','translated_text':'네'}}
    with patch('scribe.urllib.request.urlopen',side_effect=AssertionError('new paid request')) as network:
        result=check_batch([p],{'key':'unused'},root)
    network.assert_not_called()
    assert result[0]['generation']['tts_output_check']['action']=='keep'

def test_multiple_separated_vocalizations_cannot_fill_missing_short_reply(tmp_path):
    from tts_output_guard import short_vocalization
    reference=tmp_path/'ref.wav';source=tmp_path/'tts.wav';values=np.zeros(44100,dtype='float32')
    values[2000:8000]=np.sin(np.arange(6000)*.05)*.1;sf.write(reference,values,44100)
    values[27000:33000]=np.sin(np.arange(6000)*.05)*.1;sf.write(source,values,44100)
    assert short_vocalization(source,reference,'네') is None

def test_long_donor_replay_is_rejected_without_lowering_crop_standards(tmp_path):
    segment={'original_text':'あっ','translated_text':'앗',
        '_guard_reference_texts':[{'original_text':'スパイ、ミッション、わくわく！','translated_text':'스파이, 미션, 두근!'}]}
    words=[w('스파이',0,1),w('미션',1,2),w('와쿠와쿠',2,3),w('아냐',3,4)]
    report=inspect_output(None,words,segment,tmp_path/'out.wav')
    assert report['action']=='reject' and report['reason']=='reference_replay_without_target'
    assert not (tmp_path/'out.wav').exists()

def test_short_asr_disagreement_preserves_audio_and_records_uncertainty(tmp_path):
    path=tmp_path/'tts.wav';sf.write(path,.1*np.sin(np.arange(44100)*.05),44100)
    p={'index':0,'generated':path,'folder':tmp_path,'generation':{},
        'segment':{'original_text':'はい','translated_text':'네'}}
    with patch('tts_output_guard.transcribe_words',return_value={'words':[w('예',.1,.5)]}):
        checked=check_batch([p],{'key':'test'},tmp_path/'checks')[0]
    assert checked['generation']['tts_output_check']['action']=='keep_uncertain'
    assert '_guard_error' not in checked and checked['generated']==path

def test_asr_end_drift_into_inspection_silence_does_not_veto_correct_content(tmp_path):
    path=tmp_path/'tts.wav';sf.write(path,.1*np.sin(np.arange(44100)*.05),44100)
    p={'index':0,'generated':path,'folder':tmp_path,'generation':{},
        'segment':{'original_text':'はい','translated_text':'네'}}
    with patch('tts_output_guard.transcribe_words',return_value={'words':[w('네',.1,1.2)]}):
        checked=check_batch([p],{'key':'test'},tmp_path/'checks')[0]
    report=checked['generation']['tts_output_check']
    assert report['action']=='keep' and report['asr_boundary_clipped']
    assert '_guard_error' not in checked

def test_silent_generated_audio_is_rejected_even_if_asr_hallucinates_target(tmp_path):
    path=tmp_path/'tts.wav';sf.write(path,np.zeros(44100),44100)
    p={'index':0,'generated':path,'folder':tmp_path,'generation':{},
        'segment':{'original_text':'はい','translated_text':'네'}}
    with patch('tts_output_guard.transcribe_words',return_value={'words':[w('네',.1,.5)]}):
        checked=check_batch([p],{'key':'test'},tmp_path/'checks')[0]
    assert checked['_guard_error']=='unusable_generated_audio'

def test_corrupt_output_is_isolated_without_poisoning_valid_batch_members(tmp_path):
    good=tmp_path/'good.wav';bad=tmp_path/'bad.mp3'
    sf.write(good,.1*np.sin(np.arange(44100)*.05),44100);bad.write_bytes(b'broken')
    prepared=[{'index':i,'generated':path,'folder':tmp_path,'generation':{},
        'segment':{'original_text':'はい','translated_text':'네'}} for i,path in enumerate([bad,good])]
    with patch('tts_output_guard.transcribe_words',return_value={'words':[w('네',.1,.5)]}) as asr:
        checked=check_batch(prepared,{'key':'unused'},tmp_path/'checks')
    assert asr.call_count==1
    assert checked[0]['_guard_error']=='unusable_generated_audio'
    assert checked[1]['generation']['tts_output_check']['action']=='keep'
