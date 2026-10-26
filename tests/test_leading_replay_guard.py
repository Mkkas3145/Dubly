from unittest.mock import patch
import numpy as np
import soundfile as sf
from leading_replay_guard import propose,inspect,POLICY
from core import SR


def query():
    return {'starts':np.array([.1,.2,.3,.4,.8,.9,1.,1.1]),
        'ends':np.array([.16,.26,.36,.46,.86,.96,1.06,1.16])}


GROUPS=[{'start':.08,'end':.5},{'start':.75,'end':1.2}]


def candidate(a,b,score):return {'a':a,'b':b,'score':score,'start':.1,'end':.46}


def test_complete_prefix_match_with_distinct_remaining_speech_proposes_quiet_gap():
    with patch('leading_replay_guard.candidates',return_value=[candidate(0,4,.8),candidate(4,8,.4)]):
        r=propose(query(),{},None,GROUPS)
    assert r['action']=='crop_candidate' and r['gap_start']==.5 and r['gap_end']==.75


def test_matching_both_turns_never_deletes_one_as_replay():
    with patch('leading_replay_guard.candidates',return_value=[candidate(0,4,.8),candidate(4,8,.76)]):
        assert propose(query(),{},None,GROUPS)['action']=='keep'


def test_partial_reference_cannot_delete_meaningful_leading_words():
    with patch('leading_replay_guard.candidates',return_value=[candidate(2,4,.95)]):
        assert propose(query(),{},None,GROUPS)['action']=='keep'


def test_no_gap_or_no_distinct_voice_does_not_propose_cut():
    assert propose(query(),{},None,[GROUPS[0]])['action']=='keep'
    assert propose(query(),{},None,[GROUPS[0],{'start':.4,'end':1.2}])['action']=='keep'
    q=query();q['starts']=q['starts'][:5];q['ends']=q['ends'][:5]
    assert propose(q,{},None,GROUPS)['action']=='keep'


def test_cognate_output_is_protected_without_loading_model(tmp_path):
    refs=[{'own':True,'original_text':'スパイ','translated_text':'스파이','path':'missing'}]
    with patch('phonetic_output_guard._get_runtime',side_effect=AssertionError('protected')):
        r=inspect('missing',refs,tmp_path/'out.wav',{'action':'keep','utterances':GROUPS})
    assert r['action']=='keep' and r['leading_replay_check']['action']=='skip'


def test_cut_preserves_complete_remaining_wave_and_does_not_edit_raw(tmp_path):
    from types import SimpleNamespace
    source=tmp_path/'source.wav';values=np.random.default_rng(4).normal(0,.1,(2*SR,2)).astype('float32')
    sf.write(source,values,SR,subtype='FLOAT');before=source.read_bytes()
    refs=[{'own':True,'original_text':'六つ','translated_text':'여섯','path':'reference'}]
    fake=SimpleNamespace(encode=lambda *args:{})
    with patch('phonetic_output_guard._get_runtime',return_value=(fake,None)),patch('leading_replay_guard.propose',
        return_value={'action':'crop_candidate','gap_start':.5,'gap_end':.75}):
        r=inspect(source,refs,tmp_path/'out.wav',{'action':'keep','utterances':GROUPS})
    assert r['action']=='cropped' and not r['content_verified']
    actual,_=sf.read(tmp_path/'out.wav',always_2d=True)
    assert np.max(np.abs(actual-values[round(.625*SR):]))<2**-22
    assert source.read_bytes()==before


def test_guard_failure_keeps_audio_instead_of_aborting(tmp_path):
    refs=[{'own':True,'original_text':'六つ','translated_text':'여섯','path':'missing'}]
    with patch('phonetic_output_guard._get_runtime',side_effect=RuntimeError('unavailable')):
        r=inspect('missing',refs,tmp_path/'out.wav',{'action':'keep','utterances':GROUPS})
    assert r['action']=='keep' and r['leading_replay_check']['action']=='unavailable'
