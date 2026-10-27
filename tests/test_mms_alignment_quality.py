import copy,math
import pytest
import torch,torchaudio
from torchaudio.functional import TokenSpan
from mms_alignment_quality import describe,refine,improvement,row_quality


def fixture():
    bundle=torchaudio.pipelines.MMS_FA
    vocab=bundle.get_dict();star=vocab['*']
    emission=torch.full((60,len(vocab)),-9.)
    emission[:,0]=-.01;emission[:,star]=0
    return bundle,vocab,star,emission


def speak(emission,token,start,end):
    emission[start:end,0]=-8;emission[start:end,token]=-.001


def span(token,start,end,emission):
    return [TokenSpan(token,start,end,float(emission[start:end,token].exp().mean()))]


def test_real_ctc_moves_weak_word_to_better_audio_and_preserves_strong_neighbor():
    bundle,v,star,e=fixture();speak(e,v['a'],20,24);speak(e,v['b'],45,49)
    units=[{'word':'A','roman':'a'},{'word':'B','roman':'b'}]
    spans=[span(v['a'],4,5,e),span(v['b'],45,49,e)]
    words,review=refine(units,spans,bundle.get_tokenizer()(['a','b']),e,bundle.get_aligner(),star,1.2)
    assert review['accepted_regions']==1
    assert words[0]['start']==pytest.approx(.4)
    assert words[0]['end']==pytest.approx(.48)
    assert words[1]['start']==pytest.approx(.9)
    assert words[1]['end']==pytest.approx(.98)
    assert words[0]['alignment_quality']['score']>.9
    assert review['checks'][0]['reason']=='stronger_acoustic_evidence'


def test_two_reliable_neighbors_bound_recovery_on_original_clock():
    bundle,v,star,e=fixture()
    for c,a,b in [('a',5,9),('b',25,29),('c',45,49)]:speak(e,v[c],a,b)
    units=[{'word':c,'roman':c} for c in 'abc']
    spans=[span(v['a'],5,9,e),span(v['b'],10,11,e),span(v['c'],45,49,e)]
    words,review=refine(units,spans,bundle.get_tokenizer()(list('abc')),e,bundle.get_aligner(),star,1.2)
    assert review['accepted_regions']==1
    assert words[1]['start']==pytest.approx(.5)
    assert [words[i]['start'] for i in [0,2]]==pytest.approx([.1,.9])
    assert review['checks'][0]['window']==pytest.approx([.18,.9])


def test_uncertain_candidate_never_overwrites_existing_alignment_or_fails_video():
    bundle,v,star,e=fixture();speak(e,v['b'],45,49)
    units=[{'word':'A','roman':'a'},{'word':'B','roman':'b'}];spans=[span(v['a'],4,5,e),span(v['b'],45,49,e)]
    old=describe(units,spans,e,star,duration=1.2)
    words,review=refine(units,spans,bundle.get_tokenizer()(['a','b']),e,bundle.get_aligner(),star,1.2)
    assert review['accepted_regions']==0 and words==old
    def broken(*a):raise RuntimeError('candidate failed')
    words,review=refine(units,spans,[[v['a']],[v['b']]],e,broken,star,1.2)
    assert words==old and review['checks'][0]['reason']=='candidate_alignment_failed'


def test_supported_words_are_not_researched_and_low_token_is_visible_in_row_score():
    bundle,v,star,e=fixture();speak(e,v['a'],5,9)
    units=[{'word':'A','roman':'a'}];spans=[span(v['a'],5,9,e)]
    def forbidden(*a):raise AssertionError('No retry for reliable speech')
    words,review=refine(units,spans,[[v['a']]],e,forbidden,star,1.2)
    assert review['checked_regions']==0 and row_quality(words)['status']=='supported'
    words=describe(units,[span(v['a'],30,31,e)],e,star,duration=1.2)
    assert row_quality(words)['status']=='uncertain'
    assert row_quality(words)['suspect_words'][0]['word']=='A'


def test_higher_group_average_cannot_sacrifice_neighbor_word():
    bundle,v,star,e=fixture();speak(e,v['b'],45,49)
    units=[{'word':'A','roman':'a'},{'word':'B','roman':'b'}]
    old=describe(units,[span(v['a'],4,5,e),span(v['b'],45,49,e)],e,star,duration=1.2)
    new=copy.deepcopy(old)
    new[0]['alignment_quality']=copy.deepcopy(old[1]['alignment_quality'])
    new[1]['alignment_quality']=copy.deepcopy(old[0]['alignment_quality'])
    assert improvement(old,new)==(False,'neighbor_confidence_decreased')

def test_average_confidence_cannot_hide_one_unsupported_phoneme():
    bundle,v,star,e=fixture();speak(e,v['a'],5,9)
    words=describe([{'word':'AB','roman':'ab'}],[span(v['a'],5,9,e)+span(v['b'],10,11,e)],e,star,duration=1.2)
    q=words[0]['alignment_quality']
    assert 'weak_individual_phoneme' in q['reasons']
    assert q['weakest_phoneme']<.001


def test_short_retry_window_does_not_raise_or_force_a_new_location():
    bundle,v,star,e=fixture();e=e[:2]
    units=[{'word':'A','roman':'a'}];spans=[span(v['a'],0,1,e)]
    words,review=refine(units,spans,[[v['a']]],e,bundle.get_aligner(),star,.04)
    assert words[0]['start']==0 and words[0]['end']==.02
    assert review['checks'][0]['reason']=='insufficient_frames'


def test_fresh_acoustics_recovers_whole_utterance_despite_uncertain_neighbor():
    from mms_alignment_quality import refresh_uncertain
    bundle,v,star,e=fixture()
    speak(e,v['c'],50,54)
    units=[{'word':c,'roman':c} for c in 'abc']
    old=describe(units,[span(v['a'],4,5,e),span(v['b'],10,11,e),span(v['c'],50,54,e)],e,star,duration=1.2)
    fresh=e.clone();speak(fresh,v['b'],30,34)
    calls=[]
    def acoustics(left,right):calls.append((left,right));return fresh[left:right]
    words,review=refresh_uncertain(old,units,bundle.get_tokenizer()(list('abc')),[(0,1),(1,2),(2,3)],
        acoustics,bundle.get_aligner(),star,1.2)
    assert words[0]==old[0] and words[2]==old[2]
    assert .6<=words[1]['start']<=.62 and words[1]['end']==pytest.approx(.68)
    assert review['accepted_utterances']==1 and calls==[(0,50)]


def test_fresh_acoustic_failure_keeps_valid_global_alignment():
    from mms_alignment_quality import refresh_uncertain
    bundle,v,star,e=fixture();units=[{'word':'A','roman':'a'}]
    old=describe(units,[span(v['a'],4,5,e)],e,star,duration=1.2)
    def unavailable(*a):raise RuntimeError('local acoustic worker failed')
    words,review=refresh_uncertain(old,units,[[v['a']]],[(0,1)],unavailable,bundle.get_aligner(),star,1.2)
    assert words==old and review['accepted_utterances']==0
    assert review['checks'][0]['reason']=='candidate_alignment_failed'
