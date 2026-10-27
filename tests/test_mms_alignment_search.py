import copy
import pytest
from mms_alignment_search import select_paths,metrics,windows


def choice(a,b,score,name):return {'start':a,'end':b,'utility':score,'name':name}


def test_sequence_can_relocate_two_uncertain_neighbors_together():
    # Independently best choices overlap. The best consistent sequence moves
    # both neighbors instead of treating either old boundary as ground truth.
    options=[[choice(1,2,.2,'old A'),choice(4,5,.8,'new A')],
             [choice(2.1,3,.2,'old B'),choice(5.1,6,.7,'new B')],
             [choice(8,9,1.,'anchor')]]
    result=select_paths(options)
    assert [v['name'] for v in result]==['new A','new B','anchor']


def test_candidate_cannot_cross_locked_anchor_or_reverse_transcript_order():
    options=[[choice(1,2,.2,'baseline'),choice(4,6,.9,'wrong')],
             [choice(5,5.5,1.,'anchor')]]
    assert select_paths(options)[0]['name']=='baseline'
    assert select_paths([[choice(4,5,1,'A')],[choice(2,3,1,'B')]]) is None


def test_utterance_score_does_not_require_every_shouted_phoneme_to_be_clear():
    def word(p,d,s):return {'alignment_quality':{'posterior':p,'log_posterior':__import__('math').log(p),
        'phoneme_dominance':d,'speech_support':s}}
    q=metrics([word(.8,1,1),word(.03,.4,.3)],[{'roman':'abcdef'},{'roman':'g'}])
    assert q['posterior']>.4 and q['score']>.5


def test_context_windows_search_beyond_bad_baseline_without_unbounded_model_input():
    result=windows(0,24,12,13)
    assert (0,1200) in result
    assert any(a*.02>=16 for a,b in result)
    assert all(0<b-a<=1500 for a,b in result)


def test_real_mms_ctc_search_preserves_anchor_and_moves_only_supported_utterance():
    import torch,torchaudio
    from torchaudio.functional import TokenSpan
    from mms_alignment_quality import describe
    from mms_alignment_search import search
    bundle=torchaudio.pipelines.MMS_FA;v=bundle.get_dict();star=v['*']
    old=torch.full((100,len(v)),-9.);old[:,0]=-.01;old[:,star]=0
    old[80:84,0]=-8;old[80:84,v['c']]=-.001
    units=[{'word':'B','roman':'b'},{'word':'C','roman':'c'}]
    spans=[[TokenSpan(v['b'],10,11,.001)],[TokenSpan(v['c'],80,84,.999)]]
    baseline=describe(units,spans,old,star,duration=2.)
    fresh=old.clone();fresh[50:54,0]=-8;fresh[50:54,v['b']]=-.001
    def provider(left,right,view):return fresh[left:right]
    words,audit=search(baseline,units,bundle.get_tokenizer()(['b','c']),[(0,1),(1,2)],
        provider,bundle.get_aligner(),star,2.,views=('clean','original'))
    assert audit['accepted_utterances']==1
    assert words[0]['start']==pytest.approx(1.)
    assert words[1]==baseline[1]


def test_search_errors_preserve_existing_timestamps_and_text():
    import torch,torchaudio
    from torchaudio.functional import TokenSpan
    from mms_alignment_quality import describe
    from mms_alignment_search import search
    bundle=torchaudio.pipelines.MMS_FA;v=bundle.get_dict();star=v['*']
    e=torch.full((100,len(v)),-9.);e[:,0]=-.01;e[:,star]=0
    units=[{'word':'B','roman':'b'}]
    baseline=describe(units,[[TokenSpan(v['b'],10,11,.001)]],e,star,duration=2.)
    def unavailable(*args):raise RuntimeError('acoustic view unavailable')
    words,audit=search(baseline,units,bundle.get_tokenizer()(['b']),[(0,1)],unavailable,bundle.get_aligner(),star,2.)
    assert words==baseline and audit['accepted_utterances']==0


def test_exact_sequence_keeps_valid_low_scoring_prefix_among_many_wrong_ones():
    # A global top-24 beam previously discarded the only prefix capable of
    # reaching the later fixed anchor.
    wrong=[choice(3+i*.01,5+i*.01,1.,str(i)) for i in range(40)]
    options=[[choice(1,2,.1,'valid'),*wrong],[choice(2.1,2.5,1.,'anchor')]]
    assert [c['name'] for c in select_paths(options)]==['valid','anchor']


def test_ambiguity_scores_include_paths_that_converge_on_same_final_dialogue():
    from mms_alignment_search import sequence_scores
    options=[[choice(1,2,.5,'A'),choice(3,4,.49,'B')],[choice(5,6,1.,'same ending')]]
    path,totals,best=sequence_scores(options)
    assert path[0]['name']=='A'
    assert best==pytest.approx(1.5)
    assert totals[0][1]==pytest.approx(1.49)


def test_context_width_adapts_to_short_text_without_changing_audio_clock():
    short=windows(0,20,10,11,phonemes=4)
    assert (0,100) in short and (50,150) in short
    assert all(0<=a<b<=1000 for a,b in short)


def test_repeatable_low_absolute_posterior_is_not_discarded_before_context_validation():
    import torch,torchaudio
    from torchaudio.functional import TokenSpan
    from mms_alignment_quality import describe
    from mms_alignment_search import search
    bundle=torchaudio.pipelines.MMS_FA;v=bundle.get_dict();star=v['*']
    old=torch.full((100,len(v)),-9.);old[:,0]=-.01;old[:,star]=0
    old[80:84,0]=-8;old[80:84,v['c']]=-.001
    units=[{'word':'B','roman':'b'},{'word':'C','roman':'c'}]
    baseline=describe(units,[[TokenSpan(v['b'],10,11,.001)],
        [TokenSpan(v['c'],80,84,.999)]],old,star,duration=2.)
    fresh=old.clone();fresh[50:54,0]=-8;fresh[50:54,v['b']]=-1.9
    def provider(left,right,view):return fresh[left:right]
    words,audit=search(baseline,units,bundle.get_tokenizer()(['b','c']),[(0,1),(1,2)],
        provider,bundle.get_aligner(),star,2.,views=('clean',))
    assert .1<words[0]['alignment_quality']['posterior']<.2
    assert words[0]['start']==pytest.approx(1.)
    assert audit['accepted_utterances']==1 and audit['accepted'][0]['consensus']>=2
    assert words[1]==baseline[1]


def test_merging_dialogue_rows_cannot_hide_a_misaligned_prefix_behind_internal_anchor():
    import torch,torchaudio
    from torchaudio.functional import TokenSpan
    from mms_alignment_quality import describe
    from mms_alignment_search import search,anchor_ranges
    bundle=torchaudio.pipelines.MMS_FA;v=bundle.get_dict();star=v['*']
    e=torch.full((100,len(v)),-9.);e[:,0]=-.01;e[:,star]=0
    e[80:84,0]=-8;e[80:84,v['c']]=-.001
    units=[{'word':'B','roman':'b'},{'word':'C','roman':'c'}]
    old=describe(units,[[TokenSpan(v['b'],10,11,.001)],
        [TokenSpan(v['c'],80,84,.999)]],e,star,duration=2.)
    fresh=e.clone();fresh[50:54,0]=-8;fresh[50:54,v['b']]=-.001
    provider=lambda left,right,view:fresh[left:right]
    args=(old,units,bundle.get_tokenizer()(['b','c']))
    split,a=search(*args,[(0,1),(1,2)],provider,bundle.get_aligner(),star,2.)
    merged,b=search(*args,[(0,2)],provider,bundle.get_aligner(),star,2.)
    assert merged==split and merged[0]['start']==pytest.approx(1.)
    assert merged[1]==old[1]  # a good internal word is immutable
    assert b['original_utterances']==1 and b['search_groups']==2
    assert b['accepted'][0]['row']==0 and b['accepted'][0]['word_range']==[0,1]
    assert anchor_ranges(old,units,[(0,2)])[1]==[0,0]
