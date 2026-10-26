import copy
import numpy as np
from native_speaker_repair import decide,apply_decisions
from mai_pipeline import word_regions


def fixture():
    native=[{'source_index':0,'start':0.,'end':.5,'speaker':'A'},
            {'source_index':1,'start':.8,'end':2.,'speaker':'A'}]
    anchors=[{'source_index':i,'start':20.+i*5,'end':22.+i*5,'seconds':2.,'speaker':'A' if i<3 else 'B'} for i in range(6)]
    vectors=np.array([[1.,0.,0.],[0.,1.,0.]]+[[1.,0.,0.]]*3+[[0.,1.,0.]]*3)
    return native,anchors,vectors


def test_adjacent_mislabel_corrects_whole_turn_and_its_words_before_merging():
    native,anchors,vectors=fixture();decisions,breaks=decide(native,anchors,vectors)
    assert [d['speaker'] for d in decisions]==['A','B'] and breaks==[.8]
    raw={'segments':[{'start':0.,'end':.5,'speaker':'A','text':'はい！'},
                     {'start':.8,'end':2.,'speaker':'A','text':'本当？'}],
         'words':[{'start':0.,'end':.5,'speaker':'A','word':'はい'},
                  {'start':.8,'end':2.,'speaker':'A','word':'本当'}]}
    prior=copy.deepcopy(raw);repaired=apply_decisions(raw,native,decisions)
    assert raw==prior
    assert [w['speaker'] for w in repaired['words']]==['A','B']
    for key in ('segments','words'):
        assert [{k:v for k,v in item.items() if k!='speaker'} for item in repaired[key]]==[{k:v for k,v in item.items() if k!='speaker'} for item in raw[key]]
    assert [r['text'] for r in word_regions(repaired,3.,merge_breaks=breaks)]==['はい！','本当？']


def test_clear_same_voice_continuation_still_merges_without_splitting_words():
    native,anchors,vectors=fixture();vectors[1]=vectors[0]
    decisions,breaks=decide(native,anchors,vectors)
    assert not breaks and all(d['action']=='keep' for d in decisions)
    raw={'segments':[{'start':0.,'end':.5,'speaker':'A','text':'あ…'},
                     {'start':.8,'end':2.,'speaker':'A','text':'本当？'}]}
    assert word_regions(raw,3.)==[{'start':0.,'end':.5,'speaker':'A','text':'あ…'},{'start':.8,'end':2.,'speaker':'A','text':'本当？'}]
    raw['words']=[{'start':0.,'end':.5,'speaker':'A','word':'あ'},
                  {'start':.8,'end':2.,'speaker':'A','word':'本当'}]
    assert len(word_regions(raw,3.,merge_breaks=breaks))==1
    assert len(word_regions(raw,3.,merge_breaks=[.8]))==2


def test_no_correction_without_three_independent_long_witnesses_or_adjacency():
    native,anchors,vectors=fixture()
    anchors=anchors[:-1];vectors=vectors[:-1]
    decisions,_=decide(native,anchors,vectors)
    assert all(d['action']=='keep' for d in decisions)
    native,anchors,vectors=fixture();native[1]['start']=2.
    decisions,breaks=decide(native,anchors,vectors)
    assert not breaks and all(d['action']=='keep' for d in decisions)


def test_reciprocal_diarization_swap_needs_independent_evidence_for_both_sides():
    native,anchors,vectors=fixture();native[1]['speaker']='B'
    vectors[[0,1]]=vectors[[1,0]]
    decisions,breaks=decide(native,anchors,vectors)
    assert [d['speaker'] for d in decisions]==['B','A'] and breaks==[.8]
    assert all(d.get('reason')=='reciprocal_independent_voice_witnesses' for d in decisions)
    vectors[1]=[0.,0.,1.]
    decisions,breaks=decide(native,anchors,vectors)
    assert all(d['action']=='keep' for d in decisions)


def test_long_provider_segment_is_repaired_by_word_owned_phrases_not_whole_segment():
    from native_speaker_repair import source_units
    raw={'segments':[{'start':0.,'end':5.,'speaker':'A','text':'一。二！'}],
         'words':[{'start':0.,'end':1.,'speaker':'A','word':'一'},
                  {'start':4.,'end':5.,'speaker':'A','word':'二'}]}
    units=source_units(raw,6.)
    assert len(units)==2
    decisions=[{'source_index':0,'speaker':'B','action':'reassign_native_turn'}]
    result=apply_decisions(raw,units,decisions)
    assert result['segments'][0]==raw['segments'][0]
    assert [w['speaker'] for w in result['words']]==['B','A']
    assert [r['text'] for r in word_regions(result,6.)]==['一。','二！']


def test_repaired_id_cannot_newly_merge_with_acoustically_conflicting_predecessor():
    # A low-confidence predecessor stays B; the reciprocal swap turns its
    # neighboring A into B, but that must not erase their audible voice change.
    native=[{'source_index':i,'start':i*2.,'end':i*2.+1.5,'speaker':s} for i,s in enumerate(('B','A','B'))]
    anchors=[{'source_index':i,'start':30.+i*5,'end':32.+i*5,'seconds':2.,'speaker':'A' if i<3 else 'B'} for i in range(6)]
    vectors=np.array([[.3,0.,np.sqrt(.91)],[0.,1.,0.],[1.,0.,0.]]+[[1.,0.,0.]]*3+[[0.,1.,0.]]*3)
    decisions,breaks=decide(native,anchors,vectors)
    assert [d['speaker'] for d in decisions]==['B','B','A']
    assert native[1]['start'] in breaks and decisions[1]['post_reassignment_merge_blocked']
