import numpy as np
from reference_bank_guard import decide
from shared_reference import plan


def fixture():
    # Three consistent long adult anchors, four child references, one mislabeled
    # adult: arbitrary order/IDs, no text or absolute timestamp evidence.
    rows=[{'index':i,'speaker':'adult','seconds':3.} for i in range(3)]
    rows += [{'index':i,'speaker':'child','seconds':.7} for i in range(3,7)]
    rows += [{'index':7,'speaker':'child','seconds':3.}]
    vectors=np.array([[1,.01,0],[1,-.01,0],[1,0,.01],
                      [0,1,.01],[0,1,-.01],[.01,1,0],[-.01,1,0],[1,.02,0]],dtype=float)
    vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    return rows,vectors


def test_only_consensus_outlier_changes_reference_bank():
    rows,vectors=fixture();decisions=decide(rows,vectors)
    changed=[d for d in decisions if d['action']!='keep']
    assert len(changed)==1 and changed[0]['index']==7 and changed[0]['bank_speaker']=='adult'
    assert rows[7]['speaker']=='child'
    # Reordering must not alter decisions.
    order=[7,4,1,6,0,5,3,2]
    again=decide([rows[i] for i in order],vectors[order])
    assert {(d['index'],d['bank_speaker']) for d in decisions}=={(d['index'],d['bank_speaker']) for d in again}


def test_short_outlier_or_only_two_witnesses_cannot_reassign():
    rows,vectors=fixture();rows[7]['seconds']=.8
    assert all(d['action']=='keep' for d in decide(rows,vectors))
    rows[7]['seconds']=3.;rows[2]['seconds']=.8
    assert all(d['action']=='keep' for d in decide(rows,vectors))


def test_unusual_acting_without_strong_other_identity_is_preserved():
    rows,vectors=fixture();vectors[7]=[0.,0.,1.]
    assert all(d['action']=='keep' for d in decide(rows,vectors))


def test_reference_plan_uses_verified_bank_without_changing_asr_id():
    segments=[{'start':i*2.,'end':i*2.+1.,'speaker_id':str(i%2),
               'original_text':'source','translated_text':'번역'} for i in range(4)]
    timeline={'turns':[{'start':s['start'],'end':s['end'],'speaker':s['speaker_id']} for s in segments]}
    segments[1]['_reference_bank_speaker']='0'
    groups=plan(segments,timeline)
    assert groups[0]['speaker']=='0' and groups[0]['indices']==[0,1,2]
    assert segments[1]['speaker_id']=='1'


def test_acoustic_donors_keep_target_first_and_never_follow_other_voice():
    from reference_bank_guard import select_donors
    rows=[{'index':i,'speaker':'A' if i<4 else 'B','seconds':2.} for i in range(7)]
    vectors=np.array([[1.,0.],[.8,.6],[.9,.4],[.7,.7],[1.,0.],[1.,0.],[1.,0.]])
    segments=[{'start':i*20.,'end':i*20.+2.,'speaker_id':r['speaker']} for i,r in enumerate(rows)]
    decisions=[{'index':r['index'],'bank_speaker':r['speaker']} for r in rows]
    select_donors(rows,vectors,decisions,segments)
    assert segments[0]['_reference_donor_indices']==[0,2,1,3]
    assert all(segments[i]['speaker_id']=='A' for i in segments[0]['_reference_donor_indices'])


def test_contaminated_original_bank_does_not_block_independent_consensus():
    rows,vectors=fixture()
    # A fragmented original partition must not immunize its mislabeled member.
    vectors[3:7]=[[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    changed={d['index']:d for d in decide(rows,vectors) if d['action']!='keep'}
    assert changed[7]['bank_speaker']=='adult'
    assert changed[7]['own_peer_cohesion']<.35


def test_existing_close_cohort_prevents_relabeling_an_emotional_variant():
    rows,vectors=fixture()
    vectors[3:6]=[[.62,.78,0],[.62,.78,0],[.62,.78,0]]
    # Stronger alternative voices alone do not invalidate a real own-bank cohort.
    assert next(d for d in decide(rows,vectors) if d['index']==7)['action']=='keep'


def test_donors_must_agree_with_target_and_each_other():
    from reference_bank_guard import select_donors
    rows=[{'index':i,'speaker':'mixed','seconds':2.} for i in range(5)]
    vectors=np.array([[1,0,0],[.75,.66,0],[.75,-.66,0],[.9,.1,0],[0,0,1]],float)
    segments=[{'start':i*20.,'end':i*20.+2.,'speaker_id':'mixed'} for i in range(5)]
    ds=[{'index':i,'bank_speaker':'mixed'} for i in range(5)]
    selected=select_donors(rows,vectors,ds,segments)[0]
    assert selected['reference_indices']==[0,3,1]
    assert {d['reason'] for d in selected['rejected']}=={'weak_target_identity','conflicting_donor_identity'}
