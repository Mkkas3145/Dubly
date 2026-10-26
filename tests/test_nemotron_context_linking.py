import numpy as np
from unittest.mock import patch
from nemotron_context_linking import assess,verify

def scene():
    active=np.zeros((200,2),bool);active[:100,0]=1;active[100:,1]=1
    return {'start_frame':0,'end_frame':200,'active':active}

def test_full_context_matching_preserves_every_local_voice():
    w=scene();prediction=np.zeros((450,8),bool)
    prediction[:100,3]=1;prediction[125:225,3]=1
    prediction[250:350,3]=1;prediction[350:,4]=1
    result,detail=assess(prediction,w,250,0,[(0,1),(1.25,2.25)])
    assert result is True and detail['labels']=={0:3,1:4}
    prediction[350:,4]=0;prediction[350:,3]=1
    result,detail=assess(prediction,w,250,0,[(0,1),(1.25,2.25)])
    assert result is None and detail['reason']=='native_different_voices_collapsed'
    result,detail=assess(prediction,{**w,'trusted_constraints':False},250,0,[(0,1),(1.25,2.25)])
    assert result is True  # A single advisory view must not permanently split a known voice.

def test_partial_context_disagreement_and_reference_identity_change_are_rejected():
    w=scene();prediction=np.zeros((450,8),bool)
    prediction[:100,2]=1;prediction[125:225,3]=1
    prediction[250:350,3]=1;prediction[350:,4]=1
    result,detail=assess(prediction,w,250,0,[(0,1),(1.25,2.25)])
    assert result is False
    prediction[250:300,3]=0;prediction[250:300,5]=1
    assert assess(prediction,w,250,0,[(0,1),(1.25,2.25)])[0] is None

def test_returning_voice_requires_both_context_orders_and_two_reference_utterances():
    audio=np.zeros(12*16000,np.float32)
    # The two windows contain known different native voices.
    def native(lo,first,second):
        active=np.zeros((400,2),bool);active[:200,0]=1;active[200:,1]=1
        audio[lo*16000:(lo+2)*16000]=first;audio[(lo+2)*16000:(lo+4)*16000]=second
        return {'start_frame':lo*100,'end_frame':(lo+4)*100,'active':active}
    windows=[native(0,1,2),native(6,1,3)]
    groups={i:{'component':i,'nodes':[[wi,slot]],'windows':{wi},
        'examples':[{'start':(lo+j)*16000,'end':(lo+j+1)*16000} for j in range(2)]}
        for i,(wi,slot,lo) in enumerate([(0,0,0),(0,1,2),(1,0,6),(1,1,8)])}
    calls=[]
    def infer(wave):
        calls.append(wave.copy());values=np.rint(wave[::160]).astype(int);active=np.zeros((len(values),8),bool)
        for label in range(1,4):active[values==label,label]=1
        return active
    from nemotron_context_linking import context_labels
    from nemotron_local_consensus import labels as unit_labels
    with patch('nemotron_context_linking.context_labels',wraps=context_labels) as frame_checks, \
         patch('nemotron_local_consensus.labels',wraps=unit_labels) as unit_checks:
        events=[]
        edges,audit=verify(groups,windows,audio,infer,lambda stage,**data:events.append((stage,data)))
        # Shared scene/mask results are evaluated once for all target voices.
        keys=[(id(c.args[0]),id(c.args[1]),c.args[2]) for c in frame_checks.call_args_list]
        assert len(keys)==len(set(keys))
        assert unit_checks.call_count==frame_checks.call_count
        paired=[r for r in audit if r.get('method')=='two_complete_scenes']
        assert frame_checks.call_count<2*len(paired)
        assert any(stage=='verification_detail' and data['candidate']<=data['candidates'] for stage,data in events)
    assert len(edges)==1 and edges[0]['nodes']==[[0,0],[1,0]]
    matches=[r for r in audit if r['target']==2 and r['candidates']==[0]]
    assert [r['trial'] for r in matches]==[0,1] and all(r['result'] for r in matches)
    assert all(r['reason']!='native_different_voices_collapsed' for r in matches)
    count=len(calls)
    repeated_edges,repeated_audit=verify(groups,windows,audio,infer)
    assert repeated_edges==edges
    assert [{k:v for k,v in row.items() if k!='reused_exact_comparison'} for row in repeated_audit]==audit
    assert any(row.get('reused_exact_comparison') for row in repeated_audit)
    assert len(calls)==count  # Exact audio reuse must not re-run inference.

def test_native_capacity_is_never_exceeded_by_adding_a_reference():
    active=np.zeros((800,8),bool)
    for i in range(8):active[i*100:(i+1)*100,i]=1
    windows=[{'start_frame':0,'end_frame':800,'active':active},
        {'start_frame':800,'end_frame':1600,'active':active}]
    groups={i:{'component':i,'nodes':[[i,0]],'windows':{i},'examples':[
        {'start':i*128000,'end':i*128000+16000},{'start':i*128000+16000,'end':i*128000+32000}]} for i in range(2)}
    def forbidden(wave):raise AssertionError('must not submit nine different voices')
    assert not verify(groups,windows,np.zeros(256000,np.float32),forbidden)[0]

def test_only_confirmed_negative_controls_can_block_target_identity():
    from nemotron_context_linking import target_label
    observed={0:3,1:3,2:5}
    assert target_label(observed,{'cannot_link_slots':{(0,2)}},0)==3
    assert target_label(observed,{'cannot_link_slots':{(0,1)}},0) is None
    # An unstable unrelated voice must not invalidate a clean target.
    native=np.zeros((200,2),bool);native[:100,0]=1;native[100:,1]=1
    prediction=np.zeros((200,8),bool);prediction[:100,3]=1;prediction[100:150,4]=1;prediction[150:,5]=1
    from nemotron_context_linking import context_labels
    labels,_=context_labels(prediction,{'active':native},0)
    assert labels=={0:3} and target_label(labels,{},0)==3


def test_exact_comparison_cache_preserves_edges_and_invalidates_changed_controls():
    audio=np.zeros(12*16000,np.float32);windows=[]
    for start in (0,6):
        active=np.zeros((400,2),bool);active[:200,0]=1;active[200:,1]=1
        audio[start*16000:(start+2)*16000]=1;audio[(start+2)*16000:(start+4)*16000]=2
        windows.append({'start_frame':start*100,'end_frame':(start+4)*100,'active':active})
    groups={i:{'component':i,'nodes':[[wi,slot]],'windows':{wi},'examples':[
        {'start':(wi*6+slot*2+j)*16000,'end':(wi*6+slot*2+j+1)*16000} for j in range(2)]}
        for i,(wi,slot) in enumerate([(0,0),(0,1),(1,0),(1,1)])}
    def create_infer():
        def infer(wave):
            values=np.rint(wave[::160]).astype(int);active=np.zeros((len(values),8),bool)
            for label in (1,2):active[values==label,label]=1
            return active
        return infer
    baseline,_=verify(groups,windows,audio,create_infer(),reuse_scene_results=False)
    infer=create_infer()
    optimized,audit=verify(groups,windows,audio,infer)
    assert optimized==baseline and len(optimized)==2
    repeated,reused=verify(groups,windows,audio,infer)
    assert repeated==baseline and any(r.get('reused_exact_comparison') for r in reused)
    windows[0]['cannot_link_slots']={(0,1)}
    changed,checks=verify(groups,windows,audio,infer)
    assert changed==baseline
    assert not any(r.get('reused_exact_comparison') for r in checks)
