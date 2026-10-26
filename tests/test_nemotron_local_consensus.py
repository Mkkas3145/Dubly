import numpy as np
from nemotron_local_consensus import repair,units

def fixture():
    active=np.zeros((500,1),bool);active[:100]=1;active[200:300]=1;active[400:]=1
    window={'start_frame':0,'end_frame':500,'active':active,'examples':{0:[{'start':0,'end':16000},{'start':32000,'end':48000}]}}
    audit=[{'method':'two_complete_scenes','context_window':0,'reference_window':other,'trial':trial,
        'target_unit_labels':labels,'reference_unit_labels':[]} for other in [1,2] for trial,labels in [(0,[3,4,3]),(1,[7,2,7])]]
    return [window],audit

def test_repeated_two_order_contexts_split_a_fused_voice_without_moving_boundaries():
    windows,audit=fixture();before=windows[0]['active'].any(1).copy()
    changed,report=repair(windows,audit)
    assert changed and report[0]['after']==2
    active=windows[0]['active']
    np.testing.assert_array_equal(active.any(1),before)
    assert active[50,0] and active[450,0] and active[250,1]
    assert windows[0]['examples'][0]==[{'start':0,'end':16000}]
    assert windows[0]['examples'][1]==[{'start':32000,'end':48000}]

def test_one_comparison_or_order_dependent_identity_is_not_enough():
    windows,audit=fixture()
    assert not repair(windows,audit[:2])[0]
    windows,audit=fixture();audit[1]['target_unit_labels']=[7,7,7]
    assert not repair(windows,audit)[0]

def test_point_eight_gap_never_bridges_another_voice():
    active=np.zeros((200,2),bool);active[:50,0]=1;active[80:130,0]=1;active[55:75,1]=1
    value={'active':active}
    assert len([u for u in units(value) if u['slot']==0])==2

def test_speech_anchored_views_repair_a_fused_track_from_source_data():
    from nemotron_local_consensus import short_views
    wave=np.zeros(20*16000,np.float32);active=np.zeros((2000,1),bool)
    for a,b,voice in [(1,2,1),(4,5,2),(7,8,1),(10,11,2)]:
        wave[a*16000:b*16000]=voice;active[a*100:b*100]=1
    windows=[{'start_frame':0,'end_frame':2000,'active':active,'examples':{}}]
    def infer(values):
        levels=np.rint(values[::160]).astype(int);prediction=np.zeros((len(levels),8),bool)
        for voice in [1,2]:prediction[levels==voice,voice]=1
        return prediction
    views,audit=short_views(windows,wave,infer)
    assert any(v['end_frame']-v['start_frame']!=2000 for v in views)
    assert repair(windows,audit)[0]
    assert windows[0]['active'][150,0] and windows[0]['active'][750,0]
    assert windows[0]['active'][450,1] and windows[0]['active'][1050,1]

def test_split_and_merge_can_correct_identity_without_changing_track_count():
    active=np.zeros((500,2),bool);active[:100,0]=1;active[200:300,0]=1;active[400:,1]=1
    windows=[{'start_frame':0,'end_frame':500,'active':active,'examples':{}}]
    audit=[{'method':'short_native_view','context_window':0,'view_start_frame':i,'view_end_frame':i+1000,
        'target_unit_labels':[3,4,4]} for i in [0,100,200]]
    changed,report=repair(windows,audit)
    assert changed and report[0]['before']==report[0]['after']==2
    result=windows[0]['active']
    assert result[50,0] and result[250,1] and result[450,1]
    np.testing.assert_array_equal(result.any(1),active.any(1))

def test_simultaneous_voices_never_merge_or_trigger_a_noop_reverification():
    active=np.ones((100,2),bool)
    windows=[{'start_frame':0,'end_frame':100,'active':active,'examples':{}}]
    audit=[{'method':'short_native_view','context_window':0,'view_start_frame':i,'view_end_frame':i+1000,
        'target_unit_labels':[3,3]} for i in [0,100,200]]
    assert not repair(windows,audit)[0]
    assert not windows[0].get('identity_repaired')
