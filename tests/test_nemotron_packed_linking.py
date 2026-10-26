import numpy as np
from nemotron_packed_linking import verify,strict_target


class Native:
    batch_size=2
    def __init__(self):self.items=0
    def batch(self,waves):
        out=[]
        for wave in waves:
            self.items+=1;ids=np.rint(wave[::160]).astype(int)
            voices=sorted(set(ids)-{0});assert len(voices)<=8
            out.append(np.stack([ids==v for v in voices]+[np.zeros(len(ids),bool)]*(8-len(voices)),axis=1))
        return out


def fixture(voices):
    audio=np.repeat(np.asarray(voices,np.float32),48000);windows=[];groups={}
    for i,voice in enumerate(voices):
        windows.append({'start_frame':i*300,'end_frame':(i+1)*300,'active':np.ones((300,1),bool)})
        start=i*48000
        groups[i]={'component':i,'nodes':[[i,0]],'windows':{i},
            'examples':[{'start':start,'end':start+16000},{'start':start+16000,'end':start+32000}]}
    return audio,windows,groups


def test_native_packs_link_over_sixteen_global_voices_without_exceeding_eight_local_slots():
    audio,windows,groups=fixture([*range(1,17),*range(1,9)])
    infer=Native();edges,audit=verify(groups,windows,audio,infer)
    assert len(edges)==8
    for edge in edges:
        a,b=[n[0] for n in edge['nodes']];assert a==b-16
    assert any(len(r['candidates'])==7 for r in audit)
    assert infer.items==len(audit)
    # Every candidate pack takes two predictions instead of one pair of
    # complete-context predictions for each individual candidate.
    assert sum(len(r['candidates']) for r in audit)>infer.items*3


def test_two_reference_excerpts_must_both_match():
    audio,windows,groups=fixture([1,1])
    audio[16000:32000]=2
    edges,_=verify(groups,windows,audio,Native())
    assert not edges


def test_native_different_voice_constraint_blocks_a_merge_even_if_audio_sounds_same():
    audio,windows,groups=fixture([1,1])
    groups[1]['windows']={0}
    edges,_=verify(groups,windows,audio,Native())
    assert not edges


def test_long_majority_does_not_hide_short_other_speaker_in_same_native_track():
    window={'active':np.ones((300,1),bool)}
    prediction=np.zeros((300,2),bool);prediction[:270,0]=1;prediction[270:,1]=1
    # The old 85% majority rule accepts the 90% dominant voice. A real second
    # voice at the end must instead block using this whole track as one person.
    identity,mixed,_=strict_target(prediction,window,0,0,{0:0})
    assert identity is None and mixed


def test_consistent_single_voice_keeps_identity_with_variable_pause_lengths():
    active=np.ones((300,1),bool);active[90:120]=0;active[200:210]=0
    prediction=np.zeros((300,2),bool);prediction[:,0]=active[:,0]
    identity,mixed,_=strict_target(prediction,{'active':active},0,0,{0:0})
    assert identity==0 and not mixed
