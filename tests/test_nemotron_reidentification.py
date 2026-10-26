import numpy as np
from nemotron_reidentification import verify,distinct_examples,dominant
from nemotron_linking import link

def test_same_model_reconnects_returning_voice_without_external_embeddings():
    audio=np.zeros(16000*12,np.float32)
    def group(identity,window,spans,code):
        for a,b in spans:audio[a*16000:b*16000]=code
        return {'component':identity,'nodes':[[window,identity]],'windows':{window},
            'examples':[{'start':a*16000,'end':b*16000} for a,b in spans]}
    groups={0:group(0,0,[(0,1),(1,2)],1),1:group(1,0,[(2,3),(3,4)],2),
        2:group(2,1,[(5,6),(6,7)],1),3:group(3,1,[(8,9),(9,10)],3)}
    calls=[]
    def infer(wave):
        calls.append(wave)
        values=np.rint(wave[::160]).astype(int);out=np.zeros((len(values),8),bool)
        for i in range(1,4):out[values==i,i]=True
        return out
    edges,audit=verify(groups,audio,infer)
    assert len(edges)==1 and edges[0]['nodes']==[[0,0],[1,2]]
    assert calls and all(sum(np.any(infer_value,axis=0))<=8 for infer_value in [infer(calls[0])])
    assert all(len(a['anchors'])==1 for a in audit)

def test_one_short_example_is_not_identity_evidence():
    group={'component':0,'nodes':[[0,0]],'windows':{0},'examples':[{'start':0,'end':16000}]}
    other={'component':1,'nodes':[[1,0]],'windows':{1},'examples':[{'start':16000,'end':32000}]}
    def forbidden(wave):raise AssertionError('one example must not authorize a merge')
    edges,audit=verify({0:group,1:other},np.zeros(32000,np.float32),forbidden)
    assert not edges and not audit

def test_physical_speech_cannot_count_twice_and_mixed_voice_abstains():
    assert len(distinct_examples([{'start':0,'end':32000},{'start':100,'end':32100}]))==1
    active=np.zeros((100,8),bool);active[:50,0]=1;active[50:,1]=1
    assert dominant(active,0,1) is None

def test_verified_edges_respect_native_cannot_link_and_disable_embeddings():
    a=np.zeros((200,2),bool);a[:100,0]=1;a[100:,1]=1
    windows=[{'start_frame':i*200,'end_frame':(i+1)*200,'active':a} for i in range(2)]
    edges=[{'nodes':[[0,0],[1,0]]},{'nodes':[[0,1],[1,0]]}]
    turns,audit=link(windows,400,verified_edges=edges,use_embeddings=False)
    assert audit['global_speakers']==3
    assert audit['same_model_verifications'][0]['accepted']
    assert not audit['same_model_verifications'][1]['accepted']
    assert not audit['embedding_merges'] and audit['calibration']['status']=='disabled_no_external_embedding_model'

def test_real_contiguous_examples_exclude_other_speaker_islands():
    from nemotron_samples import contiguous_samples
    from nemotron_policy import WINDOW_POLICY
    active=np.zeros((400,2),bool);active[:150,0]=1;active[150:200,1]=1;active[200:,0]=1
    probabilities=np.where(active,.99,.01);wave=np.zeros(400*160,np.float32)
    samples=contiguous_samples(wave,probabilities,active,16000,160,WINDOW_POLICY)
    assert samples[0]
    assert all(b<=150*160 or a>=200*160 for a,b in samples[0])
