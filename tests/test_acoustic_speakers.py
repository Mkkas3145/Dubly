import copy
import numpy as np
from acoustic_speakers import discover

def sample():
    rows=[];vectors=[]
    for voice,vector,original in [('a',[1.,0.,0.],'0'),('b',[0.,1.,0.],'0'),('c',[0.,0.,1.],'1')]:
        for j in range(5):
            i=len(rows);rows.append({'index':i,'speaker':original,'seconds':2.,'start':i*3.,'end':i*3.+2})
            vectors.append(vector)
    return rows,np.array(vectors)

def test_discovers_new_identity_and_preserves_text_independent_inputs():
    rows,z=sample();before=copy.deepcopy(rows);report=discover(rows,z)
    groups=[{d['bank_speaker'] for d in report['decisions'][a:a+5]} for a in (0,5,10)]
    assert all(len(g)==1 for g in groups)
    assert len(set.union(*groups))==3 and rows==before
    assert report==discover(rows,z)

def test_locked_speaker_and_isolated_short_noise_are_not_reassigned():
    rows,z=sample();rows[5]['speaker_locked']=True
    rows.append({'index':15,'speaker':'1','seconds':.3,'start':50.,'end':50.3})
    z=np.vstack([z,[-1.,-1.,-1.]])
    report=discover(rows,z)
    assert report['decisions'][5]['bank_speaker']=='0'
    assert report['decisions'][15]['bank_speaker']=='1'

def test_two_small_groups_do_not_invent_a_speaker():
    rows,z=sample();chosen=[0,1,5,6]
    assert not discover([rows[i] for i in chosen],z[chosen])['changes']
