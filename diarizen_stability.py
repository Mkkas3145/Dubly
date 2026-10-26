"""Temporal decoding of DiariZen votes, without text/time hints."""
import math
import numpy as np

VERSION=1
POLICY={'switch_prior':'median_single_speaker_speech_island_frames','confident_probability':.9,
        'silence_and_overlap':'immutable'}

def stabilize(votes,active,switch_odds=None,confident_probability=POLICY['confident_probability']):
    votes=np.asarray(votes,dtype=np.float64);active=np.asarray(active)
    if votes.shape!=active.shape or votes.ndim!=2:raise ValueError('Expected matching frame × speaker arrays')
    result=active.copy();single=(active>0).sum(1)==1
    # Silence and real overlapping speech are immutable separators.
    changes=np.diff(np.r_[False,single,False].astype(np.int8));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
    # Empirical geometric dwell prior: estimate the typical uninterrupted
    # speech-island length from the entire recording, not a target dialogue.
    if switch_odds is None:switch_odds=max(2.,float(np.median(ends-starts))-1.) if len(starts) else 2.
    if switch_odds<=1:raise ValueError('Switch odds must exceed 1')
    blocks=zip(starts,ends)
    penalty=math.log(switch_odds);reports=[]
    for lo,hi in blocks:
        allowed=np.flatnonzero(np.any(active[lo:hi]>0,axis=0))
        if len(allowed)<2:continue
        v=np.maximum(0,np.nan_to_num(votes[lo:hi,allowed]));den=v.sum(1,keepdims=True)
        if np.any(den==0):continue
        p=v/den;emission=np.log(np.maximum(p,1e-12))
        # Never smooth away a short, strongly supported identity change.
        confident=p.max(1)>=confident_probability
        for t in np.flatnonzero(confident):
            winner=int(np.argmax(p[t]));emission[t,:]=-np.inf;emission[t,winner]=0.
        n,k=p.shape;back=np.empty((n,k),dtype=np.int32);cost=emission[0].copy();back[0]=-1
        for t in range(1,n):
            transitions=cost[:,None]-penalty*(1-np.eye(k))
            back[t]=transitions.argmax(0);cost=emission[t]+transitions[back[t],np.arange(k)]
        states=np.empty(n,dtype=np.int32);states[-1]=int(np.argmax(cost))
        for t in range(n-1,0,-1):states[t-1]=back[t,states[t]]
        before=np.argmax(active[lo:hi],axis=1);after=allowed[states]
        if np.array_equal(before,after):continue
        result[lo:hi]=0;result[np.arange(lo,hi),after]=1
        reports.append({'start_frame':int(lo),'end_frame':int(hi),'changed_frames':int(np.sum(before!=after)),
            'original_switches':int(np.sum(before[1:]!=before[:-1])),'stabilized_switches':int(np.sum(after[1:]!=after[:-1])),
            'protected_confident_frames':int(confident.sum()),'estimated_switch_odds':float(switch_odds)})
    return result,reports

def turns_from_frames(active,frame_start,frame_step,frame_duration,labels,duration):
    turns=[];center=frame_start+frame_duration/2
    for speaker in range(active.shape[1]):
        edges=np.diff(np.r_[False,active[:,speaker]>0,False].astype(np.int8))
        for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            start=max(0.,center+lo*frame_step);end=min(duration,center+hi*frame_step)
            if end>start:turns.append({'start':start,'end':end,'speaker':labels[str(speaker)],'raw_speaker':str(speaker)})
    return sorted(turns,key=lambda t:(t['start'],t['end'],t['speaker']))
