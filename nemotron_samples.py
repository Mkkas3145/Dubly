"""Sample real contiguous clean speech for identity; never splice frame islands."""
import numpy as np
def contiguous_samples(wave,probabilities,active,rate,hop,policy):
    from timing import runs
    exclusive=np.asarray(active).sum(1)==1;result={}
    minimum=round(policy['embedding_min_seconds']*rate)
    target=round(policy['embedding_sample_seconds']*rate)
    for speaker in np.flatnonzero(np.asarray(active).any(0)):
        pure=(probabilities[:,speaker]>=policy['embedding_pure_probability'])&exclusive
        candidates=[]
        for a,b in runs(pure):
            lo,hi=int(a*hop),min(len(wave),int(b*hop))
            if hi-lo<minimum:continue
            count=max(1,int(np.ceil((hi-lo)/target)))
            for start in np.linspace(lo,max(lo,hi-target),count).astype(int):
                end=min(hi,start+target)
                if end-start>=minimum:candidates.append((int(start),int(end)))
        candidates=sorted(set(candidates))
        if len(candidates)>policy['embedding_max_samples']:
            candidates=[candidates[i] for i in np.linspace(0,len(candidates)-1,policy['embedding_max_samples']).astype(int)]
        if candidates:result[int(speaker)]=candidates
    return result
