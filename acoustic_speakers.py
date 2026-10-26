"""Discover acoustic voice groups without fixing the speaker count to ASR labels."""
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment

POLICY={'version':1,'anchor_seconds':1.5,'minimum_anchors':4,
        'average_linkage_cosine_distance':.4,'minimum_witness_cosine':.6,
        'minimum_median_cosine':.65,'minimum_margin':.08,
        'uncertain':'preserve_original_id','locked':'preserve_user_selection'}

def discover(rows,vectors):
    z=np.asarray(vectors,dtype=np.float32)
    if z.ndim!=2 or len(z)!=len(rows) or not np.isfinite(z).all():raise ValueError('Invalid speaker embeddings')
    norms=np.linalg.norm(z,axis=1)
    if np.any(norms<1e-6):raise ValueError('Empty speaker embedding')
    z=z/norms[:,None]
    eligible=[i for i,r in enumerate(rows) if r['seconds']>=POLICY['anchor_seconds'] and not r.get('speaker_locked')]
    if len(eligible)<6:return {'policy':POLICY,'groups':[],'decisions':[],'changes':[]}
    labels=fcluster(linkage(pdist(z[eligible],metric='cosine'),method='average'),POLICY['average_linkage_cosine_distance'],criterion='distance')
    groups=[ [eligible[j] for j,label in enumerate(labels) if label==k] for k in sorted(set(labels))]
    groups=[g for g in groups if len(g)>=POLICY['minimum_anchors']]
    groups.sort(key=lambda g:min(rows[i]['index'] for i in g))
    if len(groups)<2:return {'policy':POLICY,'groups':[],'decisions':[],'changes':[]}
    voices=sorted({str(r['speaker']) for r in rows})
    weights=np.array([[sum(min(rows[i]['seconds'],4) for i in g if str(rows[i]['speaker'])==v) for v in voices] for g in groups])
    a,b=linear_sum_assignment(-weights);names={int(i):voices[j] for i,j in zip(a,b) if weights[i,j]>0}
    reserved=set(voices);next_id=0
    for g in range(len(groups)):
        if g in names:continue
        while str(next_id) in reserved:next_id+=1
        names[g]=str(next_id);reserved.add(str(next_id))
    scores=z@z.T;decisions=[]
    for i,row in enumerate(rows):
        entry={'index':row['index'],'original_speaker':str(row['speaker']),'bank_speaker':str(row['speaker']),'action':'keep'}
        ranking=[]
        for k,group in enumerate(groups):
            peers=[j for j in group if j!=i and (rows[j].get('end',0)<=row.get('start',0) or rows[j].get('start',0)>=row.get('end',0))]
            best=sorted(peers,key=lambda j:float(scores[i,j]),reverse=True)[:3]
            if len(best)<3:continue
            values=[float(scores[i,j]) for j in best]
            ranking.append({'speaker':names[k],'median':float(np.median(values)),'minimum':min(values),'witnesses':[rows[j]['index'] for j in best]})
        ranking.sort(key=lambda r:r['median'],reverse=True);entry['ranking']=ranking
        if not row.get('speaker_locked') and ranking:
            best=ranking[0];margin=best['median']-ranking[1]['median'] if len(ranking)>1 else 1.
            if best['minimum']>=POLICY['minimum_witness_cosine'] and best['median']>=POLICY['minimum_median_cosine'] and margin>=POLICY['minimum_margin']:
                entry.update(bank_speaker=best['speaker'],confidence_margin=margin)
                if best['speaker']!=entry['original_speaker']:entry['action']='acoustic_recluster'
        decisions.append(entry)
    # A new identity needs at least three independently confident utterances.
    supported={v for v in names.values() if sum(d['bank_speaker']==v and d['action']=='acoustic_recluster' for d in decisions)>=3}
    for d in decisions:
        if d['bank_speaker'] not in voices and d['bank_speaker'] not in supported:
            d.update(bank_speaker=d['original_speaker'],action='keep')
    retained={k for k,v in names.items() if v in voices or v in supported}
    new=sorted({d['bank_speaker'] for d in decisions}-set(voices))
    compact={};number=0
    for voice in new:
        while str(number) in voices:number+=1
        compact[voice]=str(number);number+=1
    for d in decisions:d['bank_speaker']=compact.get(d['bank_speaker'],d['bank_speaker'])
    for k,v in list(names.items()):names[k]=compact.get(v,v)
    return {'policy':POLICY,'groups':[{'speaker':names[k],'anchors':[rows[i]['index'] for i in g]} for k,g in enumerate(groups) if k in retained],'decisions':decisions,'changes':[d for d in decisions if d['action']!='keep']}
