"""Constrained EEND-VC linking. No learned model or audio dependency here."""
import numpy as np

VERSION=3
POLICY={'version':VERSION,'overlap_agreement':.8,'minimum_anchor_seconds':.5,
        'profile':'animation_native_reference_consensus_v1',
        'positive_quantile':.1,'negative_quantile':.999,
        'embedding_threshold':'midpoint_positive_p10_negative_p999_or_conservative_abstention',
        'clustering':'complete_link_with_same_window_cannot_link','ownership':'largest_distance_from_window_edges',
        'global_speaker_cap':None}

class Components:
    def __init__(self,nodes):
        self.parent=list(range(len(nodes)));self.members={i:{i} for i in self.parent}
        self.windows={i:({node.get('constraint_window',node['window'])} if node.get('constraint_window',node['window']) is not None else set()) for i,node in enumerate(nodes)}
    def find(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return True
        if self.windows[a]&self.windows[b]:return False
        if a>b:a,b=b,a
        self.parent[b]=a;self.members[a]|=self.members.pop(b);self.windows[a]|=self.windows.pop(b)
        return True

def norm(x):
    x=np.asarray(x,dtype=float);length=np.linalg.norm(x)
    return x/length if np.isfinite(x).all() and length>0 else None

def calibrated_threshold(nodes,components):
    positive=[];negative=[]
    for i,a in enumerate(nodes):
        samples=[norm(v) for v in a.get('samples',[])]
        samples=[v for v in samples if v is not None]
        positive.extend(float(x@y) for j,x in enumerate(samples) for y in samples[j+1:])
        for j,b in enumerate(nodes[:i]):
            if a.get('vector') is None or b.get('vector') is None:continue
            score=float(a['vector']@b['vector'])
            if a['window']==b['window']:negative.append(score)
            elif components.find(i)==components.find(j):positive.append(score)
    if not positive or not negative:
        return None,{'positive_pairs':len(positive),'negative_pairs':len(negative),'status':'insufficient_calibration_no_embedding_merges'}
    p=float(np.quantile(positive,POLICY['positive_quantile']));n=float(np.quantile(negative,POLICY['negative_quantile']))
    separated=p>n
    # If the distributions overlap, do not claim a calibrated identity test.
    threshold=(p+n)/2 if separated else None
    return threshold,{'positive_pairs':len(positive),'negative_pairs':len(negative),
        'positive_p10':p,'negative_p999':n,'positive_quantile':POLICY['positive_quantile'],
        'negative_quantile':POLICY['negative_quantile'],'threshold':threshold,
        'status':'empirical_separation' if separated else 'ambiguous_calibration_no_embedding_merges'}

def link(windows,total_frames,step=.01,verified_edges=None,use_embeddings=True):
    nodes=[];lookup={}
    for wi,w in enumerate(windows):
        active=np.asarray(w['active'],dtype=bool)
        if len(active)!=w['end_frame']-w['start_frame']:raise ValueError('Window frame extent mismatch')
        for speaker in np.flatnonzero(active.any(0)):
            samples=w.get('embeddings',{}).get(int(speaker),[])
            vectors=[norm(v) for v in samples];vectors=[v for v in vectors if v is not None]
            vector=norm(np.mean(vectors,axis=0)) if vectors else None
            lookup[wi,int(speaker)]=len(nodes)
            nodes.append({'window':wi,'speaker':int(speaker),'samples':vectors,'vector':vector,
                'constraint_window':wi if w.get('trusted_constraints',True) else None})
    components=Components(nodes);edges=[];rejected=[]
    for wi,a in enumerate(windows):
        for wj in range(wi+1,len(windows)):
            b=windows[wj];lo=max(a['start_frame'],b['start_frame']);hi=min(a['end_frame'],b['end_frame'])
            if hi<=lo:continue
            av=np.asarray(a['active'][lo-a['start_frame']:hi-a['start_frame']],dtype=bool)
            bv=np.asarray(b['active'][lo-b['start_frame']:hi-b['start_frame']],dtype=bool)
            # Only exclusive single-speaker frames can establish an anchor.
            av=av & ((av.sum(1)==1)[:,None]);bv=bv & ((bv.sum(1)==1)[:,None])
            intersection=av.astype(float).T@bv.astype(float)
            union=av.sum(0)[:,None]+bv.sum(0)[None,:]-intersection
            scores=intersection/np.maximum(union,1)
            for sa in range(scores.shape[0]):
                sb=int(np.argmax(scores[sa]));score=float(scores[sa,sb])
                if score<POLICY['overlap_agreement'] or np.argmax(scores[:,sb])!=sa:continue
                if intersection[sa,sb]*step<POLICY['minimum_anchor_seconds']:continue
                if (wi,sa) not in lookup or (wj,sb) not in lookup:continue
                edges.append((score,lookup[wi,sa],lookup[wj,sb]))
    anchors=[]
    for score,a,b in sorted(edges,reverse=True):
        accepted=components.join(a,b)
        (anchors if accepted else rejected).append({'nodes':[a,b],'agreement':score,'method':'shared_audio_anchor'})
    verifications=[]
    for edge in verified_edges or []:
        a,b=(lookup[tuple(value)] for value in edge['nodes'])
        accepted=components.join(a,b)
        record={**edge,'accepted':accepted,'method':edge.get('method','same_nemotron_reference_probe')}
        verifications.append(record)
        if not accepted:rejected.append(record)
    if use_embeddings:threshold,calibration=calibrated_threshold(nodes,components)
    else:threshold,calibration=None,{'status':'disabled_no_external_embedding_model'}
    embedding_edges=[]
    if threshold is not None:
        while True:
            best=None;roots=sorted(components.members)
            for i,a in enumerate(roots):
                va=[nodes[n]['vector'] for n in components.members[a] if nodes[n]['vector'] is not None]
                if not va:continue
                for b in roots[i+1:]:
                    if components.windows[a]&components.windows[b]:continue
                    vb=[nodes[n]['vector'] for n in components.members[b] if nodes[n]['vector'] is not None]
                    if not vb:continue
                    score=min(float(x@y) for x in va for y in vb)
                    if score>=threshold and (best is None or score>best[0]):best=(score,a,b)
            if best is None:break
            score,a,b=best;embedding_edges.append({'nodes':[a,b],'similarity':score,'method':'complete_link_embedding'})
            components.join(a,b)
    mapping={i:components.find(i) for i in range(len(nodes))}
    # Select the most central prediction; never add probabilities from multiple
    # windows and accidentally invent extra overlapping voices at a seam.
    owners=np.full(total_frames,-1,dtype=int);strength=np.full(total_frames,-1.,dtype=float)
    for wi,w in enumerate(windows):
        lo,hi=w['start_frame'],w['end_frame'];frames=np.arange(hi-lo)
        quality=np.minimum(frames,hi-lo-1-frames)
        take=quality>strength[lo:hi];owners[lo:hi][take]=wi;strength[lo:hi][take]=quality[take]
    if np.any(owners<0):raise ValueError('Diarization windows do not cover the recording')
    roots=sorted(components.members);root_columns={r:i for i,r in enumerate(roots)}
    active=np.zeros((total_frames,len(roots)),dtype=bool)
    for wi,w in enumerate(windows):
        selected=np.flatnonzero(owners[w['start_frame']:w['end_frame']]==wi)
        for sa in np.flatnonzero(np.asarray(w['active']).any(0)):
            use=selected[np.asarray(w['active'])[selected,sa]]
            active[w['start_frame']+use,root_columns[mapping[lookup[wi,int(sa)]]]]=True
    turns=[]
    for c in range(active.shape[1]):
        change=np.diff(np.r_[False,active[:,c],False].astype(np.int8))
        for a,b in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)):
            turns.append({'start':float(a*step),'end':float(b*step),'speaker':f'E{c}'})
    turns.sort(key=lambda t:(t['start'],t['end'],t['speaker']))
    # Stable public IDs follow first appearance, not union-find root order.
    labels={}
    for t in turns:t['speaker']=labels.setdefault(t['speaker'],f'E{len(labels)}')
    report={'policy':POLICY,'local_nodes':len(nodes),'global_speakers':len(labels),
        'overlap_anchors':anchors,'constraint_rejections':rejected,'embedding_merges':embedding_edges,
        'calibration':calibration,'same_model_verifications':verifications,'node_mapping':[{'window':n['window'],'local_speaker':n['speaker'],
            'component':mapping[i]} for i,n in enumerate(nodes)]}
    return turns,report
