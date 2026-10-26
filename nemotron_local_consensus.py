"""Repair fused local tracks only from repeated same-model scene comparisons."""
from collections import defaultdict
import numpy as np

POLICY={'version':5,'minimum_other_contexts':2,'different_to_same_vote_ratio':2,
    'unit_gap_seconds':.8,'minimum_unit_seconds':.12,'unit_label_agreement':.85,
    'timing':'preserve_original_activity_frames','repair':'split_fused_and_merge_confirmed_duplicate_local_tracks',
    'merge_minimum_contexts':3,'merge_same_to_different_vote_ratio':3,
    'short_native_view_seconds':20.,'short_native_view_step_seconds':5.,'view_margin_seconds':.5,
    'additional_views':'complete_utterance_boundaries_with_one_and_two_neighboring_turns','maximum_scene_seconds':30.,
    'short_view_identity':'advisory_until_repeated_consensus_not_a_hard_cannot_link'}

def units(window):
    active=np.asarray(window['active'],bool);result=[]
    for slot in np.flatnonzero(active.any(0)):
        changes=np.diff(np.r_[False,active[:,slot],False].astype(np.int8));parts=[]
        for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            if parts and a-parts[-1][1]<=round(window.get('identity_unit_gap_seconds',POLICY['unit_gap_seconds'])*100) and not np.any(np.delete(active[parts[-1][1]:a],slot,axis=1)):
                parts[-1][1]=int(b)
            else:parts.append([int(a),int(b)])
        for a,b in parts:result.append({'slot':int(slot),'start':a,'end':b})
    return sorted(result,key=lambda u:(u['start'],u['end'],u['slot']))

def labels(prediction,window,offset):
    active=np.asarray(window['active'],bool);observed=prediction[offset:offset+len(active)];out=[]
    if len(observed)!=len(active):return [None]*len(units(window))
    for unit in units(window):
        a,b=unit['start'],unit['end'];mask=active[a:b,unit['slot']]&(active[a:b].sum(1)==1)
        values=observed[a:b][mask];counts=(values&(values.sum(1)==1)[:,None]).sum(0)
        if len(values)<round(POLICY['minimum_unit_seconds']*100) or not len(counts):out.append(None);continue
        best=int(counts.argmax());out.append(best if counts[best]/len(values)>=POLICY['unit_label_agreement'] else None)
    return out

def repair(windows,audit):
    # Inspect minority turns before voting. Reuse the already computed native
    # predictions, splitting analysis units only in scenes with observed voice
    # changes inside a supposedly single speaker track.
    atomic_windows={r['context_window'] for r in audit if r.get('method')=='packed_reference_context' and r.get('mixed_target_voice') and 0<=r['context_window']<len(windows)}
    for wi in atomic_windows:windows[wi]['identity_unit_gap_seconds']=0.
    evidence=defaultdict(dict)
    for row in audit:
        if row.get('method')=='packed_reference_context':
            key=('packed',tuple(row['candidates']))
            values=row.get('atomic_unit_labels',row['target_unit_labels']) if row['context_window'] in atomic_windows else row['target_unit_labels']
            evidence[row['context_window']].setdefault(key,{})[row['trial']]=values
            continue
        if row.get('method')=='short_native_view':
            evidence[row['context_window']].setdefault(('short',row['view_start_frame'],row.get('view_end_frame')), {})['native']=row['target_unit_labels']
            continue
        if row.get('method')!='two_complete_scenes':continue
        for wi,other,key in [(row['context_window'],row['reference_window'],'target_unit_labels'),
                (row['reference_window'],row['context_window'],'reference_unit_labels')]:
            if key in row:evidence[wi].setdefault(other,{})[row['trial']]=row[key]
    reports=[];changed=False
    for wi,window in enumerate(windows):
        if not window.get('trusted_constraints',True):continue
        native=window['active'];items=units(window);negative=set();positive=[];votes={}
        for i,u in enumerate(items):
            for j,v in enumerate(items[:i]):
                same=different=0
                for trials in evidence[wi].values():
                    if set(trials) not in ({0,1},{'native'}):continue
                    relations=[]
                    for value in trials.values():
                        if len(value)!=len(items) or value[i] is None or value[j] is None:break
                        relations.append(value[i]==value[j])
                    if len(relations)==len(trials) and len(set(relations))==1:
                        if relations[0]:same+=1
                        else:different+=1
                votes[j,i]={'same':same,'different':different}
                if different>=POLICY['minimum_other_contexts'] and different>=window.get('identity_different_to_same_vote_ratio',POLICY['different_to_same_vote_ratio'])*same:
                    negative.add((j,i))
                if u['slot']!=v['slot'] and same>=POLICY['merge_minimum_contexts'] and same>=POLICY['merge_same_to_different_vote_ratio']*different:
                    positive.append((same-POLICY['merge_same_to_different_vote_ratio']*different,same,j,i))
        if not any(items[a]['slot']==items[b]['slot'] for a,b in negative) and not positive:continue
        clusters={i:{i} for i in range(len(items))};owner=list(range(len(items)))
        def join(a,b):
            a,b=owner[a],owner[b]
            if a==b:return
            for i in clusters[a]:
                for j in clusters[b]:
                    if tuple(sorted((i,j))) in negative:return
                    u,v=items[i],items[j];lo=max(u['start'],v['start']);hi=min(u['end'],v['end'])
                    if hi>lo and np.any(native[lo:hi,u['slot']] & native[lo:hi,v['slot']]):return
            merged=clusters[a]|clusters.pop(b);clusters[a]=merged
            for i in merged:owner[i]=a
        for _,__,a,b in sorted(positive,reverse=True):join(a,b)
        defaults=sorted((abs(u['start']-v['start']),j,i) for i,u in enumerate(items) for j,v in enumerate(items[:i]) if u['slot']==v['slot'])
        for _,a,b in defaults:join(a,b)
        clusters=[sorted(c) for c in clusters.values()]
        # Positive evidence can be rejected by stronger negative/overlap evidence.
        # A no-op partition must not trigger another whole-video verification or
        # disqualify an otherwise valid primary context merely by renumbering it.
        if all((items[i]['slot']==items[j]['slot'])==(owner[i]==owner[j]) for i in range(len(items)) for j in range(i)):continue
        if len(clusters)>8:
            reports.append({'window':wi,'status':'repair_exceeds_native_capacity','proposed_tracks':len(clusters)});continue
        clusters.sort(key=lambda c:min(items[i]['start'] for i in c));updated=np.zeros((len(native),len(clusters)),bool)
        for slot,c in enumerate(clusters):
            for i in c:
                u=items[i];updated[u['start']:u['end'],slot]=native[u['start']:u['end'],u['slot']]
        if not np.array_equal(updated.any(1),native.any(1)):raise AssertionError('Identity repair must not change speech activity timing')
        # Retain only original clean contiguous reference samples lying in a
        # single repaired identity. Never stitch individual frame islands.
        examples={}
        for spans in window.get('examples',{}).values():
            for sample in spans:
                a=(sample['start']-window['start_frame']*160)//160;b=(sample['end']-window['start_frame']*160)//160
                counts=updated[a:b].sum(0)
                if len(counts) and counts.max()>=.9*max(1,b-a):examples.setdefault(int(counts.argmax()),[]).append(sample)
        window['active']=updated;window['examples']=examples;window['identity_repaired']=True;changed=True
        reports.append({'window':wi,'start_frame':window['start_frame'],'end_frame':window['end_frame'],
            'status':'refined_by_repeated_native_scene_consensus','before':int(native.any(0).sum()),
            'after':int(updated.any(0).sum()),'units':items,'clusters':clusters,
            'different_pairs':[{'units':list(pair),**votes[pair]} for pair in sorted(negative)],
            'same_pairs':[{'units':[a,b],**votes[a,b]} for _,__,a,b in positive]})
    return changed,reports

def short_views(windows,audio,infer,event=lambda *a,**k:None):
    """Independent staggered local contexts expose identities fused in a long view."""
    hop=160;size=round(POLICY['short_native_view_seconds']*100);step=round(POLICY['short_native_view_step_seconds']*100)
    margin=round(POLICY['view_margin_seconds']*100);total=len(audio)//hop
    requested=defaultdict(set)
    for wi,w in enumerate(windows):
        items=units(w)
        for i,u in enumerate(items):
            for v in items[i+1:]:
                if u['slot']!=v['slot']:continue
                low=w['start_frame']+min(u['start'],v['start'])-margin
                high=w['start_frame']+max(u['end'],v['end'])+margin
                if high-low>size:continue
                for start in range(max(0,((high-size+step-1)//step)*step),max(0,low)+1,step):
                    if start+size<=total:requested[start,start+size].add(wi)
        for slot in {u['slot'] for u in items}:
            indexes=[i for i,u in enumerate(items) if u['slot']==slot]
            for first,last in zip(indexes,indexes[1:]):
                for neighbors in (1,2):
                    a=max(0,first-neighbors);b=min(len(items)-1,last+neighbors)
                    lo=max(0,w['start_frame']+items[a]['start']-margin)
                    hi=min(total,w['start_frame']+max(u['end'] for u in items[a:b+1])+margin)
                    if hi-lo<=round(POLICY['maximum_scene_seconds']*100):requested[lo,hi].add(wi)
    audit=[];views=[]
    def predictions():
        ordered=sorted(requested.items());size=max(1,getattr(infer,'batch_size',1))
        for first in range(0,len(ordered),size):
            items=ordered[first:first+size]
            waves=[audio[start*hop:end*hop] for (start,end),_ in items]
            values=infer.batch(waves) if hasattr(infer,'batch') else [infer(w) for w in waves]
            yield from zip(items,values)
    for number,(((start,end),owners),prediction) in enumerate(predictions()):
        count=end-start
        view={'start_frame':start,'end_frame':end,'active':prediction,'examples':{},'trusted_constraints':False}
        # Reuse only original physically clean continuous reference excerpts.
        for w in windows:
            if w['end_frame']<=start or w['start_frame']>=end:continue
            for spans in w['examples'].values():
                for e in spans:
                    a=e['start']//hop-start;b=e['end']//hop-start
                    if a<0 or b>count:continue
                    counts=prediction[a:b].sum(0)
                    if len(counts) and counts.max()>=.9*max(1,b-a):
                        selected=view['examples'].setdefault(int(counts.argmax()),[])
                        if e not in selected:selected.append(e)
        views.append(view)
        for wi in owners:
            w=windows[wi];aligned=np.zeros_like(w['active'],shape=(len(w['active']),prediction.shape[1]))
            lo=max(start,w['start_frame']);hi=min(end,w['end_frame'])
            aligned[lo-w['start_frame']:hi-w['start_frame']]=prediction[lo-start:hi-start]
            values=labels(aligned,w,0)
            for i,u in enumerate(units(w)):
                if w['start_frame']+u['start']<start+margin or w['start_frame']+u['end']>end-margin:values[i]=None
            audit.append({'method':'short_native_view','context_window':wi,'view_start_frame':start,'view_end_frame':end,'target_unit_labels':values})
        event('local_views',done=number+1,total=len(requested))
    return views,audit
