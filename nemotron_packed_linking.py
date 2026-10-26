"""Nemotron-only, multi-candidate full-context identity verification.

Each prediction checks every reference bank in its pack. No external speaker
encoder, pairwise all-scene cross product, or guessed embedding identity.
"""
import numpy as np
from nemotron_reidentification import distinct_examples,dominant
from nemotron_context_linking import context_labels,target_label
from nemotron_local_consensus import labels as unit_labels

POLICY={'version':5,'external_embedding_model':None,'capacity':8,'identity_unit_gap_seconds':.8,
    'identity_different_to_same_vote_ratio':0,
    'references_per_candidate':2,'orders':'references_then_context_and_context_then_references',
    'merge':'all_references_and_both_orders_agree_with_native_context_controls',
    'reuse':'exact_audio_model_precision_cache_shared_by_all_candidates',
    'minority_voice_guard':'every_meaningful_contiguous_native_unit_must_agree',
    'uncertain':'keep_separate','global_speaker_cap':None}


def strict_target(prediction,window,offset,slot,observed):
    """A long majority must never authorize merging another speaker's turns."""
    from nemotron_local_consensus import units,POLICY as UNIT_POLICY
    from nemotron_context_linking import POLICY as CONTEXT_POLICY
    temporary={**window,'identity_unit_gap_seconds':0.}
    values=unit_labels(prediction,temporary,offset);items=units(temporary)
    main=target_label(observed,window,slot)
    meaningful=[value for u,value in zip(items,values) if u['slot']==slot and u['end']-u['start']>=round(100*UNIT_POLICY['minimum_unit_seconds'])]
    known={v for v in meaningful if v is not None}
    active=window['active'];mask=active[:,slot]&(active.sum(1)==1)
    observed_frames=np.asarray(prediction[offset:offset+len(active)])[mask]
    if len(observed_frames):
        counts=(observed_frames&(observed_frames.sum(1)==1)[:,None]).sum(0)
        known.update(int(v) for v in np.flatnonzero(counts>=round(CONTEXT_POLICY['minimum_context_voice_seconds']*100)))
    mixed=len(known)>1
    valid=main is not None and not mixed and bool(meaningful) and all(v==main for v in meaningful)
    return main if valid else None,mixed,values


def verify(groups,windows,audio,infer,event=lambda *a,**k:None,rate=16000,**kwargs):
    banks=[];edges=[];audit=[];hop=rate//100;gap=np.zeros(rate//4,np.float32)
    ordered=sorted(groups.values(),key=lambda g:min(windows[wi]['start_frame'] for wi,slot in g['nodes']))
    for number,target in enumerate(ordered):
        contexts=[]
        for wi,slot in target.get('comparison_nodes',target['nodes']):
            w=windows[wi];active=w['active'];count=int(active.any(0).sum())
            if count>=POLICY['capacity']:continue
            pure=int((active[:,slot]&(active.sum(1)==1)).sum())
            if pure>=30:contexts.append((pure,wi,slot,count))
        contexts.sort(key=lambda c:(-c[0],c[3],c[1]))
        eligible=[b for b in banks if not b['windows']&target['windows'] and len(distinct_examples(b['examples']))>=2]
        matches=[]
        if contexts:
            first=contexts[0]
            count=first[3];pack_size=POLICY['capacity']-count
            packs=[eligible[i:i+pack_size] for i in range(0,len(eligible),pack_size)]
            requests=[]
            for pi,pack in enumerate(packs):
                examples=[distinct_examples(b['examples']) for b in pack]
                flat=[e for samples in examples for e in samples]
                for trial,context in enumerate((first,first)):
                    _,wi,slot,_=context;w=windows[wi]
                    refs=[(e['start'],e['end']) for e in flat]
                    main=(w['start_frame']*hop,w['end_frame']*hop)
                    ranges=refs+[main] if trial==0 else [main]+refs
                    clips=[];positions=[];cursor=0
                    for a,b in ranges:
                        positions.append(cursor);clips.extend((audio[a:b],gap));cursor+=b-a+len(gap)
                    requests.append((pi,trial,wi,slot,flat,positions,np.concatenate(clips)))
            results={}
            batch=max(1,getattr(infer,'batch_size',1))
            for start in range(0,len(requests),batch):
                items=requests[start:start+batch]
                waves=[r[-1] for r in items]
                predictions=infer.batch(waves) if hasattr(infer,'batch') else [infer(w) for w in waves]
                for request,prediction in zip(items,predictions):
                    pi,trial,wi,slot,flat,positions,_=request;pack=packs[pi]
                    offset=positions[-1 if trial==0 else 0]//hop
                    ref_positions=positions[:-1] if trial==0 else positions[1:]
                    observed,detail=context_labels(prediction,windows[wi],offset)
                    target_id,mixed,atomic_labels=strict_target(prediction,windows[wi],offset,slot,observed)
                    ref_ids=[dominant(prediction,p/rate,(p+e['end']-e['start'])/rate) for p,e in zip(ref_positions,flat)]
                    found=[]
                    for i,bank in enumerate(pack):
                        ids=ref_ids[2*i:2*i+2]
                        if target_id is not None and len(ids)==2 and all(v==target_id for v in ids):found.append(bank['component'])
                    results[pi,trial]=set(found)
                    audit.append({'method':'packed_reference_context','target':target['component'],
                        'candidates':[b['component'] for b in pack],'trial':trial,'context_window':wi,
                        'context_local_speaker':slot,'target_unit_labels':unit_labels(prediction,windows[wi],offset),
                        'atomic_unit_labels':atomic_labels,'mixed_target_voice':mixed,
                        'target_labels':observed,'reference_labels':ref_ids,'matched_components':found,
                        'references':flat,'context_check':detail})
                event('verification_detail',done=number,total=len(ordered),candidate=min(start+batch,len(requests)),
                    candidates=len(requests),verified_merges=len(edges))
            for pi,pack in enumerate(packs):
                accepted=results.get((pi,0),set())&results.get((pi,1),set())
                matches.extend(b for b in pack if b['component'] in accepted)
        # Several banks may be disconnected copies of the same voice. Native
        # same-scene different-voice constraints still forbid joining them.
        used=set(target['windows']);valid=bool(matches)
        for bank in matches:
            if used&bank['windows']:valid=False;break
            used|=bank['windows']
        if valid:
            base=matches[0]
            for other in [*matches[1:],target]:
                edges.append({'nodes':[base['nodes'][0],other['nodes'][0]],
                    'method':'packed_native_reference_context_consensus','evidence':'two_clean_references_both_orders_native_controls'})
                extra_nodes=list(other['nodes']);extra_contexts=list(other.get('comparison_nodes',extra_nodes))
                base['nodes']+=extra_nodes;base['windows']|=other['windows'];base['examples']+=other['examples']
                if 'comparison_nodes' in base:base['comparison_nodes']+=extra_contexts
                if other is not target:banks.remove(other)
        else:
            banks.append({**target,'nodes':list(target['nodes']),'windows':set(target['windows']),
                'examples':list(target['examples']),
                **({'comparison_nodes':list(target['comparison_nodes'])} if 'comparison_nodes' in target else {})})
        event('verification',done=number+1,total=len(ordered),banks=len(banks),verified_merges=len(edges))
    return edges,audit
