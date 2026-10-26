"""Reference-anchored re-diarization with the complete native target context.

Do not ask Nemotron to identify a speaker from isolated two-second pairs.
Keep every real local voice in the target window as a cannot-collapse control,
and require the same answer with references before and after that context.
"""
import numpy as np
import hashlib
from collections import OrderedDict
from nemotron_reidentification import distinct_examples,dominant

POLICY={'version':10,'model':'same_loaded_nemotron','external_embedding_model':None,
    'target':'complete_original_native_analysis_window','reference_examples':2,
    'orders':['reference_then_context','context_then_reference'],
    'context_checks':'target_exclusive_frame_consensus_and_target_specific_confident_negative_controls',
    'minimum_context_voice_seconds':.3,'minimum_context_agreement':.85,
    'reference_gap_seconds':.25,'capacity':'native_local_count_plus_one_at_most_eight',
    'primary_comparison':'two_complete_native_contexts_both_orders_combined_slots_at_most_eight',
    'ambiguous_multiple_banks':'joint_reference_confirmation_or_abstain','global_speaker_cap':None}
POLICY['global_refinement']='repeat_with_enlarged_verified_reference_banks_until_no_new_merges'
POLICY['query_authority']='primary_source_frame_ownership_not_auxiliary_view_id'
POLICY['negative_controls']='confirmed_primary_cannot_link_graph_not_merely_different_global_names'
POLICY['comparison_reuse']='exact_source_masks_negative_controls_reference_bank_and_query_order_only'

def context_labels(prediction,window,offset,rate=100):
    local=np.asarray(window['active'],bool)
    observed=np.asarray(prediction[offset:offset+len(local)],bool)
    if len(observed)!=len(local):return None,{'reason':'frame_extent'}
    exclusive=local.sum(1)==1;labels={};agreements={}
    for slot in np.flatnonzero(local.any(0)):
        mask=local[:,slot]&exclusive
        if mask.sum()/rate<POLICY['minimum_context_voice_seconds']:continue
        values=observed[mask];counts=(values & (values.sum(1)==1)[:,None]).sum(0)
        label=int(counts.argmax());fraction=float(counts[label]/max(1,mask.sum()))
        agreements[int(slot)]=fraction
        if fraction<POLICY['minimum_context_agreement']:continue
        labels[int(slot)]=label
    return labels,{'agreements':agreements}

def target_label(labels,window,speaker):
    if labels is None or speaker not in labels:return None
    value=labels[speaker]
    controls=window.get('cannot_link_slots')
    if window.get('trusted_constraints',True) and any(slot!=speaker and label==value and
            (controls is None or tuple(sorted((slot,speaker))) in controls) for slot,label in labels.items()):return None
    return value

def assess(prediction,window,offset,speaker,reference_spans,rate=100,*,context_result=None):
    """Check identity and ensure the full native scene has not collapsed."""
    labels,detail=context_labels(prediction,window,offset,rate) if context_result is None else context_result
    if labels is None:return None,detail
    label=target_label(labels,window,speaker)
    if label is None:return None,{'reason':'native_different_voices_collapsed' if speaker in labels else 'insufficient_target_context','labels':labels,**detail}
    references=[dominant(prediction,a,b,rate) for a,b in reference_spans]
    if any(v is None for v in references):return None,{'reason':'unclear_reference','labels':labels,'references':references}
    matched=all(v==label for v in references)
    return matched,{'reason':'match' if matched else 'different_voice','labels':labels,'references':references,**detail}

def verify(groups,windows,audio,infer,event=lambda *a,**k:None,rate=16000,*,reuse_scene_results=True):
    from nemotron_local_consensus import labels as unit_labels
    banks=[];edges=[];audit=[];hop=rate//100;gap=np.zeros(round(rate*POLICY['reference_gap_seconds']),np.float32)
    cache=getattr(infer,'_nemotron_scene_cache',None)
    if not isinstance(cache,OrderedDict):cache=OrderedDict();infer._nemotron_scene_cache=cache
    source_key=getattr(infer,'source_identity',None) or hashlib.sha256(audio.tobytes()).hexdigest()
    # A scene prediction is shared by several voice comparisons. Its frame
    # labels depend on the fixed context mask, not on which voice is queried.
    # Keep prediction objects alive with entries to prevent object-ID reuse.
    analyses=OrderedDict()
    def analyzed(prediction,index,offset):
        key=(id(prediction),index,offset)
        if key in analyses:
            analyses.move_to_end(key);return analyses[key][1:]
        labels,detail=context_labels(prediction,windows[index],offset)
        units=unit_labels(prediction,windows[index],offset)
        analyses[key]=(prediction,labels,detail,units)
        if len(analyses)>512:analyses.popitem(last=False)
        return labels,detail,units
    def scene(ranges,tail_gap=True):
        key=(source_key,tuple(ranges),len(gap),tail_gap);offsets=[];cursor=0
        for start,end in ranges:offsets.append(cursor);cursor+=end-start+len(gap)
        if key in cache:
            cache.move_to_end(key);return cache[key],offsets
        clips=[]
        for i,(start,end) in enumerate(ranges):
            clips.append(audio[start:end])
            if i+1<len(ranges) or tail_gap:clips.append(gap)
        prediction=infer(np.concatenate(clips));cache[key]=prediction
        if len(cache)>512:cache.popitem(last=False)
        return prediction,offsets
    def prefetch(requests):
        if not hasattr(infer,'batch'):return
        pending={}
        for ranges,tail_gap in requests:
            key=(source_key,tuple(ranges),len(gap),tail_gap)
            if key in cache or key in pending:continue
            clips=[]
            for i,(start,end) in enumerate(ranges):
                clips.append(audio[start:end])
                if i+1<len(ranges) or tail_gap:clips.append(gap)
            pending[key]=np.concatenate(clips)
        if pending:
            for key,prediction in zip(pending,infer.batch(list(pending.values()))):
                cache[key]=prediction
                if len(cache)>512:cache.popitem(last=False)
    stats={}
    for wi,w in enumerate(windows):
        active=w['active'];exclusive=active.sum(1)==1
        for slot in np.flatnonzero(active.any(0)):
            stats[wi,int(slot)]=(int((active[:,slot]&exclusive).sum()),int(active.any(0).sum()),len(active))
    ordered=sorted(groups.values(),key=lambda g:min(windows[wi]['start_frame'] for wi,slot in g['nodes']))
    plan_cache={}
    comparisons=getattr(infer,'_nemotron_comparison_cache',None)
    if not isinstance(comparisons,OrderedDict):comparisons=OrderedDict();infer._nemotron_comparison_cache=comparisons
    window_keys={wi:(wi,w['start_frame'],w['end_frame'],hashlib.sha256(w['active'].tobytes()).digest(),
        w.get('trusted_constraints',True),w.get('identity_repaired',False),
        None if w.get('cannot_link_slots') is None else tuple(sorted(w['cannot_link_slots']))) for wi,w in enumerate(windows)}
    def group_key(group):
        nodes=group.get('comparison_nodes',group['nodes'])
        value=(tuple((window_keys[wi],slot) for wi,slot in nodes),
            tuple((e['start'],e['end']) for e in group['examples']))
        return hashlib.sha256(repr(value).encode()).digest()
    group_keys={}
    def comparison_key(target,candidates):
        def key(group):
            # Banks grow after a target is processed; preserve order and content.
            signature=(group['component'],len(group['nodes']),len(group.get('comparison_nodes',group['nodes'])),len(group['examples']))
            if signature not in group_keys:group_keys[signature]=group_key(group)
            return group_keys[signature]
        return (source_key,key(target),tuple(key(bank) for bank in candidates))
    def plans(target,bank):
        key=(target['component'],bank['component'],len(target.get('comparison_nodes',target['nodes'])),len(bank.get('comparison_nodes',bank['nodes'])))
        if key in plan_cache:return plan_cache[key]
        def choices():
            for tw,ts in target.get('comparison_nodes',target['nodes']):
                for bw,bs in bank.get('comparison_nodes',bank['nodes']):
                    a,b=windows[tw],windows[bw]
                    if a.get('identity_repaired') or b.get('identity_repaired'):continue
                    ta,tc,td=stats[tw,ts];ba,bc,bd=stats[bw,bs]
                    if tc+bc<=8:yield (min(ta,ba),tw,ts,bw,bs,tc+bc,td+bd)
        first=min(choices(),key=lambda c:(-c[0],c[5],c[6],c[1],c[3]),default=None)
        selected=[]
        if first is not None:
            selected.append(first)
            second=min((c for c in choices() if (c[1],c[3])!=(first[1],first[3])),
                key=lambda c:(c[1]==first[1],c[3]==first[3],-c[0],c[5],c[6],c[1],c[3]),default=None)
            if second is not None:selected.append(second)
        plan_cache[key]=selected
        return selected
    def paired_contexts(target,bank):
        for _,tw,ts,bw,bs,*_ in plans(target,bank):
            ranges=[(windows[index]['start_frame']*hop,windows[index]['end_frame']*hop) for index in (tw,bw)]
            prefetch([(ranges,True),(list(reversed(ranges)),True)])
            outcomes=[]
            for trial in range(2):
                sequence=[tw,bw] if trial==0 else [bw,tw]
                prediction,positions=scene([(windows[index]['start_frame']*hop,windows[index]['end_frame']*hop) for index in sequence])
                offsets={index:position//hop for index,position in zip(sequence,positions)}
                tl,td,tu=analyzed(prediction,tw,offsets[tw]);bl,bd,bu=analyzed(prediction,bw,offsets[bw])
                target_id=target_label(tl,windows[tw],ts);reference_id=target_label(bl,windows[bw],bs)
                valid=target_id is not None and reference_id is not None
                matched=bool(valid and target_id==reference_id);outcomes.append(matched)
                row={'target':target['component'],'candidates':[bank['component']],
                    'trial':trial,'method':'two_complete_scenes','context_window':tw,'reference_window':bw,
                    'result':matched,'reason':'match' if matched else ('different_voice' if valid else 'context_disagreement'),
                    'target_labels':tl,'reference_labels':bl,'target_check':td,'reference_check':bd,
                    'target_unit_labels':tu,'reference_unit_labels':bu}
                audit.append(row)
            if len(outcomes)==2 and all(outcomes):return True
        return False
    def evaluate(target,candidates):
        if len(candidates)==1 and paired_contexts(target,candidates[0]):return True
        examples=[distinct_examples(b['examples']) for b in candidates]
        if any(len(v)<2 for v in examples):return False
        contexts=[]
        for wi,slot in target.get('comparison_nodes',target['nodes']):
            w=windows[wi];active=w['active'];pure=active[:,slot] & (active.sum(1)==1)
            if w.get('identity_repaired'):continue
            if active.any(0).sum()+len(candidates)>8:continue
            contexts.append((int(pure.sum()),wi,slot))
        contexts.sort(reverse=True)
        if not contexts:return False
        outcomes=[]
        for trial in range(2):
            _,wi,slot=contexts[min(trial,len(contexts)-1)];w=windows[wi]
            parts=[e for selected in examples for e in selected]
            ranges=[(e['start'],e['end']) for e in parts];context=(w['start_frame']*hop,w['end_frame']*hop)
            ranges=([context]+ranges) if trial else (ranges+[context])
            prediction,positions=scene(ranges,tail_gap=bool(trial))
            context_offset=positions[0 if trial else -1]//hop
            reference_positions=positions[1:] if trial else positions[:-1]
            spans=[(position/rate,(position+e['end']-e['start'])/rate) for position,e in zip(reference_positions,parts)]
            labels,check,_=analyzed(prediction,wi,context_offset)
            result,detail=assess(prediction,w,context_offset,slot,spans,context_result=(labels,check))
            audit.append({'target':target['component'],'candidates':[b['component'] for b in candidates],
                'trial':trial,'context_window':wi,'context_local_speaker':slot,'references':parts,'result':result,**detail})
            outcomes.append(result)
            if result is not True:return False
        return all(v is True for v in outcomes)
    def compare(target,candidates):
        key=comparison_key(target,candidates)
        if reuse_scene_results and key in comparisons:
            result,records=comparisons[key];comparisons.move_to_end(key)
            audit.extend({**row,'target':target['component'],'candidates':[b['component'] for b in candidates],
                'reused_exact_comparison':True} for row in records)
            return result
        start=len(audit);result=evaluate(target,candidates)
        if reuse_scene_results:
            comparisons[key]=(result,audit[start:])
            if len(comparisons)>1024:comparisons.popitem(last=False)
        return result
    for number,target in enumerate(ordered):
        candidates=[]
        # Batch only first scenes that the unchanged comparison path will need.
        requests=[]
        for bank in banks:
            if bank['windows']&target['windows']:continue
            if reuse_scene_results and comparison_key(target,[bank]) in comparisons:continue
            selected=plans(target,bank)
            if selected:
                _,tw,ts,bw,bs,*_=selected[0]
                ranges=[(windows[index]['start_frame']*hop,windows[index]['end_frame']*hop) for index in (tw,bw)]
                requests.extend([(ranges,True),(list(reversed(ranges)),True)])
        prefetch(requests)
        for bank_number,bank in enumerate(banks):
            if bank['windows']&target['windows']:continue
            event('verification_detail',done=number,total=len(ordered),
                candidate=bank_number+1,candidates=len(banks),verified_merges=len(edges))
            if compare(target,[bank]):candidates.append(bank)
        # Two provisional banks can themselves be disconnected copies of the
        # same voice. Verify them together before joining; never guess a winner.
        valid=bool(candidates)
        if len(candidates)>1:
            used=set(target['windows'])
            for bank in candidates:
                if used&bank['windows']:valid=False;break
                used|=bank['windows']
            if valid:
                valid=all(compare(target,[candidates[0],bank]) for bank in candidates[1:])
        if valid:
            bank=candidates[0]
            for extra in [*candidates[1:],target]:
                edges.append({'nodes':[bank['nodes'][0],extra['nodes'][0]],'evidence':'full_context_reference_prefix_suffix_consensus'})
                bank['nodes']+=extra['nodes'];bank['windows']|=extra['windows'];bank['examples']+=extra['examples']
                if 'comparison_nodes' in bank:bank['comparison_nodes']+=extra.get('comparison_nodes',extra['nodes'])
                if extra is not target:banks.remove(extra)
        else:banks.append({**target,'nodes':list(target['nodes']),'windows':set(target['windows']),'examples':list(target['examples']),
            **({'comparison_nodes':list(target['comparison_nodes'])} if 'comparison_nodes' in target else {})})
        event('verification',done=number+1,total=len(ordered),banks=len(banks),verified_merges=len(edges))
    return edges,audit
