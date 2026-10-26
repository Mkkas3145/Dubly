"""Re-identify disconnected tracks by replaying real speech to Nemotron itself.

This is a conservative experimental reference comparison, not an expansion of
the model's eight native slots. No external speaker embedding is used.
"""
import numpy as np

POLICY={'version':2,'model':'same_loaded_nemotron','external_embedding_model':None,
    'anchors_per_probe':1,'independent_target_examples':2,'silence_seconds':.25,
    'minimum_exclusive_seconds':.5,'dominant_exclusive_fraction':.85,
    'merge':'unique_match_in_two_nonoverlapping_target_examples',
    'uncertain':'keep_separate','same_window_cannot_link':True,'global_speaker_cap':None}

def dominant(active,start,end,rate=100):
    part=np.asarray(active[round(start*rate):round(end*rate)],dtype=bool)
    if not len(part):return None
    exclusive=part & (part.sum(1)==1)[:,None]
    counts=exclusive.sum(0);total=int(counts.sum())
    if not total or total/rate<POLICY['minimum_exclusive_seconds']:return None
    index=int(counts.argmax())
    if counts[index]/total<POLICY['dominant_exclusive_fraction']:return None
    return index

def distinct_examples(examples,limit=2):
    """Prefer long clean samples and reject reuse of the same physical speech."""
    selected=[]
    for example in sorted(examples,key=lambda x:(-(x['end']-x['start']),x['start'])):
        if any(min(example['end'],v['end'])>max(example['start'],v['start']) for v in selected):continue
        selected.append(example)
        if len(selected)>=limit:break
    return selected

def verify(groups,audio,infer,event=lambda *a,**k:None,rate=16000):
    """groups: component -> nodes (window, slot), windows and source examples.

    infer accepts one short diagnostic waveform and returns native active frames.
    Its implementation caches by waveform content and pinned model revision.
    """
    banks=[];edges=[];audit=[]
    ordered=sorted(groups.values(),key=lambda g:min(g['examples'],key=lambda e:e['start'])['start'] if g['examples'] else float('inf'))
    gap=np.zeros(round(rate*POLICY['silence_seconds']),np.float32)
    for position,target in enumerate(ordered):
        samples=distinct_examples(target['examples']);candidates=[]
        eligible=[bank for bank in banks if not bank['windows'] & target['windows'] and bank['examples']]
        if len(samples)>=POLICY['independent_target_examples']:
            for offset in range(0,len(eligible),POLICY['anchors_per_probe']):
                batch=eligible[offset:offset+POLICY['anchors_per_probe']];matches=[]
                # A known different local voice is a negative control when available.
                control=next((g for g in ordered if g is not target and g['windows'] & target['windows'] and g['examples']),None)
                for trial,sample in enumerate(samples):
                    references=[distinct_examples(g['examples'])[trial % len(distinct_examples(g['examples']))] for g in batch]
                    parts=references+[sample]+([distinct_examples(control['examples'])[0]] if control else [])
                    clips=[];spans=[];cursor=0
                    for part in parts:
                        clip=audio[part['start']:part['end']]
                        clips.extend([clip,gap]);spans.append((cursor/rate,(cursor+len(clip))/rate));cursor+=len(clip)+len(gap)
                    prediction=infer(np.concatenate(clips))
                    labels=[dominant(prediction,*span) for span in spans]
                    anchors=labels[:len(batch)];label=labels[len(batch)]
                    valid=label is not None and all(v is not None for v in anchors) and len(set(anchors))==len(anchors)
                    if control:valid=valid and labels[-1] is not None and label!=labels[-1]
                    found=[i for i,v in enumerate(anchors) if valid and v==label]
                    matches.append(found[0] if len(found)==1 else None)
                    audit.append({'target':target['component'],'anchors':[g['component'] for g in batch],
                        'trial':trial,'source_examples':parts,'labels':labels,'valid':bool(valid),
                        'match':batch[found[0]]['component'] if len(found)==1 else None})
                if matches and matches[0] is not None and all(m==matches[0] for m in matches):candidates.append(batch[matches[0]])
        if len(candidates)==1:
            bank=candidates[0]
            edges.append({'nodes':[bank['nodes'][0],target['nodes'][0]],'evidence':'two_distinct_source_examples_same_native_slot'})
            bank['nodes']+=target['nodes'];bank['windows']|=target['windows'];bank['examples']+=target['examples']
        else:
            banks.append({**target,'nodes':list(target['nodes']),'windows':set(target['windows']),'examples':list(target['examples'])})
        event('verification',done=position+1,total=len(ordered),banks=len(banks),verified_merges=len(edges))
    return edges,audit
