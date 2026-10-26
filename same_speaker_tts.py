"""Collect up to 16 turns per Nemotron speaker, then restore source placement."""
from copy import deepcopy

VERSION=4
MAX_PARTS=16
POLICY={'version':VERSION,'max_parts':MAX_PARTS,
        'identity':'nemotron_eend_vc_single_speaker','membership':'same_speaker_video_order',
        'tags':'distinct_reference_slot_0_to_15','reference':'concatenated_selected_turns',
        'reference_gap_seconds':0.,
        'generated_placement':'scribe_once_then_per_turn_alignment'}

def group_transcript(transcript,timeline):
    result=deepcopy(transcript)
    if result.get('tts_grouping_policy')==POLICY:return result
    segments=result.get('ungrouped_segments',result.get('segments',[]));turns=timeline['turns']
    def identity(segment):
        voices={t.get('speaker') for t in turns
                if min(t['end'],segment['end'])-max(t['start'],segment['start'])>1e-7}
        if len(voices)!=1 or None in voices or 'unknown' in voices:return None
        return next(iter(voices))
    grouped=[];open_groups={}
    for index,segment in enumerate(segments):
        voice=identity(segment)
        part={k:segment[k] for k in ('start','end','original_text','translated_text','source_region_id')}
        part.update(_tts_source_index=index,tts_speaker_id=voice)
        group=open_groups.get(voice) if voice is not None else None
        if group is None or len(group['_tts_parts'])>=MAX_PARTS:
            group={**segment,'_tts_parts':[],'source_region_ids':[],'tts_speaker_id':voice}
            grouped.append(group)
            if voice is not None:open_groups[voice]=group
        group['_tts_parts'].append(part);group['end']=segment['end']
        group['source_region_ids'].append(segment['source_region_id'])
        for key in ('original_text','translated_text'):
            group[key]=' '.join(p[key] for p in group['_tts_parts'])
    result['ungrouped_segments']=deepcopy(segments);result['segments']=grouped
    result['tts_grouping_policy']=POLICY
    result['tts_grouping_counts']={'source_segments':len(segments),'generation_groups':len(grouped),
        'group_sizes':[len(g['_tts_parts']) for g in grouped]}
    result.pop('reference_windows',None)
    return result

def tagged_request(segment):
    parts=segment.get('_tts_parts')
    if not parts:return segment['original_text'],segment['translated_text']
    if not 1<=len(parts)<=MAX_PARTS:raise ValueError('TTS reference slot count exceeds 16')
    if '_tts_output_slot' in segment:
        slot=segment['_tts_output_slot']
        if type(slot) is not int or not 0<=slot<len(parts):raise ValueError('Invalid individual output reference slot')
        reference=' '.join(f'<|speaker:{i}|>{part["original_text"]}' for i,part in enumerate(parts))
        return reference,f'<|speaker:{slot}|>{parts[slot]["translated_text"]}'
    return tuple(' '.join(f'<|speaker:{i}|>{part[key]}' for i,part in enumerate(parts))
                 for key in ('original_text','translated_text'))

def restore_turns_for_alignment(grouped):
    """Return chronological turns; groups can contain nonadjacent source indices."""
    result=deepcopy(grouped);ordered=[]
    for group_index,group in enumerate(grouped['segments']):
        for slot,part in enumerate(group['_tts_parts']):
            turn=dict(part);source_index=turn.pop('_tts_source_index')
            ordered.append((source_index,{**turn,'_tts_group':group_index,'_tts_slot':slot}))
    ordered.sort(key=lambda item:item[0])
    if [i for i,_ in ordered]!=list(range(len(ordered))):raise RuntimeError('TTS group source indices are duplicated or missing')
    turns=[turn for _,turn in ordered];plan=[[] for _ in grouped['segments']]
    for i,turn in enumerate(turns):plan[turn['_tts_group']].append(i)
    result['segments']=turns;result['tts_group_plan']=plan
    result.pop('reference_windows',None)
    return result

def synthesize_aligned(segments,clean,folder,cfg,progress,status=None):
    from batch_tts import synthesize_batches
    import render
    by_group={}
    for i,segment in enumerate(segments):
        by_group.setdefault(segment['_tts_group'],[]).append(i)
    groups=[sorted(indices,key=lambda i:segments[i]['_tts_slot']) for indices in by_group.values()]
    return synthesize_batches(segments,clean,folder,cfg,progress,status,
        groups=groups,tag_mode='reference_slot',reference_mode='concatenated',reference_gap_seconds=POLICY['reference_gap_seconds'],
        mode='same_speaker_tagged_groups_aligned',instructions=render.tts_instructions(cfg['language']))
