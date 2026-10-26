"""Joint acoustic boundaries for adjacent, differently identified voices.

No speaker embeddings or additional model calls. A waveform gap is evidence of
a cut, not evidence of speaker identity; Qwen's voice IDs remain unchanged.
"""
import soundfile as sf
from core import SR
from mai_pipeline import correct_edges as provisional_edges
from timing import envelope,measured_energy_gate,runs,HOP

POLICY={'version':2,'speaker_change':'joint_quiet_gap_between_utterance_centres',
        'max_anchor_uncertainty_seconds':1.,'minimum_quiet_frames':4,
        'gap_score':'duration_divided_by_one_plus_twice_anchor_distance',
        'utterance_edges':'anchored_acoustic_body_with_neighbor_and_voice_change_limits',
        'core_relative_rms':.10,'within_body_gap_seconds':.14,'articulation_guard_seconds':.03,
        'weak_edge_search_seconds':.20,'weak_edge_relative_to_core':.25,
        'unresolved':'retain_provisional_boundary_without_inventing_voice_identity'}


def acoustic_edges(clean,rows,corrected,audit):
    """Recover whole acoustic bodies, including short turns missing their anchor.

    Voice-change cuts remain hard fences. Inside them a text turn selects real
    acoustic support; an estimated timestamp in silence is not an empty voice.
    """
    info=sf.info(clean);snapshot=[dict(r) for r in corrected]
    with sf.SoundFile(clean) as audio:
        for i,(row,old) in enumerate(zip(rows,snapshot)):
            a,b=row['start'],row['end'];radius=POLICY['max_anchor_uncertainty_seconds']
            lower=snapshot[i-1]['end'] if i else 0.
            upper=snapshot[i+1]['start'] if i+1<len(rows) else info.duration
            previous=audit[i-1].get('following_voice_boundary',{}) if i else {}
            following=audit[i].get('following_voice_boundary',{})
            if previous.get('action')=='split_at_shared_pause':lower=max(lower,previous['right_start'])
            if following.get('action')=='split_at_shared_pause':upper=min(upper,following['left_end'])
            lo=max(0.,lower,a-radius);hi=min(info.duration,upper,b+radius)
            if hi<=lo:continue
            first=round(lo*SR);last=round(hi*SR)
            audio.seek(first);values=audio.read(last-first,dtype='float32',always_2d=True)
            levels=envelope(values);gate,noise=measured_energy_gate(levels,.015,values)
            core_gate=max(gate,noise['speech_rms']*POLICY['core_relative_rms'])
            strong=[(x,y) for x,y in runs(levels>core_gate) if y-x>=2]
            clusters=[];gap_frames=round(POLICY['within_body_gap_seconds']*SR/HOP)
            for x,y in strong:
                if clusters and x-clusters[-1][1]<=gap_frames:clusters[-1][1]=y
                else:clusters.append([x,y])
            # Choose the core bodies actually covered by the estimated turn.
            # If none is covered, recover the nearest bounded body, rather than
            # returning a silent clip at a late model timestamp.
            local_a=(a-first/SR)*SR/HOP;local_b=(b-first/SR)*SR/HOP
            selected=[c for c in clusters if min(c[1],local_b)-max(c[0],local_a)>=1]
            fallback=False
            if not selected and clusters:
                selected=[min(clusters,key=lambda c:max(c[0]-local_b,local_a-c[1],0))];fallback=True
            report={'method':'neighbor_fenced_acoustic_body','core_gate':core_gate,'support_gate':gate,
                'noise':noise,'search_start':lo,'search_end':hi,'silent_anchor_recovered':fallback}
            if not selected:
                audit[i]['acoustic_body']={**report,'action':'no_supported_body_keep_provisional'};continue
            start,end=selected[0][0],selected[-1][1]
            weak_gate=max(gate,core_gate*POLICY['weak_edge_relative_to_core'])
            supports=runs(levels>weak_gate);report['weak_edge_gate']=weak_gate
            max_weak=round(POLICY['weak_edge_search_seconds']*SR/HOP)
            # Retain weak consonants/decay connected to the selected strong body,
            # while refusing to follow an isolated later sound across a pause.
            changed=True
            while changed:
                changed=False
                for x,y in supports:
                    if y<start and start-y<=gap_frames and x>=selected[0][0]-max_weak:
                        start=x;changed=True
                    elif x<=start<y:start=max(x,selected[0][0]-max_weak)
                    if x>end and x-end<=gap_frames and y<=selected[-1][1]+max_weak:
                        end=y;changed=True
                    elif x<end<=y:end=min(y,selected[-1][1]+max_weak)
            guard=round(POLICY['articulation_guard_seconds']*SR)
            start=max(first,first+start*HOP-guard)/SR
            end=min(last,first+end*HOP+guard)/SR
            if end<=start:continue
            corrected[i]['start']=start;corrected[i]['end']=end
            audit[i]['start']=start;audit[i]['end']=end
            audit[i]['acoustic_body']={**report,'action':'use_supported_body','start':start,'end':end,
                'before_start':old['start'],'before_end':old['end']}
    # If two independent recoveries compete for the same body, do not invent a
    # split or fail the entire video. Retain their already non-overlapping
    # provisional acoustic windows and record why recovery was inconclusive.
    reverted=set()
    while True:
        conflicts={j for i,(left,right) in enumerate(zip(corrected,corrected[1:]))
            if left['end']>right['start'] for j in (i,i+1)}-reverted
        if not conflicts:break
        for i in conflicts:
            corrected[i]=dict(snapshot[i]);audit[i]['start']=snapshot[i]['start'];audit[i]['end']=snapshot[i]['end']
            audit[i].setdefault('acoustic_body',{}).update(action='competing_body_keep_provisional',
                start=snapshot[i]['start'],end=snapshot[i]['end'])
        reverted.update(conflicts)
    return corrected,audit


def correct_edges(clean,rows,progress=lambda *a:None):
    corrected,audit=provisional_edges(clean,rows,progress)
    if len(rows)<2:return acoustic_edges(clean,rows,corrected,audit)
    info=sf.info(clean)
    with sf.SoundFile(clean) as audio:
        for i,(left,right) in enumerate(zip(rows,rows[1:])):
            if left['speaker']==right['speaker'] and left['speaker']!='unknown':continue
            centre_left=(left['start']+left['end'])/2
            centre_right=(right['start']+right['end'])/2
            anchor=(left['end']+right['start'])/2
            radius=POLICY['max_anchor_uncertainty_seconds']
            # Read complete gaps, then constrain their centres. Cropping the
            # audio at the uncertainty radius would hide a long real pause.
            lo=max(0.,left['start']);hi=min(info.duration,right['end'])
            if hi<=lo:continue
            first=round(lo*SR);last=round(hi*SR)
            audio.seek(first);values=audio.read(last-first,dtype='float32',always_2d=True)
            levels=envelope(values);gate,noise=measured_energy_gate(levels,.015,values)
            candidates=[]
            for a,b in runs(levels<=gate):
                if b-a<POLICY['minimum_quiet_frames']:continue
                # A truncated gap at a speech-centre/search-window edge is not
                # evidence of the intervening pause; both speech sides matter.
                if a==0 or b>=len(levels):continue
                start=(first+a*HOP)/SR;end=min(last/SR,(first+b*HOP)/SR)
                midpoint=(start+end)/2
                if not centre_left<=midpoint<=centre_right or abs(midpoint-anchor)>radius:continue
                if start<=corrected[i]['start'] or end>=corrected[i+1]['end']:continue
                distance=abs(midpoint-anchor)
                score=(end-start)/(1+2*distance)
                candidates.append({'start':start,'end':end,'score':score})
            report={'method':'joint_different_voice_quiet_gap','neighbor_source_index':i+1,
                'anchor':anchor,'search_start':lo,'search_end':hi,'candidates':candidates,'noise':noise}
            if not candidates:
                report['action']='no_complete_quiet_gap_keep_provisional'
            else:
                gap=max(candidates,key=lambda g:g['score'])
                if gap['start']>corrected[i]['start'] and gap['end']<corrected[i+1]['end']:
                    report.update(action='split_at_shared_pause',before_left_end=corrected[i]['end'],
                        before_right_start=corrected[i+1]['start'],left_end=gap['start'],right_start=gap['end'])
                    corrected[i]['end']=gap['start'];corrected[i+1]['start']=gap['end']
                    audit[i]['end']=gap['start'];audit[i+1]['start']=gap['end']
                else:report['action']='invalid_gap_keep_provisional'
            audit[i]['following_voice_boundary']=report
    return acoustic_edges(clean,rows,corrected,audit)
