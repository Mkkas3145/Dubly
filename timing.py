"""Verified pauses first; accelerate with Speedy only to avoid the next dialogue."""
from __future__ import annotations
import subprocess
import numpy as np
import soundfile as sf
from core import SR, command
from stretch import stretch,ENGINE as TEMPO_ENGINE,VERSION as TEMPO_VERSION
from stretch import stretch as collision_stretch,ENGINE as COLLISION_ENGINE,VERSION as COLLISION_VERSION
from mfa_hint import alignment_hint
from pause_matching import source_guided_plan,word_boundary_insertions

FIT_VERSION = 48
TIMING_POLICY = 'verified_pauses_then_unspoken_tail_reduction_then_speedy_on_overlap'
MAX_SLOW_DURATION_FACTOR = 1.0  # Never slow speech down.
NATURAL_MIN_TEMPO = 1.0
NATURAL_MAX_TEMPO = None  # No speedup ceiling.
MIN_TEMPO = .85  # Historical R3 comparison tools only; production has no slowdown floor.
HOP = round(.01*SR)


def natural_audio(source,destination,frames,speech_window=None):
    """Decode the entire TTS. Preserve silence, articulation and tail samples."""
    temporary=destination.with_name(destination.stem+'-natural-partial.wav')
    try:
        command(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-ar',str(SR),'-ac','2',
            '-c:a','pcm_f32le',str(temporary)])
        values,rate=sf.read(temporary,dtype='float32',always_2d=True)
        if rate!=SR or values.shape[1]!=2 or not len(values) or not np.isfinite(values).all():
            raise RuntimeError('TTS audio decoding failed.')
        if float(np.max(np.abs(values)))<1e-6:raise RuntimeError('TTS returned silence.')
        temporary.replace(destination)
        duration=len(values)/SR
        return {'fit_version':FIT_VERSION,'timing_method':'natural_tts_full_duration_no_pause_or_speed_adjustment_overlap_allowed','output_frames':len(values),
            'tempo':1.,'tempo_engine':'none','tempo_engine_version':None,'tempo_fallback':False,
            'tempo_duration_change_seconds':0.,'pause_adjustment_enabled':False,'pause_adjustments':[],
            'pause_added_seconds':0.,'pause_removed_seconds':0.,'pause_adjustment_share':0.,'new_pause_count':0,
            'mfa_status':'not_used','word_alignment_status':'not_used','pause_planning':'disabled',
            'pause_planning_api_requests':0,'tts_speech_seconds_before':duration,'tts_speech_seconds_after':duration,
            'tts_natural_seconds':duration,'source_interval_seconds':frames/SR,
            'speech_window_frames':list(speech_window or (0,frames)),
            'placement_offset_frames':0,'overlap_allowed':True,'tts_samples_trimmed':0}
    finally:temporary.unlink(missing_ok=True)


def measured_energy_gate(levels,relative=.02,values=None):
    """Estimate a persistent quiet floor separately from speech amplitude."""
    levels=np.asarray(levels);peak=float(levels.max()) if len(levels) else 0.
    positive=levels[levels>max(1e-7,peak*.001)]
    strong=levels[levels>peak*.01]
    level=float(np.percentile(strong,90)) if len(strong) else peak
    threshold=max(1e-6,level*relative);floor=0.;persistent=False
    flatness=None;protected=False
    if len(positive):
        candidate=float(np.percentile(positive,35))
        # A low-energy population is only noise evidence when it persists and
        # is clearly separated from the strong speech population.
        low=(levels>1e-7)&(levels<=candidate*2)
        sustained=max((b-a for a,b in runs(low)),default=0)
        persistent=bool(candidate<=level*.08 and sustained*HOP/SR>=.12)
        if persistent:
            # Quiet harmonic speech must not be treated as stationary noise.
            # Examine only a handful of real low-energy windows in the speech
            # band, avoiding codec high-frequency roll-off in this statistic.
            if values is not None and len(values)>=1024:
                channel=int(np.argmax(np.mean(values**2,axis=0)))
                mono=values[:,channel];indices=np.flatnonzero(low)
                selected=indices[np.linspace(0,len(indices)-1,min(12,len(indices))).astype(int)]
                frequency=np.fft.rfftfreq(1024,1/SR);band=(frequency>=150)&(frequency<=8000)
                measurements=[]
                for i in selected:
                    start=max(0,min(len(mono)-1024,int(i)*HOP-512))
                    power=np.abs(np.fft.rfft(mono[start:start+1024]*np.hanning(1024)))**2+1e-20
                    power=power[band]
                    measurements.append(float(np.exp(np.mean(np.log(power)))/np.mean(power)))
                flatness=float(np.median(measurements))
                protected=flatness<.025
            if protected:
                persistent=False
                threshold=max(1e-6,min(threshold,candidate*.6))
            else:
                floor=candidate
                threshold=max(threshold,min(level*.15,floor*3))
    return float(threshold),{'noise_floor_rms':float(floor),'speech_rms':float(level),
        'threshold_rms':float(threshold),'persistent_noise_detected':persistent,
        'low_energy_spectral_flatness':flatness,'quiet_harmonic_speech_protected':bool(protected)}


def trim_outer_silence(values):
    """Measure quiet edges from frame energy, keeping articulation guards."""
    levels=envelope(values)
    if not len(levels) or levels.max()<1e-7:raise RuntimeError('TTS returned silence.')
    threshold,_=measured_energy_gate(levels,.02,values)
    # Keep guards around measured speech; never use ASR endpoints here.
    active=levels>threshold
    stable=[(a,b) for a,b in runs(active) if b-a>=3]
    if not stable:return values,0,0
    lo=int(max(0,stable[0][0]*HOP-round(.08*SR)))
    hi=int(min(len(values),stable[-1][1]*HOP+round(.10*SR)))
    # A short low-energy edge is not enough evidence to discard an expressive
    # release/breath. Keep it intact. Long stationary synthesis noise can still
    # be removed so it does not force needless collision acceleration.
    if lo<=round(.08*SR):lo=0
    tail_peak=float(np.max(np.abs(values[hi:]))) if hi<len(values) else 0.
    if len(values)-hi<=round(.35*SR) and tail_peak>max(1e-7,float(np.max(np.abs(values)))*.0001):hi=len(values)
    return values[lo:hi],lo,len(values)-hi


def pause_only_audio(source,destination,frames,reference=None,text=None,language='ko',speech_window=None,*,alignment_cache=None,preserve_outer_edges=False,preserve_tail=False):
    """Edit verified quiet pauses, keeping all other decoded samples and any overrun."""
    decoded=destination.with_name(destination.stem+'-pause-source.wav')
    temporary=destination.with_name(destination.stem+'-pause-partial.wav')
    try:
        result=natural_audio(source,decoded,frames,speech_window)
        values,_=sf.read(decoded,dtype='float32',always_2d=True)
        _,noise_floor_measurement=measured_energy_gate(envelope(values),.02,values)
        if preserve_outer_edges:trimmed_head=trimmed_tail=0
        elif preserve_tail:
            _,trimmed_head,_=trim_outer_silence(values)
            values=values[trimmed_head:];trimmed_tail=0
        else:values,trimmed_head,trimmed_tail=trim_outer_silence(values)
        window_lo,window_hi=speech_window or (0,frames)
        if not 0<=window_lo<window_hi<=frames:raise RuntimeError('Invalid dialogue speech interval.')
        source_lo,source_hi=reference_layout(reference,frames)
        source_lo=int(max(source_lo,window_lo));source_hi=int(min(source_hi,window_hi))
        if source_hi<=source_lo:raise RuntimeError('Invalid source speech interval.')
        # Fit the actual utterance, not seconds of generated external silence.
        target=source_hi-source_lo
        hint={'status':'not_needed'};plan={'boundaries':[],'matches':[],'mfa_reviews':[]};changes=[]
        edited=values
        if text and target!=len(values):
            hint=alignment_hint(values,text,language,alignment_cache or destination.with_name(destination.stem+'-mfa.json'))
            if hint['status']=='aligned':
                plan=source_guided_plan(values,reference,frames,(window_lo,window_hi),hint,
                    allow_new_pauses=target>len(values))
                edited,changes=adjust_pauses(values,target,plan['boundaries'])
        if not np.isfinite(edited).all():raise RuntimeError('Invalid pause adjustment result.')
        placed=np.concatenate([np.zeros((source_lo,values.shape[1]),dtype=np.float32),edited])
        sf.write(temporary,placed,SR,subtype='FLOAT');temporary.replace(destination)
        added=sum(max(0,c['added_frames']) for c in changes)/SR
        removed=sum(max(0,-c['added_frames']) for c in changes)/SR
        delta=target-len(values)
        result.update(timing_method='trim_external_silence_then_verified_pause_adjustment',output_frames=len(placed),pause_adjustment_enabled=True,
            pause_adjustments=changes,pause_added_seconds=added,pause_removed_seconds=removed,
            pause_adjustment_share=1. if changes else 0.,
            pause_target_coverage=min(1.,abs(len(edited)-len(values))/max(1,abs(delta))),
            pause_target_seconds=(source_lo+target)/SR,remaining_duration_error_seconds=(len(edited)-target)/SR,
            tts_speech_seconds_after=len(edited)/SR,tts_samples_trimmed=trimmed_head+trimmed_tail,
            outer_silence_measurement=noise_floor_measurement,
            tts_leading_silence_removed_seconds=trimmed_head/SR,tts_trailing_silence_removed_seconds=trimmed_tail/SR,
            verified_batch_edges_preserved=bool(preserve_outer_edges),
            tts_tail_preserved_until_collision=bool(preserve_tail or preserve_outer_edges),
            aligned_spoken_end_frames=(round(max(w['end'] for w in hint.get('words',[]))*SR)+source_lo+
                sum(c['added_frames'] for c in changes if c['at_frame']<round(max(w['end'] for w in hint['words'])*SR)))
                if hint.get('status')=='aligned' and hint.get('words') else None,
            placement_offset_frames=source_lo,
            source_speech_start_seconds=source_lo/SR,source_speech_end_seconds=source_hi/SR,
            mfa_status=hint['status'],word_alignment_status=hint['status'],
            mfa_runtime=hint.get('runtime_stats'),mfa_reason=hint.get('reason'),
            mfa_role='advisory_word_and_phone_boundary_check',pause_planning='algorithmic_pause_only',
            pause_boundary_count=len(plan['boundaries']),pause_boundary_reviews=plan['mfa_reviews'],
            source_pause_matches=plan['matches'],word_junction_reviews=plan.get('word_junction_reviews',[]),
            new_pause_count=sum(c['kind']=='inserted_word_boundary_pause' for c in changes),
            pause_boundary_policy='interword_silence_only_no_consonant_closures',
            pause_distribution='existing_pause_weighted_adaptive_capacity',speed_correction_enabled=False)
        return result
    finally:
        decoded.unlink(missing_ok=True);temporary.unlink(missing_ok=True)


def shorten_collision_pauses(values,budget):
    """Use measured long quiet pauses before changing any spoken samples."""
    amount=max(0,len(values)-budget)
    if not amount:return values,[]
    candidates=[{**p,'boundary':{'minimum_pause_seconds':.08,
        'expansion_policy':'measured_silence_before_collision_speedup'}}
        for p in acoustic_pause_slots(values,.18)]
    if not candidates:return values,[]
    capacities=[max(0,p['hi']-p['lo']-round(.08*SR)) for p in candidates]
    changes=allocate(min(amount,sum(capacities)),[p['weight'] for p in candidates],capacities)
    return edit_pause_slots(values,candidates,[-n for n in changes])


def reduce_quiet_head_for_collision(values,budget,placement_offset=0):
    """Keep intentional timing unless speedup is needed; never remove source padding."""
    if budget is None or len(values)<=budget:return values,0
    offset=max(0,min(len(values),int(placement_offset)))
    body=values[offset:]
    if not len(body):return values,0
    # Deliberately much quieter than breath or consonants. Stop at the first
    # audible window, retaining an onset guard instead of trusting ASR starts.
    gate=min(.0002,float(np.max(np.abs(body)))*.002)
    if gate<=0:return values,0
    hop=max(1,round(.01*SR));quiet=0
    for lo in range(0,len(body),hop):
        window=body[lo:lo+hop]
        if float(np.sqrt(np.mean(window.astype(np.float64)**2)))>gate:break
        quiet=lo+len(window)
    removable=max(0,quiet-round(.08*SR))
    removed=min(removable,len(values)-budget)
    if not removed:return values,0
    return np.concatenate([values[:offset],body[removed:]]),removed


def reduce_unspoken_tail(values,budget,spoken_end):
    """Reduce only a verified post-word tail, with a gentle fade, before speedup."""
    report={'removed_frames':0,'fade_frames':0,'status':'not_needed'}
    if budget is None or len(values)<=budget:return values,report
    if spoken_end is None or not 0<spoken_end<=len(values):
        return values,{**report,'status':'no_verified_spoken_endpoint'}
    protected=min(len(values),int(spoken_end)+round(.08*SR))
    end=max(int(budget),protected)
    if end>=len(values):return values,{**report,'status':'no_unspoken_tail_available'}
    result=values[:end].copy()
    fade=min(round(.05*SR),max(0,end-int(spoken_end)-round(.02*SR)))
    if fade>1:result[-fade:]*=((1+np.cos(np.linspace(0,np.pi,fade)))/2).astype(np.float32)[:,None]
    return result,{'removed_frames':len(values)-end,'fade_frames':fade,'status':'reduced_only_for_collision'}


def stretch_speech_keep_pauses(values,destination,frames):
    """Stretch speech chunks while copying measured interior quiet samples."""
    pauses=acoustic_pause_slots(values,.075)
    intervals=[];cursor=0
    for pause in pauses:
        intervals.append((cursor,pause['lo'],False));intervals.append((pause['lo'],pause['hi'],True));cursor=pause['hi']
    intervals.append((cursor,len(values),False))
    speech=[(lo,hi) for lo,hi,quiet in intervals if not quiet and hi>lo]
    quiet_frames=int(sum(hi-lo for lo,hi,quiet in intervals if quiet))
    available=frames-quiet_frames
    if available<len(speech):raise RuntimeError('Insufficient space to apply speech-only time stretching while preserving speech.')
    sizes=allocate(available,[hi-lo for lo,hi in speech]);parts=[];files=[];index=0
    try:
        for lo,hi,quiet in intervals:
            if hi<=lo:continue
            if quiet:parts.append(values[lo:hi]);continue
            source=destination.with_name(destination.stem+f'-speech-{index:03d}.wav')
            output=destination.with_name(destination.stem+f'-speech-{index:03d}-out.wav');files.extend([source,output])
            sf.write(source,values[lo:hi],SR,subtype='FLOAT')
            corrected=collision_stretch(source,output,sizes[index]);index+=1
            if not isinstance(corrected,np.ndarray) or corrected.ndim!=2 or corrected.shape[1]!=2 or not len(corrected) or not np.isfinite(corrected).all():
                raise RuntimeError('Speedy returned invalid speech chunk')
            parts.append(corrected)
        return np.concatenate(parts),quiet_frames,len(speech)
    finally:
        for path in files:path.unlink(missing_ok=True)


def bounded_timing_audio(source,destination,frames,reference=None,text=None,language='ko',speech_window=None,*,collision_budget_frames=None,preserve_outer_edges=False,protected_spoken_end_frames=None,preserve_tail=False):
    """Correct collisions when possible; retain valid pause audio if Speedy fails."""
    if collision_budget_frames is not None and (not isinstance(collision_budget_frames,(int,np.integer)) or collision_budget_frames<1):
        raise RuntimeError('Invalid time interval before the next dialogue.')
    if collision_budget_frames is not None:collision_budget_frames=int(collision_budget_frames)
    paused=destination.with_name(destination.stem+'-pause-only.wav')
    body_file=destination.with_name(destination.stem+'-tempo-input.wav')
    stretched=destination.with_name(destination.stem+'-tempo-output.wav')
    temporary=destination.with_name(destination.stem+'-bounded-partial.wav')
    try:
        result=pause_only_audio(source,paused,frames,reference,text,language,speech_window,
            alignment_cache=destination.with_name(destination.stem+'-mfa.json'),preserve_outer_edges=preserve_outer_edges,preserve_tail=preserve_tail)
        values,_=sf.read(paused,dtype='float32',always_2d=True)
        protected_end=result.get('aligned_spoken_end_frames')
        if protected_spoken_end_frames is not None:
            external=int(protected_spoken_end_frames)-round(result['tts_leading_silence_removed_seconds']*SR)
            external+=sum(c['added_frames'] for c in result['pause_adjustments'] if c['at_frame']<external)
            external+=result['placement_offset_frames']
            protected_end=max(protected_end or 0,external)
        initial_pause_frames=len(values)
        collision_changes=[]
        if collision_budget_frames is not None:
            values,collision_changes=shorten_collision_pauses(values,collision_budget_frames)
        collision_removed=(initial_pause_frames-len(values))/SR
        result['pause_adjustments'].extend(collision_changes)
        result['pause_removed_seconds']+=collision_removed
        if protected_end is not None:
            protected_end+=sum(c['added_frames'] for c in collision_changes if c['at_frame']<protected_end)
        values,head_removed=reduce_quiet_head_for_collision(values,collision_budget_frames,result['placement_offset_frames'])
        if protected_end is not None:protected_end-=head_removed
        result['collision_leading_silence_removed_seconds']=head_removed/SR
        values,tail_reduction=reduce_unspoken_tail(values,collision_budget_frames,protected_end)
        result.update(unspoken_tail_removed_seconds=tail_reduction['removed_frames']/SR,
            unspoken_tail_fade_seconds=tail_reduction['fade_frames']/SR,
            unspoken_tail_reduction_status=tail_reduction['status'],
            protected_spoken_end_frames=protected_end)
        paused_frames=len(values)
        preserved_quiet_frames=0;speech_chunks=0
        overlap=max(0,paused_frames-collision_budget_frames) if collision_budget_frames is not None else 0
        engine='none';actual=1.;change=0.;region='none';runtime=None
        correction_status='not_needed';fallback_reason=None
        if overlap:
            lo,hi=(0,len(values)) if preserve_outer_edges or preserve_tail else map(int,speech_bounds(values,relative=.02))
            body=values[lo:hi];tail=values[hi:]
            output_frames=collision_budget_frames-lo-len(tail)
            region='speech_body_preserve_leading_and_tail'
            # If untouched silence/tails alone exceed the gap, accelerate the
            # full recording rather than discarding samples or leaving overlap.
            if output_frames<1:
                lo=0;body=values;tail=values[:0]
                output_frames=collision_budget_frames;region='full_audio'
            sf.write(body_file,body,SR,subtype='FLOAT')
            try:
                corrected,preserved_quiet_frames,speech_chunks=stretch_speech_keep_pauses(body,stretched,output_frames)
                if (not isinstance(corrected,np.ndarray) or corrected.ndim!=2 or
                    corrected.shape[1]!=2 or not len(corrected) or not np.isfinite(corrected).all()):
                    raise RuntimeError('Speedy returned invalid audio')
                actual=len(body)/len(corrected);change=(len(corrected)-len(body))/SR
                values=np.concatenate([values[:lo],corrected,tail])
                engine=COLLISION_ENGINE;correction_status='applied'
                region='speech_chunks_preserve_quiet_pauses_and_edges'
            except (RuntimeError,OSError,ValueError,subprocess.TimeoutExpired) as error:
                # The already decoded pause-only audio remains authoritative.
                # Never use a partial worker output or crop a spoken syllable
                # just to satisfy the collision deadline.
                correction_status='skipped_keep_pause_audio'
                fallback_reason=f'{type(error).__name__}: {error}'
        remaining_overlap=max(0,len(values)-collision_budget_frames) if collision_budget_frames is not None else 0
        if not np.isfinite(values).all():raise RuntimeError('Speedy audio validation failed.')
        sf.write(temporary,values,SR,subtype='FLOAT');temporary.replace(destination)
        pauses=result['pause_added_seconds']+result['pause_removed_seconds']
        result.update(timing_method=TIMING_POLICY,output_frames=len(values),tempo=round(actual,6),
            tempo_engine=engine,tempo_engine_version=COLLISION_VERSION if engine!='none' else None,
            tempo_runtime=runtime,
            tempo_quality='google_speedy_nonlinear_sonic_duration_calibrated' if engine!='none' else None,
            tempo_fallback=engine!='none',tempo_duration_change_seconds=change,
            tts_speech_seconds_after=len(values)/SR,remaining_duration_error_seconds=len(values)/SR-result['pause_target_seconds'],
            speed_correction_enabled=True,requested_tempo=round(actual,6),
            tempo_limits=[NATURAL_MIN_TEMPO,NATURAL_MAX_TEMPO],
            tempo_limited=False,
            max_slowdown_duration_factor=MAX_SLOW_DURATION_FACTOR,
            speedup_unlimited=True,slowdown_unlimited=False,overlap_allowed=bool(remaining_overlap),
            collision_budget_frames=collision_budget_frames,
            overlap_before_speedup_frames=overlap,overlap_after_speedup_frames=remaining_overlap,
            speed_correction_status=correction_status,speed_correction_fallback_reason=fallback_reason,
            speed_correction_failure_policy='keep_valid_pause_audio_without_truncation',
            speed_correction_trigger='next_dialogue_overlap' if overlap else 'none',
            speed_correction_region=region,pause_only_output_frames=paused_frames,
            collision_silence_removed_seconds=collision_removed,
            speedup_preserved_quiet_seconds=preserved_quiet_frames/SR,speedup_speech_chunks=speech_chunks,
            quiet_pauses_stretched=False,
            tempo_limit_policy='no_slowdown_speedup_only_to_next_dialogue_start',
            pause_adjustment_share=pauses/max(1e-9,pauses+abs(change)))
        return result
    finally:
        for p in [paused,body_file,stretched,temporary]:p.unlink(missing_ok=True)


def envelope(values):
    count=(len(values)+HOP-1)//HOP
    padded=np.pad(values,((0,count*HOP-len(values)),(0,0)))
    return np.sqrt(np.max(np.mean(padded.reshape(count,HOP,values.shape[1])**2,axis=1),axis=1))


def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def speech_bounds(values,relative=.035):
    """Energy estimate of articulation, distinct from the quieter trailing decay."""
    rms=envelope(values)
    if not len(rms) or rms.max()<1e-7:return 0,len(values)
    threshold,_=measured_energy_gate(rms,relative,values)
    mask=rms>=threshold
    stable=[(a,b) for a,b in runs(mask) if b-a>=3]
    if not stable:return 0,len(values)
    return max(0,stable[0][0]*HOP-round(.025*SR)),min(len(values),stable[-1][1]*HOP+round(.05*SR))


def reference_layout(reference,frames):
    if reference is None:return 0,frames
    values,rate=sf.read(reference,dtype='float32',always_2d=True)
    if rate!=SR or len(values)!=frames:raise RuntimeError('Invalid reference audio duration or sample rate.')
    lo,hi=speech_bounds(values)
    return lo,max(lo+1,hi)


def acoustic_pause_slots(values,minimum_seconds=.04):
    rms=envelope(values)
    if not len(rms) or rms.max()<1e-7:return []
    level=float(np.percentile(rms[rms>rms.max()*.01],90))
    quiet=rms<=max(1e-6,level*.012)
    result=[]
    for a,b in runs(quiet):
        lo,hi=a*HOP,min(len(values),b*HOP)
        duration=(hi-lo)/SR
        if duration<minimum_seconds or lo<.10*SR or len(values)-hi<.10*SR:continue
        if not np.any(~quiet[max(0,a-30):a]) or not np.any(~quiet[b:b+30]):continue
        margin=round(.015*SR)
        quietest=np.max(np.abs(values[lo+margin:hi-margin]),axis=1)
        cut=lo+margin+int(np.argmin(quietest))
        result.append({'lo':lo,'hi':hi,'cut':cut,'weight':duration**2})
    return result


def pause_candidates(values,boundaries=None):
    if not boundaries:return []
    semantic=any(b.get('semantic_approved') for b in boundaries)
    result=[]
    for slot in acoustic_pause_slots(values,.04 if semantic else .08):
        matches=[b for b in boundaries if slot['lo']+.01*SR<=b['time']*SR<=slot['hi']-.01*SR]
        if not matches:continue
        boundary=max(matches,key=lambda b:b.get('confidence',1.))
        result.append({**slot,'weight':boundary.get('weight',slot['weight']),'boundary':boundary})
    for boundary in boundaries:
        if boundary.get('new_pause'):
            cut=boundary['insertion_frame']
            if 0<cut<len(values):result.append({'lo':cut,'hi':cut,'cut':cut,'weight':boundary.get('weight',1.),'boundary':boundary})
    return sorted(result,key=lambda p:p['cut'])


def allocate(amount,weights,capacities=None):
    """Allocate samples proportionally, without an artificial expansion ceiling."""
    counts=[0]*len(weights);remaining=amount
    while remaining:
        active=[i for i in range(len(weights)) if capacities is None or counts[i]<capacities[i]]
        if not active:break
        total=sum(weights[i] for i in active);used=0
        for i in active:
            count=min(remaining-used,max(1,round(remaining*weights[i]/total)))
            if capacities is not None:count=min(count,capacities[i]-counts[i])
            count=int(count)
            counts[i]+=count;used+=count
        remaining-=used
    return counts


def adjust_pauses(values,target_frames,boundaries=None):
    candidates=pause_candidates(values,boundaries)
    deficit=target_frames-len(values)
    if not candidates or not deficit:return values,[]
    shrinking=deficit<0
    # The small quiet guard avoids joining two nonzero edges; all voiced samples stay intact.
    guard=round(.005*SR)
    bounded=all('maximum_pause_seconds' in p['boundary'] for p in candidates)
    if shrinking:
        capacities=[max(0,p['hi']-p['lo']-max(2*guard,round(p['boundary'].get('minimum_pause_seconds',0)*SR))) for p in candidates]
    elif bounded:
        capacities=[max(0,round(p['boundary']['maximum_pause_seconds']*SR)-(p['hi']-p['lo'])) for p in candidates]
    else:capacities=None
    budget=abs(deficit) if capacities is None else min(abs(deficit),sum(capacities))
    if not shrinking and bounded and any(p['boundary'].get('new_pause') for p in candidates):
        # Exhaust established safe pause capacity before adding any pause at a
        # separately verified quiet word junction. Neither capacity is relaxed.
        changes=[0]*len(candidates);remaining=budget
        for new_pause in [False,True]:
            indices=[i for i,p in enumerate(candidates) if bool(p['boundary'].get('new_pause'))==new_pause]
            if not indices or not remaining:continue
            assigned=allocate(min(remaining,sum(capacities[i] for i in indices)),
                [candidates[i]['weight'] for i in indices],[capacities[i] for i in indices])
            for i,amount in zip(indices,assigned):changes[i]=amount
            remaining-=sum(assigned)
    else:changes=allocate(budget,[p['weight'] for p in candidates],capacities)
    return edit_pause_slots(values,candidates,[-n if shrinking else n for n in changes])


def edit_pause_slots(values,candidates,changes):
    guard=round(.005*SR);pieces=[];cursor=0;report=[]
    for pause,amount in zip(candidates,changes):
        if not amount:continue
        shrinking=amount<0;amount=abs(amount)
        if shrinking:
            cut=pause['lo']+guard+(pause['hi']-pause['lo']-2*guard-amount)//2
            if cut<cursor:raise RuntimeError('Pause edit intervals overlap.')
            pieces.append(values[cursor:cut]);cursor=cut+amount
        else:
            cut=pause['cut']
            if cut<cursor:raise RuntimeError('Pause edit intervals overlap.')
            pieces.append(values[cursor:cut])
            inserted=np.zeros((amount,values.shape[1]),dtype=np.float32)
            # Smooth only newly inserted quiet samples; original speech is untouched.
            bridge=min(guard,amount//2)
            if bridge:
                inserted[:bridge]=np.linspace(1,0,bridge,dtype=np.float32)[:,None]*values[cut-1]
                inserted[-bridge:]=np.linspace(0,1,bridge,dtype=np.float32)[:,None]*values[cut]
            pieces.append(inserted);cursor=cut
        report.append({'at_frame':int(cut),'at_seconds':round(cut/SR,3),'added_frames':int(-amount if shrinking else amount),
            'added_seconds':round((-amount if shrinking else amount)/SR,4),
            'between_words':[pause['boundary'].get('left_word',''),pause['boundary'].get('right_word','')],
            'candidate_id':pause['boundary'].get('candidate_id'),
            'kind':'inserted_word_boundary_pause' if pause['boundary'].get('new_pause') else 'existing_pause',
            'junction_kind':pause['boundary'].get('junction_kind'),
            'junction_rms_ratio':pause['boundary'].get('junction_rms_ratio'),
            'minimum_pause_seconds':pause['boundary'].get('minimum_pause_seconds'),
            'maximum_pause_seconds':pause['boundary'].get('maximum_pause_seconds'),
            'existing_pause_seconds':pause['boundary'].get('existing_pause_seconds'),
            'adjacent_spoken_seconds':pause['boundary'].get('adjacent_spoken_seconds'),
            'expansion_policy':pause['boundary'].get('expansion_policy')})
    if not report:return values,[]
    pieces.append(values[cursor:])
    return np.concatenate(pieces),report


def directed_pauses(values,boundaries,plan):
    requested={p['candidate_id']:p['pause_seconds'] for p in plan['pauses']}
    candidates=pause_candidates(values,boundaries);changes=[]
    for pause in candidates:
        boundary=pause['boundary'];current=pause['hi']-pause['lo']
        target=round(requested.get(boundary.get('candidate_id'),current/SR)*SR)
        minimum=round(boundary.get('minimum_pause_seconds',0)*SR)
        if current:minimum=max(minimum,round(.01*SR))
        changes.append(max(minimum,target)-current)
    return edit_pause_slots(values,candidates,changes)


def fit_context(source,decoded,frames,reference=None,speech_window=None):
    command(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-ar',str(SR),'-ac','2','-c:a','pcm_f32le',str(decoded)])
    values,_=sf.read(decoded,dtype='float32',always_2d=True)
    if not len(values) or not np.isfinite(values).all():raise RuntimeError('TTS audio decoding failed.')
    peak=float(np.max(np.abs(values)))
    if peak<1e-6:raise RuntimeError('TTS returned silence.')
    audible=np.flatnonzero(np.max(np.abs(values),axis=1)>max(1e-5,peak*.003))
    margin=round(.025*SR)
    values=values[max(0,int(audible[0])-margin):min(len(values),audible[-1]+margin+1)]
    lead,target_end=reference_layout(reference,frames)
    window_lo,window_hi=speech_window or (0,frames)
    if not 0<=window_lo<window_hi<=frames:raise RuntimeError('Invalid protected inter-sentence pause interval.')
    lead=min(max(lead,window_lo),window_hi-1)
    target_end=min(max(lead+1,target_end),window_hi)
    _,tts_end=speech_bounds(values,relative=.02)
    body,tail=values[:tts_end],values[tts_end:]
    if len(tail)>window_hi-lead-1:body,tail=values,values[:0]
    budget=max(1,min(target_end-lead,window_hi-lead-len(tail)))
    return body,tail,int(lead),int(target_end),int(budget),int(window_lo),int(window_hi)


def fit_audio(source,destination,frames,reference=None,text=None,language='ko',word_alignment=None,speech_window=None,
              original_text=None,planner_key=None,planned_direction=None,analysis=None):
    # Old sample-script planner arguments are ignored; production fitting is fully local.
    decoded=destination.with_name(destination.stem+'-decoded.wav')
    stretched=destination.with_name(destination.stem+'-stretched.wav')
    temporary=destination.with_name(destination.stem+'-partial.wav')
    try:
        context=fit_context(source,decoded,frames,reference,speech_window)
        body,tail,lead,target_end,budget,window_lo,window_hi=context
        expanded=body
        # Use available outer safety margins before changing the speaking rate.
        margin_used=0
        if len(expanded)>budget:
            available=window_hi-window_lo-len(tail)
            remaining=min(len(expanded),available)
            margin_used=max(0,remaining-budget)
            budget=remaining
            lead=min(lead,max(window_lo,window_hi-budget-len(tail)))
        hint={'status':'not_needed'}
        plan={'boundaries':[],'matches':[],'mfa_reviews':[]}
        changes=[]
        if text and len(body)!=budget:
            hint=alignment_hint(body,text,language,destination.with_name(destination.stem+'-mfa.json'))
            if hint['status']=='aligned':
                plan=source_guided_plan(body,reference,frames,(window_lo,window_hi),hint,
                    allow_new_pauses=budget>len(body))
                expanded,changes=adjust_pauses(body,budget,plan['boundaries'])
        adjusted_frames=len(expanded)
        speed=adjusted_frames/budget
        engine='none'
        runtime=None
        if speed!=1:
            engine=TEMPO_ENGINE
            sf.write(decoded,expanded,SR,subtype='FLOAT')
            target=budget
            speed=len(expanded)/target
            expanded=stretch(decoded,stretched,target)
        output=np.zeros((frames,2),dtype=np.float32)
        joined=np.concatenate([expanded,tail])
        if lead+len(joined)>window_hi:raise RuntimeError('Dubbed audio exceeds its assigned interval.')
        output[lead:lead+len(joined)]=joined
        if not np.isfinite(output).all():raise RuntimeError('Invalid TTS duration adjustment result.')
        sf.write(temporary,output,SR,subtype='FLOAT');temporary.replace(destination)
        added=sum(max(0,c['added_frames']) for c in changes)/SR
        removed=sum(max(0,-c['added_frames']) for c in changes)/SR
        tempo_change=(len(expanded)-adjusted_frames)/SR
        return {'fit_version':FIT_VERSION,'tempo':round(speed,6),'tempo_engine':engine,
            'tempo_engine_version':TEMPO_VERSION if engine!='none' else None,
            'tempo_quality':'google_speedy_nonlinear_sonic_duration_calibrated' if engine!='none' else None,
            'tempo_runtime':runtime,
            'source_speech_start_seconds':round(lead/SR,4),'source_speech_end_seconds':round(target_end/SR,4),
            'source_tail_reserve_seconds':round((frames-target_end)/SR,4),
            'tts_speech_seconds_before':round(len(body)/SR,4),'tts_speech_seconds_after':round(len(expanded)/SR,4),
            'pause_added_seconds':round(added,4),'pause_removed_seconds':round(removed,4),'pause_adjustments':changes,
            'source_margin_used_seconds':round(margin_used/SR,4),'tempo_fallback':engine!='none',
            'tempo_duration_change_seconds':round(tempo_change,4),
            'pause_adjustment_share':round((added+removed)/max(1e-9,added+removed+abs(tempo_change)),6),
            'pause_adjustment_enabled':True,
            'word_alignment_status':hint['status'],'pause_boundary_policy':'interword_silence_only_no_consonant_closures',
            'pause_priority':'before_speed_change','pause_distribution':'existing_pause_weighted_adaptive_capacity_then_speedy',
            'mfa_role':'advisory_word_and_phone_boundary_check','mfa_status':hint['status'],
            'mfa_runtime':hint.get('runtime_stats'),'mfa_reason':hint.get('reason'),
            'source_pause_matches':plan['matches'],'pause_boundary_count':len(plan['boundaries']),
            'pause_boundary_reviews':plan['mfa_reviews'],
            'new_pause_count':sum(c['kind']=='inserted_word_boundary_pause' for c in changes),
            'word_junction_reviews':plan.get('word_junction_reviews',[]),
            'pause_planning':'algorithmic_mfa_and_waveform','pause_planning_api_requests':0,
            'speech_window_frames':[int(window_lo),int(window_hi)],
            'unfilled_speech_seconds':round(max(0,budget-len(expanded))/SR,4),
            'tts_tail_seconds':round(len(tail)/SR,4),'timing_method':'legacy_mfa_silence_only_adaptive_existing_pause_expansion_then_speedy'}
    finally:
        decoded.unlink(missing_ok=True);stretched.unlink(missing_ok=True);temporary.unlink(missing_ok=True)
