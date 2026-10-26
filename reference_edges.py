"""Source-text evidence selects voices; acoustic evidence selects clip edges."""
from __future__ import annotations
import numpy as np


class AcousticMeasurementError(RuntimeError):
    def __init__(self,index,reason,message):
        super().__init__(message);self.index=index;self.reason=reason


def validated_boundaries(values,sr,aligned,progress=lambda *args:None):
    """Keep strict measurements; isolate unsupported lines with diagnostics."""
    rows=list(aligned);failures=[]
    hop=max(1,round(sr*.01));x=values[:,None] if values.ndim==1 else values
    padded=np.pad(x,((0,(-len(x))%hop),(0,0)))
    levels=np.sqrt(np.mean(padded.reshape(-1,hop,x.shape[1])**2,axis=1)).max(axis=1)
    while rows:
        try:return rows,acoustic_boundaries(values,sr,rows,_levels=levels),failures
        except AcousticMeasurementError as error:
            row=rows[error.index];source=row.get('source_index',error.index)
            failures.append(dict(source_index=source,translated_text=row.get('translated_text',''),
                original_text=row.get('recognized_text',''),reason=error.reason,error=str(error),
                measured_words=row.get('words',[]),source_word_floor=row.get('source_word_floor'),
                source_word_ceiling=row.get('source_word_ceiling'),action='preserve_original_audio'))
            progress(0,f'대사 {source+1} 측정 재검토 · 근거 부족 시 원음 유지')
            rows=[r for i,r in enumerate(rows) if r.get('source_index',i)!=source]
    return [],[],failures

def spans(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))

def acoustic_boundaries(values,sr,aligned,_levels=None):
    if values.ndim==1:values=values[:,None]
    hop=max(1,round(sr*.01));dt=hop/sr;duration=len(values)/sr
    if _levels is None:
        padded=np.pad(values,((0,(-len(values))%hop),(0,0)))
        levels=np.sqrt(np.mean(padded.reshape(-1,hop,values.shape[1])**2,axis=1)).max(axis=1)
    else:levels=_levels
    def valley(lower,upper):
        a=max(0,int(np.ceil(lower/dt)));b=min(len(levels),int(np.ceil(upper/dt)))
        if b<=a:raise RuntimeError('Could not identify the acoustic boundary between connected utterances.')
        frame=a+int(np.argmin(levels[a:b]))
        lo=max(round(lower*sr),round((frame*dt-.005)*sr));hi=min(round(upper*sr),len(values),round((frame*dt+.015)*sr))
        from scipy.ndimage import uniform_filter1d
        power=np.max(values[lo:hi].astype(np.float64)**2,axis=1)
        size=max(1,round(sr*.005));guard=size//2
        if len(power)>2*guard:
            smooth=uniform_filter1d(power,size=size,mode='nearest')
            return (lo+guard+int(np.argmin(smooth[guard:-guard])))/sr
        return frame*dt
    anchors=[]
    for i,row in enumerate(aligned):
        words=[w for w in row['words'] if w['end']>w['start']]
        if not words:raise AcousticMeasurementError(i,'missing_word_timestamps',f'Scribe word timestamps are empty for dialogue {i+1}.')
        anchors.append({'first':dict(words[0]),'last':dict(words[-1]),
            'method':row.get('backend','provided_word_timestamps'),'confidence':row.get('confidence',1.)})
    cores=[]
    for i,anchor in enumerate(anchors):
        first,last=anchor['first'],anchor['last']
        search_start=max(0,first['start']-.32)
        search_end=min(duration,first['end']+.15)
        search_start=max(search_start,aligned[i].get('source_word_floor',0.))
        if cores:
            previous=cores[-1]
            a=max(0,int(np.ceil(previous['core_end']/dt)));b=min(len(levels),int(np.ceil(search_end/dt)))
            quiet=[(a+x,a+y) for x,y in spans(levels[a:b]<=max(1e-6,previous['level']*.012)) if (y-x)*dt>=.07]
            lower=quiet[0][0]*dt if quiet else previous['core_end']
            search_start=max(search_start,lower)
        a=max(0,int(np.ceil(search_start/dt-1e-7)));b=min(len(levels),int(np.ceil(search_end/dt)))
        body=levels[max(0,int(first['start']/dt)):min(len(levels),int(np.ceil(last['end']/dt)))]
        if not len(body):raise AcousticMeasurementError(i,'empty_voice_window',f'Vocalization interval is empty for dialogue {i+1}.')
        voiced=body[body>max(1e-7,float(body.max())*.01)]
        if not len(voiced):raise AcousticMeasurementError(i,'no_measured_voice',f'Could not detect vocalization at dialogue {i+1}.')
        level=max(float(np.percentile(voiced,90)),float(body.max())*.6)
        active=[(a+x,a+y) for x,y in spans(levels[a:b]>max(1e-6,level*.10)) if (y-x)*dt>=.02]
        start_recovery='not_needed'
        if not active and 'provider_start' in first and 'provider_end' in first:
            # A global ASR offset can move a short first word into silence.
            # Re-measure near its original provider anchor, bounded by the
            # preceding voice and following word evidence; never reuse a cut.
            raw_start,raw_end=first['provider_start'],first['provider_end']
            # The first quiet gap can be *after this short syllable* when
            # the previous release runs into it. Keep the preceding core as
            # the hard limit, not that already-disqualified quiet-gap guess.
            lower=max(0,cores[-1]['core_end'] if cores else 0,raw_start-.08)
            upper=min(duration,raw_end+.15)
            if i+1<len(anchors):
                following=anchors[i+1]['first']
                upper=min(upper,min(following['start'],following.get('provider_start',following['start'])))
            ra=max(0,int(np.ceil(lower/dt-1e-7)))
            rb=min(len(levels),int(np.ceil(upper/dt)))
            local=levels[ra:rb]
            if len(local):
                def candidates(threshold):
                    return [(ra+x,ra+y) for x,y in spans(local>threshold)
                        if (y-x)*dt>=.02 and (ra+x)*dt<raw_end+.10
                        and (ra+y)*dt>raw_start-.08]
                recovered=candidates(max(1e-6,level*.10))
                method='provider_anchor_with_acoustic_evidence'
                if not recovered:
                    floor=float(np.percentile(local,20));peak=float(np.percentile(local,95))
                    recovered=candidates(max(1e-6,level*.015,floor*4,peak*.1))
                    method='local_first_word_with_acoustic_evidence'
                if recovered:
                    active=recovered;a,b=ra,rb
                    first={**first,'start':raw_start,'end':raw_end}
                    start_recovery=method
        if not active:raise AcousticMeasurementError(i,'unmeasured_onset',f'Could not measure vocalization onset for dialogue {i+1}. Original timestamps were not reused.')
        selected=-1 if first.get('probability',1.)<.15 and first['end']-first['start']>.7 else 0
        if first.get('timing_source') in ('qwen_source_text_alignment','scribe_translation_match'):
            # A validated leading word is a positive audio/text anchor. Do not
            # pick an earlier grunt merely because it lies in the backward
            # energy-search margin. Preserve the connected attack of this word.
            overlapping=[j for j,(x,y) in enumerate(active)
                if x*dt<first['end']+.02 and y*dt>first['start']-.04]
            if overlapping:
                selected=min(overlapping,key=lambda j:abs(active[j][0]*dt-first['start']))
        onset=active[selected][0]*dt
        # Trace a soft consonant/breath backward from the strong attack; don't
        # let a peak-relative detector discard the start of a quiet syllable.
        before=levels[a:active[selected][0]]
        quiet_before=[(a+x,a+y) for x,y in spans(before<=max(1e-6,level*.03)) if (y-x)*dt>=.03]
        if quiet_before:onset=min(onset,quiet_before[-1][1]*dt)
        elif active[selected][0]<=a+1:
            # A late word timestamp can put the whole search window inside
            # an already sounding syllable. Recover its leading attack from
            # the closest sustained quiet, without crossing the prior core.
            lower=max(0,cores[-1]['core_end'] if cores else 0,onset-.5,aligned[i].get('source_word_floor',0.))
            back=max(0,int(np.ceil(lower/dt)))
            quiet=[(back+x,back+y) for x,y in spans(levels[back:a]<=max(1e-6,level*.03))
                if (y-x)*dt>=.03]
            if quiet:
                onset=min(onset,quiet[-1][1]*dt)
                start_recovery='search_edge_continuation_to_sustained_quiet'
        upper=min(duration,max(last['end']+.02,anchors[i+1]['first']['start']) if i+1<len(anchors) else duration,
            aligned[i].get('source_word_ceiling',duration))
        a=max(0,int(max(onset,last['start']-.08)/dt))
        b=min(len(levels),int(np.ceil(min(upper,last['end']+.10)/dt)))
        ending=[(a+x,a+y) for x,y in spans(levels[a:b]>max(1e-6,level*.10)) if (y-x)*dt>=.02]
        end_recovery='not_needed'
        if not ending and 'provider_start' in last and 'provider_end' in last:
            # A recording-wide offset is not necessarily valid at every word.
            # Search the original ASR anchor as well, but still require an
            # acoustic event at that word; never substitute an LLM boundary.
            raw_start,raw_end=last['provider_start'],last['provider_end']
            a=max(0,int(max(onset,min(last['start'],raw_start)-.08)/dt))
            b=min(len(levels),int(np.ceil(min(upper,max(last['end'],raw_end)+.10)/dt)))
            alternatives=[(a+x,a+y) for x,y in spans(levels[a:b]>max(1e-6,level*.10))
                if (y-x)*dt>=.02 and (a+x)*dt<raw_end+.10 and (a+y)*dt>raw_start-.08]
            if alternatives:
                ending=alternatives
                # Tail cleanup must protect the recovered final word too,
                # instead of mistaking its later attack for an unrelated sound.
                last={**last,'start':raw_start,'end':raw_end}
                end_recovery='provider_anchor_with_acoustic_evidence'
        if not ending:
            # A quiet final syllable may be far below the phrase's loudest
            # vowel. Measure its local contrast around ASR evidence instead
            # of requiring the entire phrase's peak-relative threshold.
            raw_start=last.get('provider_start',last['start'])
            raw_end=last.get('provider_end',last['end'])
            a=max(0,int(max(onset,min(last['start'],raw_start)-.08)/dt))
            b=min(len(levels),int(np.ceil(min(upper,max(last['end'],raw_end)+.10)/dt)))
            local=levels[a:b]
            if len(local):
                floor=float(np.percentile(local,20))
                peak=float(np.percentile(local,95))
                threshold=max(1e-6,level*.003,floor*2,peak*.05)
                alternatives=[(a+x,a+y) for x,y in spans(local>threshold)
                    if (y-x)*dt>=.02 and (a+x)*dt<max(last['end'],raw_end)+.10
                    and (a+y)*dt>min(last['start'],raw_start)]
                if alternatives:
                    ending=alternatives
                    last={**last,'start':min(last['start'],raw_start),'end':max(last['end'],raw_end)}
                    end_recovery='local_final_word_with_acoustic_evidence'
        if not ending:raise AcousticMeasurementError(i,'unmeasured_last_vocalization',f'Could not measure the final vocalization for dialogue {i+1}.')
        release=min(upper,ending[-1][1]*dt)
        cores.append({'index':i,'onset':onset,'release':release,'level':level,'core_start':onset,
            'core_end':max(release,min(last['end'],release+.08)),
            'aligned_start':first['start'],'aligned_end':last['end'],
            'first_word_end':first['end'],'last_word_start':last['start'],
            'start_anchor_recovery':start_recovery,
            'end_anchor_recovery':end_recovery,
            'source_word_floor':aligned[i].get('source_word_floor',0.),
            'source_word_ceiling':aligned[i].get('source_word_ceiling',duration),
            'alignment_backend':anchor['method'],'alignment_confidence':anchor['confidence']})
    # Resolve touching vocal cores jointly, using the *word* anchors rather than
    # a neighbor onset which might actually be this speaker's reverberation.
    for left,right in zip(cores,cores[1:]):
        if right['onset']-left['core_end']>.06:continue
        lower=max(left['last_word_start'],min(left['aligned_end'],right['aligned_start'])-.08)
        upper=min(right['first_word_end'],max(left['aligned_end'],right['aligned_start'])+.08)
        lower=max(lower,left['onset']+.01);upper=min(upper,right['core_end']-.01,duration)
        split=valley(lower,upper)
        left['core_end']=right['core_start']=right['onset']=split
    reports=[]
    for i,core in enumerate(cores):
        next_onset=cores[i+1]['onset'] if i+1<len(cores) else duration
        previous_end=cores[i-1]['core_end'] if i else 0
        start=max(0,previous_end,core['onset']-.04,core['source_word_floor'])
        next_onset=min(next_onset,core['source_word_ceiling'])
        a=min(len(levels),int(np.ceil(core['core_end']/dt)));b=min(len(levels),int(np.ceil(next_onset/dt)))
        quiet=[(a+x,a+y) for x,y in spans(levels[a:b]<=max(1e-6,core['level']*.012)) if (y-x)*dt>=.07]
        if quiet:
            decay=quiet[0][0]*dt;end=min(next_onset,max(core['core_end'],decay)+.04)
            method='sustained_quiet_after_voice_decay'
        else:
            if i+1==len(cores):decay=duration
            elif b>a:
                frame=a+int(np.argmin(levels[a:b]));lo=max(0,round((frame*dt-.005)*sr));hi=min(len(values),round((frame*dt+.015)*sr))
                from scipy.ndimage import uniform_filter1d
                power=np.max(values[lo:hi].astype(np.float64)**2,axis=1)
                size=max(1,round(sr*.005));smooth=uniform_filter1d(power,size=size,mode='nearest');guard=size//2
                cut=lo+guard+int(np.argmin(smooth[guard:-guard])) if len(smooth)>2*guard else frame*hop
                decay=max(core['core_end'],min(next_onset,cut/sr))
            else:decay=core['core_end']
            end=min(next_onset,decay)
            method='joint_neighbor_acoustic_valley' if i+1<len(cores) else 'recording_end'
        reports.append({**core,'start':start,'end':min(duration,end),'decay_end_seconds':decay,'end_method':method,
            'gemini_timestamp_fallback':False,'review_flags':[]})
    for left,right in zip(reports,reports[1:]):
        if left['end']>right['start']:
            low,high=left['core_end'],right['core_start']
            split=float(np.clip((left['end']+right['start'])/2,min(low,high),max(low,high)))
            left['end']=right['start']=split
    for row in reports:
        lo=max(0,round(row['start']*sr));hi=min(len(values),round(row['end']*sr))
        if hi<=lo:raise RuntimeError('Invalid measured reference interval duration.')
        row.update(start=lo/sr,end=hi/sr,start_frame=lo,end_frame=hi,duration=(hi-lo)/sr)
    if any(a['end_frame']>b['start_frame'] for a,b in zip(reports,reports[1:])):raise RuntimeError('Measured reference intervals overlap.')
    return reports
