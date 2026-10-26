"""Clean shared reference/replacement edges and preserve the excluded source tail."""
from __future__ import annotations
import numpy as np

VERSION=6


def sustained_tail_event(values,sr,lo,word_end,protected,points,rms,width):
    """Separate a new sustained sound after a completed, near-quiet voice decay."""
    word_hi=min(len(values),round(word_end*sr))
    if word_hi<=lo:return None
    hop=max(1,round(sr*.01))
    speech=values[lo:word_hi]
    speech=np.pad(speech,((0,(-len(speech))%hop),(0,0)))
    levels=np.sqrt(np.mean(speech.reshape(-1,hop,values.shape[1])**2,axis=1)).max(axis=1)
    level=float(np.percentile(levels,90))
    if level<1e-6:return None
    step=points[1]-points[0] if len(points)>1 else width
    quiet=rms<=level*.05
    edges=np.diff(np.r_[False,quiet,False].astype(np.int8))
    channel=int(np.argmax(np.mean(values[lo:word_hi]**2,axis=0)))
    def spectrum(part):
        power=np.abs(np.fft.rfft(part*np.hanning(len(part)),n=4096))**2
        return np.array([band.sum() for band in np.array_split(power,64)])
    old=spectrum(values[max(lo,word_hi-round(.06*sr)):word_hi,channel])
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        a=max(a,int(np.searchsorted(points,protected)))
        if b<=a or (b-a)*step<round(.07*sr):continue
        baseline=max(1e-7,float(np.median(rms[a:b])))
        active=rms[b:]>max(level*.1,baseline*4)
        turns=np.diff(np.r_[False,active,False].astype(np.int8))
        events=[(b+x,b+y) for x,y in zip(np.flatnonzero(turns==1),np.flatnonzero(turns==-1))
                if (y-x)*step>=round(.06*sr)]
        if not events:continue
        attack,stop=events[0]
        point=points[attack]
        new=spectrum(values[point:min(len(values),point+round(.06*sr)),channel])
        distance=1-float(np.dot(old,new)/max(1e-20,np.linalg.norm(old)*np.linalg.norm(new)))
        # A renewed rise alone can be an echo of the same voice. Require a
        # spectral change as well as the sustained low-energy separation.
        if distance<.25:continue
        valley=a+int(np.argmin(rms[a:attack]))
        return {'onset_frame':int(points[valley]),'onset_seconds':points[valley]/sr,
            'peak_ratio':float(np.max(rms[attack:stop])/baseline),'spectral_distance':distance,
            'protected_word_end_seconds':word_end,'attack_frame':int(point),'attack_seconds':point/sr,
            'kind':'sustained_event_after_voice_decay'}
    return None


def tail_transient(values,sr,start,end,word_end):
    """Find a new short attack after an established post-word voice decay."""
    if values.ndim==1:values=values[:,None]
    lo=max(0,round(start*sr));hi=min(len(values),round(end*sr))
    protected=max(lo,round((word_end+.04)*sr))
    if hi-protected<round(.015*sr):return None
    hop=max(1,round(.002*sr));width=max(hop,round(.008*sr))
    origin=max(lo,protected-round(.07*sr))
    analysis_hi=min(len(values),hi+round(.06*sr))
    points=np.arange(origin,analysis_hi-width+1,hop,dtype=int)
    if len(points)<5:return None
    rms=np.array([np.sqrt(np.mean(values[p:p+width]**2,axis=0)).max() for p in points])
    sustained=sustained_tail_event(values,sr,lo,word_end,protected,points,rms,width)
    # ASR may end a word before its final consonant. A spectral change or
    # short renewed attack is not enough to call that consonant an effect.
    # Require a sustained low-energy decay relative to the utterance first.
    speech=values[lo:min(len(values),round(word_end*sr))]
    frame=max(1,round(.01*sr))
    if not len(speech):return sustained
    speech=np.pad(speech,((0,(-len(speech))%frame),(0,0)))
    speech_levels=np.sqrt(np.mean(speech.reshape(-1,frame,values.shape[1])**2,axis=1)).max(axis=1)
    speech_level=float(np.percentile(speech_levels,90))
    from scipy.ndimage import uniform_filter1d
    contour=uniform_filter1d(rms,size=3,mode='nearest')
    channel=int(np.argmax(np.mean(values[lo:hi]**2,axis=0)))
    for index,p in enumerate(points):
        if p<protected or p>=hi:continue
        past=(points>=p-round(.05*sr))&(points<p-round(.008*sr))
        previous=rms[past]
        if len(previous)<6:continue
        baseline=float(np.median(previous))
        if baseline<1e-6 or rms[index]<baseline*1.85:continue
        half=len(previous)//2
        if np.median(previous[-half:])>np.median(previous[:half])*1.1:continue
        following=(points>p)&(points<=p+round(.045*sr))
        if not np.any(following) or np.min(rms[following])>baseline*1.35:continue
        prior_start=max(lo,p-round(.045*sr));prior_end=max(prior_start+1,p-round(.010*sr))
        prior=values[prior_start:prior_end,channel]
        attack=values[p:p+width,channel]
        old=np.abs(np.fft.rfft(prior*np.hanning(len(prior)),n=1024))**2
        new=np.abs(np.fft.rfft(attack*np.hanning(len(attack)),n=1024))**2
        old=np.array([x.sum() for x in np.array_split(old,32)])
        new=np.array([x.sum() for x in np.array_split(new,32)])
        distance=1-float(np.dot(old,new)/max(1e-20,np.linalg.norm(old)*np.linalg.norm(new)))
        if distance<.16 and rms[index]<baseline*2.3:continue
        attack_index=index
        while attack_index>0 and points[attack_index-1]>=max(protected,p-round(.035*sr)):
            if contour[attack_index-1]>contour[attack_index]*1.04:break
            attack_index-=1
        onset=max(protected,points[attack_index]-round(.002*sr))
        prior_mask=(points>=onset-round(.08*sr))&(points<onset)
        quiet=rms[prior_mask]<=max(1e-7,speech_level*.12)
        changes=np.diff(np.r_[False,quiet,False].astype(np.int8))
        runs=zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))
        if not any((b-a)*hop>=round(.03*sr) for a,b in runs):continue
        event={'onset_frame':int(onset),'onset_seconds':onset/sr,'peak_ratio':float(rms[index]/baseline),
            'spectral_distance':distance,'protected_word_end_seconds':word_end,
            'attack_frame':int(p),'attack_seconds':p/sr}
        return sustained if sustained is not None and sustained['onset_frame']<onset else event
    return sustained


def reference_windows(values,sr,measurements):
    windows=[]
    for row in measurements:
        event=tail_transient(values,sr,row['start'],row['end'],row['aligned_end'])
        start=row['start'];end=row['end'] if event is None else min(row['end'],event['onset_seconds'])
        lo=max(0,round(start*sr));hi=min(len(values),round(end*sr))
        if not 0<=lo<hi<=len(values):
            raise RuntimeError('Measured TTS reference interval exceeds audio duration.')
        start=lo/sr;end=hi/sr
        # Guards only affect non-lexical edge margins, never the word itself.
        fade_in=min(round(.002*sr),max(0,round((row['aligned_start']-start-.02)*sr)))
        fade_out=min(round(.002*sr),max(0,round((end-row['aligned_end']-.02)*sr)))
        preserved=[]
        measured_end=min(len(values),round(row['end']*sr))/sr
        if end<measured_end:
            # Restore source audio smoothly in the quiet lead-in before the new
            # attack. Protect the attack itself, not the earlier crop margin.
            protected_start=min(measured_end,max(end,event['attack_seconds']))
            if protected_start<measured_end:
                preserved.append({'start':protected_start,'end':measured_end,'reason':'reference_trimmed_tail'})
        windows.append({'start':start,'end':end,'start_frame':lo,'end_frame':hi,'duration':(hi-lo)/sr,
            'fade_in_frames':fade_in,'fade_out_frames':fade_out,'tail_transient':event,
            'boundary_policy':'shared_clean_interval','preserved_intervals':preserved,
            'tail_cleanup_policy':'require_sustained_low_energy_decay_before_new_attack',
            'measured_start_before_cleanup':row['start'],'measured_end_before_cleanup':row['end'],
            'replacement_start':start,'replacement_end':end})
    return windows


def edge_guard(values,fade_in=0,fade_out=0):
    result=values.copy()
    for count,reverse in [(min(len(result),fade_in),False),(min(len(result),fade_out),True)]:
        if count<2:continue
        curve=np.linspace(0,1,count,dtype=np.float32)
        if result.ndim>1:curve=curve[:,None]
        if reverse:result[-count:]*=curve[::-1]
        else:result[:count]*=curve
    return result
