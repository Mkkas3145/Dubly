"""MFA-checked quiet pauses and word junctions before any speed change."""
from __future__ import annotations
import numpy as np
import soundfile as sf
from core import SR
from mfa_hint import review_pause

VERSION=8
MIN_PAUSE_SECONDS=.18
MAX_PROGRESS_ERROR=.10


def pause_events(values,quiet_ratio=.012,minimum_seconds=MIN_PAUSE_SECONDS):
    if values.ndim==1:values=values[:,None]
    hop=round(.01*SR);count=(len(values)+hop-1)//hop
    if not count:return []
    padded=np.pad(values,((0,count*hop-len(values)),(0,0)))
    levels=np.sqrt(np.max(np.mean(padded.reshape(count,hop,values.shape[1])**2,axis=1),axis=1))
    if levels.max()<1e-7:return []
    level=float(np.percentile(levels[levels>levels.max()*.01],90))
    active=levels>max(1e-6,level*.10);cumulative=np.cumsum(active);total=max(1,int(cumulative[-1]))
    quiet=levels<=max(1e-6,level*quiet_ratio);edges=np.diff(np.r_[False,quiet,False].astype(np.int8));result=[]
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        lo=int(a*hop);hi=min(len(values),int(b*hop))
        if hi-lo<round(minimum_seconds*SR) or lo<round(.1*SR) or len(values)-hi<round(.1*SR):continue
        if np.count_nonzero(active[max(0,a-30):a])<8 or np.count_nonzero(active[b:b+30])<8:continue
        margin=round(.015*SR)
        cut=lo+margin+int(np.argmin(np.max(np.abs(values[lo+margin:hi-margin]),axis=1)))
        result.append({'lo':lo,'hi':hi,'cut':cut,'seconds':(hi-lo)/SR,'progress':float(cumulative[a]/total)})
    return result


def match_events(generated,reference):
    n,m=len(generated),len(reference);cost=np.full((n+1,m+1),np.inf)
    cost[:,0]=np.arange(n+1)*.5;cost[0,:]=np.arange(m+1)*.5;action=np.zeros((n+1,m+1),dtype=np.int8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            left,right=generated[i-1],reference[j-1];distance=abs(left['progress']-right['progress'])
            pairing=distance/MAX_PROGRESS_ERROR+.1*abs(np.log(left['seconds']/right['seconds'])) if distance<=MAX_PROGRESS_ERROR else np.inf
            choices=[cost[i-1,j]+.5,cost[i,j-1]+.5,cost[i-1,j-1]+pairing]
            selected=int(np.argmin(choices));cost[i,j]=choices[selected];action[i,j]=selected
    pairs=[];i,j=n,m
    while i and j:
        selected=action[i,j]
        if selected==0:i-=1
        elif selected==1:j-=1
        else:
            distances=sorted(abs(generated[i-1]['progress']-r['progress']) for r in reference)
            reverse=sorted(abs(reference[j-1]['progress']-r['progress']) for r in generated)
            d=abs(generated[i-1]['progress']-reference[j-1]['progress'])
            if (d<=distances[0]+1e-9 and (len(distances)==1 or distances[1]-distances[0]>=.03)
                and d<=reverse[0]+1e-9 and (len(reverse)==1 or reverse[1]-reverse[0]>=.03)):pairs.append((i-1,j-1))
            i-=1;j-=1
    return list(reversed(pairs))


def word_boundary_insertions(values,hint):
    """Require a real inter-word silence; a quiet consonant is never a pause."""
    if not hint or hint.get('status')!='aligned':return [],[]
    words=hint.get('words',[]);phones=hint.get('phones',[])
    if len(words)<2 or not phones:return [],[]
    if values.ndim==1:values=values[:,None]
    hop=max(1,round(.01*SR));padded=np.pad(values,((0,(-len(values))%hop),(0,0)))
    rms=np.sqrt(np.max(np.mean(padded.reshape(-1,hop,values.shape[1])**2,axis=1),axis=1))
    active=rms[rms>max(1e-7,float(rms.max())*.01)]
    if not len(active):return [],[]
    level=float(np.percentile(active,90));silence={'','sil','sp','<eps>'}
    boundaries=[];reviews=[];half=max(1,round(.003*SR));step=max(1,round(.001*SR))
    for index,(left,right) in enumerate(zip(words,words[1:])):
        review={'left_word':left['label'],'right_word':right['label'],'status':'rejected'}
        if left['end']>right['start']+.005:
            reviews.append({**review,'reason':'overlapping_word_alignment'});continue
        # Restrict silence evidence to the junction between these two words.
        lower=max(left['start']+.02,left['end']-.012,.04)
        upper=min(right['end']-.02,right['start']+.012,len(values)/SR-.04)
        windows=[]
        for phone in phones:
            if phone['label'].casefold() in silence:
                a=max(lower,phone['start']);b=min(upper,phone['end'])
                if b>a:windows.append((a,b,'phone_silence',phone['label']))
        choices=[];silence_agreements=[]
        for a,b,kind,phone in windows:
            # MFA can mislabel a brief closure as silence. Require sustained
            # waveform silence too, and leave both surrounding phones intact.
            if b-a<.04:continue
            lo=max(half,round(a*SR));hi=min(len(values)-half,round(b*SR))
            points=np.arange(lo,hi+1,step,dtype=int)
            if not len(points):continue
            levels=np.array([float(np.sqrt(np.max(np.mean(values[p-half:p+half]**2,axis=0)))) for p in points])
            quiet=levels<=max(1e-6,level*.012)
            # A coarse alignment can label an entire voiced syllable as sil.
            # Finding one tiny trough inside that label does not establish a
            # word gap. Require the claimed silence itself to agree with audio.
            agreement=float(np.mean(quiet))
            silence_agreements.append(agreement)
            if agreement<.65:continue
            edges=np.diff(np.r_[False,quiet,False].astype(np.int8))
            for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                if (end-start)*step<round(.02*SR):continue
                # Pick a low-energy *region*, then the quietest stereo sample inside it.
                best=start+int(np.argmin(levels[start:end]));center=int(points[best])
                sample_lo=max(lo,int(points[start]),center-step)
                sample_hi=min(hi+1,int(points[end-1])+1,center+step+1)
                cut=sample_lo+int(np.argmin(np.max(np.abs(values[sample_lo:sample_hi]),axis=1)))
                choices.append((float(levels[best]/max(level,1e-8)),cut,kind,phone))
        if not choices:
            reason='aligned_silence_contains_speech' if silence_agreements and max(silence_agreements)<.65 else 'no_sustained_interword_silence'
            reviews.append({**review,'reason':reason,
                'silence_waveform_agreement':max(silence_agreements,default=0.)});continue
        ratio,cut,kind,phone=min(choices,key=lambda x:(x[0],abs(x[1]/SR-right['start'])))
        if boundaries and cut-boundaries[-1]['insertion_frame']<round(.04*SR):
            reviews.append({**review,'reason':'junctions_too_close'});continue
        boundaries.append({'time':cut/SR,'insertion_frame':cut,'new_pause':True,'weight':1.,'confidence':1.,
            'left_word':left['label'],'right_word':right['label'],'candidate_id':f'word-junction-{index}',
            'minimum_pause_seconds':0.,'mfa_review':'word_boundary_and_quiet_phone',
            'junction_kind':kind,'junction_phone':phone,'junction_rms_ratio':ratio})
        reviews.append({**review,'status':'accepted','time':cut/SR,'kind':kind,'phone':phone,'rms_ratio':ratio})
    return boundaries,reviews


def expansion_budgets(values,boundaries):
    """Use existing pacing and adjacent spoken phrases, without requiring a source match.

    A pause may grow by half its present length, or up to half the shorter
    adjacent spoken phrase. A longer matched source pause is additional evidence.
    These are relative acoustic heuristics, not linguistic/semantic decisions.
    """
    if not boundaries:return
    if values.ndim==1:values=values[:,None]
    hop=round(.01*SR)
    padded=np.pad(values,((0,(-len(values))%hop),(0,0)))
    rms=np.sqrt(np.max(np.mean(padded.reshape(-1,hop,values.shape[1])**2,axis=1),axis=1))
    active=rms[rms>max(1e-7,float(rms.max())*.01)]
    threshold=max(1e-6,float(np.percentile(active,90))*.10) if len(active) else 1e-6
    voiced=np.repeat(rms>threshold,hop)[:len(values)]
    cumulative=np.r_[0,np.cumsum(voiced,dtype=np.int64)]
    ordered=sorted(boundaries,key=lambda b:b['pause_lo_frame'])
    for i,boundary in enumerate(ordered):
        lo,hi=boundary['pause_lo_frame'],boundary['pause_hi_frame']
        left=ordered[i-1]['pause_hi_frame'] if i else 0
        right=ordered[i+1]['pause_lo_frame'] if i+1<len(ordered) else len(values)
        shorter=min(cumulative[lo]-cumulative[min(left,lo)],
                    cumulative[max(right,hi)]-cumulative[hi])/SR
        current=(hi-lo)/SR
        maximum=max(current*1.5,shorter*.5,boundary.get('source_pause_seconds') or 0.)
        allowance=max(0.,maximum-current)
        # Point insertions already retain the detected short silence in values.
        boundary['maximum_pause_seconds']=allowance if boundary.get('new_pause') else maximum
        boundary['existing_pause_seconds']=current
        boundary['adjacent_spoken_seconds']=float(shorter)
        boundary['weight']=max(current,1/SR)
        boundary['expansion_policy']='existing_pause_and_adjacent_speech_with_optional_source_hint'


def source_guided_plan(values,reference=None,frames=None,speech_window=None,hint=None,allow_new_pauses=False):
    generated=pause_events(values,minimum_seconds=.08 if hint and hint.get('status')=='aligned' else MIN_PAUSE_SECONDS)
    source=[]
    if reference is not None:
        original,rate=sf.read(reference,dtype='float32',always_2d=True)
        if rate!=SR or frames is not None and len(original)!=frames:raise RuntimeError('Invalid source duration or sample rate for pause comparison.')
        lo,hi=speech_window or (0,len(original));source=pause_events(original[lo:hi],quiet_ratio=.05)
        for row in source:
            for key in ['lo','hi','cut']:row[key]+=lo
    matches=dict(match_events(generated,source));boundaries=[];report=[];reviews=[]
    for index,pause in enumerate(generated):
        review=review_pause(pause,hint);reviews.append({'tts_seconds':[pause['lo']/SR,pause['hi']/SR],'mfa':review})
        if review in ('phoneme_conflict','within_word') or pause['seconds']<MIN_PAUSE_SECONDS and review!='corroborated':continue
        paired=source[matches[index]] if index in matches else None
        boundaries.append({'time':pause['cut']/SR,'confidence':1.,'weight':1.,
            'minimum_pause_seconds':min(pause['seconds'],.08 if review=='corroborated' else .12),
            'source_pause_seconds':paired['seconds'] if paired else None,'acoustic_approved':True,
            'source_matched':paired is not None,'mfa_review':review,'left_word':'','right_word':'',
            'pause_lo_frame':pause['lo'],'pause_hi_frame':pause['hi']})
        if paired:report.append({'tts_seconds':[pause['lo']/SR,pause['hi']/SR],
            'source_seconds':[paired['lo']/SR,paired['hi']/SR],'progress_difference':abs(pause['progress']-paired['progress'])})
    junction_reviews=[]
    # Collect additional proven quiet inter-word junctions even when established
    # pauses exist. The allocator uses them only after established capacity.
    if allow_new_pauses:
        additional,junction_reviews=word_boundary_insertions(values,hint)
        short_pauses=pause_events(values,minimum_seconds=.04)
        short_matches=dict(match_events(short_pauses,source))
        for boundary in additional:
            index=next((i for i,p in enumerate(short_pauses)
                if p['lo']<=boundary['insertion_frame']<p['hi']),None)
            paired=source[short_matches[index]] if index in short_matches else None
            lo=short_pauses[index]['lo'] if index is not None else boundary['insertion_frame']
            hi=short_pauses[index]['hi'] if index is not None else lo
            boundary.update(pause_lo_frame=lo,pause_hi_frame=hi,
                source_pause_seconds=paired['seconds'] if paired else None,source_matched=paired is not None)
        boundaries.extend(b for b in additional if b['pause_hi_frame']>b['pause_lo_frame'] and not any(
            max(b['pause_lo_frame'],old['pause_lo_frame'])<min(b['pause_hi_frame'],old['pause_hi_frame'])
            for old in boundaries))
    expansion_budgets(values,boundaries)
    return {'version':VERSION,'boundaries':boundaries,'matches':report,'mfa_reviews':reviews,
            'word_junction_reviews':junction_reviews,
            'tts_pause_count':len(generated),'source_pause_count':len(source)}
