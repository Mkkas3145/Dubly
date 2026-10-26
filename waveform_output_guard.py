"""Conservative acoustic replay removal. No recognition or semantic model calls.

Only edge utterances acoustically explained by a reference may be removed.
This cannot tell Japanese from Korean, or certify a remaining utterance's text.
"""
from functools import lru_cache
from pathlib import Path
from difflib import SequenceMatcher
import hashlib,json,math,threading
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.fft import dct
from numba import njit
from core import SR,atomic_json
from phonetic_output_guard import POLICY as PHONETIC_POLICY
from leading_replay_guard import POLICY as LEADING_REPLAY_POLICY

POLICY={'version':4,'method':'cepstral_and_energy_dtw_reference_replay',
    'batch_size':1,'asr_requests':0,'sample_rate':16000,'analysis_hop_seconds':.02,
    'minimum_utterance_seconds':.35,'minimum_similarity':.94,
    'minimum_lower_quartile_similarity':.90,'minimum_temporal_specificity':.04,
    'minimum_energy_correlation':.80,'duration_ratio':[.65,1.55],
    'quiet_split_seconds':.18,'crop':'verified_reference_edges_with_distinct_remaining_utterance',
    'uncertain':'preserve_full_generated_audio','all_similar':'preserve_full_generated_audio',
    'text_use':'protect_input_phonetic_equivalents_only_never_locate_or_authorize_cuts',
    'content_verified':False,'regeneration':False,
    'edge_search':'contiguous_utterance_spans_and_complete_reference',
    'similarity_thresholds_relaxed':False,'alignment_edge_tolerance_seconds':.08,
    'minimum_aligned_energy_coverage':.90,'phonetic_replay':PHONETIC_POLICY,'leading_replay':LEADING_REPLAY_POLICY}
RATE=16000;HOP=320;WINDOW=512
_FEATURE_LOCK=threading.Lock()

def mono(values):
    result=values.mean(axis=1)
    if np.sqrt(np.mean(result**2))<.25*np.sqrt(np.mean(values**2)):
        result=values[:,int(np.argmax(np.mean(values**2,axis=0)))]
    return result

@lru_cache(maxsize=1)
def mel_filters():
    freq=np.fft.rfftfreq(WINDOW,1/RATE)
    points=700*(10**(np.linspace(2595*np.log10(1+80/700),2595*np.log10(1+7600/700),42)/2595)-1)
    filters=[]
    for a,b,c in zip(points,points[1:],points[2:]):
        filters.append(np.maximum(0,np.minimum((freq-a)/(b-a),(c-freq)/(c-b))))
    return np.asarray(filters,dtype=np.float32)

def features(values,rate):
    if rate!=RATE:
        divisor=math.gcd(rate,RATE);values=resample_poly(values,RATE//divisor,rate//divisor)
    values=np.asarray(values,dtype=np.float32)
    padded=np.pad(values,(WINDOW//2,WINDOW//2))
    frames=np.lib.stride_tricks.sliding_window_view(padded,WINDOW)[::HOP]
    rms=np.sqrt(np.mean(frames**2,axis=1))
    power=np.abs(np.fft.rfft(frames*np.hanning(WINDOW),axis=1))**2
    bands=np.einsum('tf,mf->tm',power,mel_filters(),optimize=False)
    cepstra=dct(np.log(np.maximum(bands,1e-9)),type=2,norm='ortho',axis=1)[:,1:21]
    delta=np.gradient(cepstra,axis=0) if len(cepstra)>1 else np.zeros_like(cepstra)
    vector=np.concatenate([cepstra,2*delta],axis=1).astype(np.float32)
    vector/=np.maximum(np.linalg.norm(vector,axis=1,keepdims=True),1e-8)
    return vector,rms

def voice_groups(values,rate):
    from timing import envelope,measured_energy_gate,runs,HOP as ENERGY_HOP
    if not len(values) or not np.isfinite(values).all():return []
    levels=envelope(values);gate,_=measured_energy_gate(levels,.02,values)
    active=[(a*ENERGY_HOP,min(len(values),b*ENERGY_HOP)) for a,b in runs(levels>=gate) if b-a>=2]
    merged=[]
    for a,b in active:
        if merged and a-merged[-1][1]<round(POLICY['quiet_split_seconds']*rate):merged[-1][1]=b
        else:merged.append([a,b])
    # Preserve weak consonants/reverberation around the measured voiced core.
    return [(max(0,a-round(.015*rate)),min(len(values),b+round(.025*rate))) for a,b in merged]

@lru_cache(maxsize=256)
def _reference(path,size,mtime):
    values,rate=sf.read(path,dtype='float32',always_2d=True)
    if rate!=SR:raise ValueError('reference sample rate')
    result=[]
    groups=voice_groups(values,rate)
    spans=list(groups)
    # The same speech may have different silence segmentation in TTS. Keep
    # the complete reference as another hypothesis, including internal pauses.
    if len(groups)>1:spans.append((groups[0][0],groups[-1][1]))
    for a,b in spans:
        if (b-a)/rate<POLICY['minimum_utterance_seconds']:continue
        vector,rms=features(mono(values[a:b]),rate)
        result.append((a/rate,b/rate,vector,rms))
    return result

@njit(cache=True,nogil=True)
def _path(distance,edge_frames=0):
    """Symmetric DTW steps (1,1), (1,2), (2,1); no arbitrarily long repeats."""
    m,n=distance.shape
    cost=np.full((m+1,n+1),np.inf);trace=np.zeros((m+1,n+1),np.int8);cost[0,0]=0.
    for k in range(1,min(m,edge_frames)+1):cost[k,0]=0.
    for k in range(1,min(n,edge_frames)+1):cost[0,k]=0.
    for i in range(1,m+1):
        for j in range(1,n+1):
            best=cost[i-1,j-1]+2*distance[i-1,j-1];step=1
            if j>=2:
                value=cost[i-1,j-2]+1.5*(distance[i-1,j-1]+distance[i-1,j-2])
                if value<best:best=value;step=2
            if i>=2:
                value=cost[i-2,j-1]+1.5*(distance[i-1,j-1]+distance[i-2,j-1])
                if value<best:best=value;step=3
            cost[i,j]=best;trace[i,j]=step
    left=np.empty(m+n,np.int32);right=np.empty(m+n,np.int32);count=0;i=m;j=n
    best=cost[m,n]
    for k in range(1,min(m,edge_frames)+1):
        if cost[m-k,n]<best:best=cost[m-k,n];i=m-k;j=n
    for k in range(1,min(n,edge_frames)+1):
        if cost[m,n-k]<best:best=cost[m,n-k];i=m;j=n-k
    while i>0 and j>0:
        left[count]=i-1;right[count]=j-1;count+=1;step=trace[i,j]
        if step==1:i-=1;j-=1
        elif step==2:
            left[count]=i-1;right[count]=j-2;count+=1;i-=1;j-=2
        else:
            left[count]=i-2;right[count]=j-1;count+=1;i-=2;j-=1
    return left[:count],right[:count]

def compare(query,reference):
    q,qe=query;r,re=reference
    ratio=len(q)/max(1,len(r))
    if not POLICY['duration_ratio'][0]<=ratio<=POLICY['duration_ratio'][1]:return None
    similarity=np.clip(np.einsum('ik,jk->ij',q,r,optimize=False),-1,1)
    qquiet=qe<=max(float(qe.max())*.02,1e-7)
    rquiet=re<=max(float(re.max())*.02,1e-7)
    # Silence has a zero feature vector, so its self cosine is zero rather
    # than one. Align mutual silence normally, but never count it as positive
    # speech evidence or penalize identical references for internal pauses.
    baseline=float(np.median(similarity[np.ix_(~qquiet,~rquiet)])) if (~qquiet).any() and (~rquiet).any() else 0.
    similarity[qquiet[:,None]&rquiet[None,:]]=1.
    edge_frames=round(POLICY['alignment_edge_tolerance_seconds']*RATE/HOP)
    rows,cols=_path(1-similarity,edge_frames)
    if not len(rows):return None
    active=~(qquiet[rows]&rquiet[cols])
    if not active.any():return None
    scores=similarity[rows[active],cols[active]];mean=float(scores.mean());lower=float(np.quantile(scores,.25))
    specificity=mean-baseline
    voiced=~qquiet[rows]&~rquiet[cols]
    a=np.log(np.maximum(qe[rows][voiced],1e-7));b=np.log(np.maximum(re[cols][voiced],1e-7))
    energy=float(np.corrcoef(a,b)[0,1]) if len(a)>=8 and min(float(a.std()),float(b.std()))>1e-5 else 0.
    coverage=min(float(np.sum(qe[np.unique(rows)]**2)/max(np.sum(qe**2),1e-12)),
        float(np.sum(re[np.unique(cols)]**2)/max(np.sum(re**2),1e-12)))
    return {'similarity':mean,'lower_quartile':lower,'temporal_specificity':specificity,
        'energy_correlation':energy,'duration_ratio':ratio,'aligned_energy_coverage':coverage,
        'matched':bool(mean>=POLICY['minimum_similarity'] and lower>=POLICY['minimum_lower_quartile_similarity']
            and specificity>=POLICY['minimum_temporal_specificity'] and energy>=POLICY['minimum_energy_correlation']
            and coverage>=POLICY['minimum_aligned_energy_coverage'])}

def protected_text(reference_text,output_text):
    # A reference-language cognate may be the intended Korean word. This veto
    # never proves a replay or selects a cut, and uses no output recognition.
    from tts_output_guard import phonetic
    a,b=phonetic(reference_text),phonetic(output_text)
    if not a or not b:return False
    matcher=SequenceMatcher(None,a,b,autojunk=False)
    if matcher.ratio()>=.82:return True
    # Protect an intended cognate inside a longer requested sentence as well.
    # A bank member containing it cannot prove that an acoustic match is extra.
    overlap=matcher.find_longest_match().size
    return overlap>=4 and (overlap>=.75*min(len(a),len(b)) or overlap>=6)

def quiet_cut(values,rate,a,b):
    from tts_output_guard import quiet_cut as find_quiet
    return find_quiet(values,rate,a,b)


def match_span(values,rate,a,b,bank):
    duration=(b-a)/rate
    eligible=[part for part in bank if POLICY['duration_ratio'][0]<=duration/(part[2]-part[1])<=POLICY['duration_ratio'][1]]
    if duration<POLICY['minimum_utterance_seconds'] or not eligible:return None
    query=features(mono(values[a:b]),rate);best=None
    for ref,lo,hi,vector,rms in eligible:
        evidence=compare(query,(vector,rms))
        if evidence and (best is None or (evidence['matched'],evidence['similarity'])>(best['matched'],best['similarity'])):
            best={**evidence,'reference_index':ref.get('index'),'reference_start':lo,'reference_end':hi}
    return best


def match_edge_spans(values,rate,groups,bank,matches):
    """Only verified contiguous edge coverage may authorize a later quiet cut.

    Search length is bounded by actual reference durations, not requested TTS
    length. Keep the original strict acoustic checks and cognate protection.
    """
    searches=[]
    for reverse in (False,True):
        order=list(range(len(groups)))
        if reverse:order.reverse()
        offset=0
        while offset<len(order):
            index=order[offset]
            if matches[index]['matched']:offset+=1;continue
            candidates=[]
            max_duration=max((hi-lo for _,lo,hi,_,_ in bank),default=0)*POLICY['duration_ratio'][1]
            for stop in range(offset+1,len(order)):
                indices=order[offset:stop+1];lo=min(indices);hi=max(indices)
                if (groups[hi][1]-groups[lo][0])/rate>max_duration:break
                candidates.append((lo,hi,stop))
            found=False
            for lo,hi,stop in reversed(candidates):
                evidence=match_span(values,rate,groups[lo][0],groups[hi][1],bank)
                searches.append({'first_group':lo,'last_group':hi,'best_match':evidence})
                if evidence and evidence['matched']:
                    for k in range(lo,hi+1):
                        matches[k].update(matched=True,best_match=evidence,matched_span=[lo,hi])
                    offset=stop+1;found=True;break
            if not found:break
    return searches

def inspect_output(source,references,destination):
    values,rate=sf.read(source,dtype='float32',always_2d=True)
    report={'policy':POLICY,'content_verified':False,'asr_requests':0,'source_frames':len(values),'sample_rate':rate}
    if rate!=SR or not len(values) or not np.isfinite(values).all() or np.max(np.abs(values))<1e-5:
        return {**report,'action':'keep_uncertain','reason':'unusable_generated_audio'}
    groups=voice_groups(values,rate)
    if len(groups)<2:return {**report,'action':'keep','reason':'no_separate_utterance','trimmed':False}
    bank=[]
    for reference in references:
        if reference.get('protected'):continue
        path=Path(reference['path']);stat=path.stat()
        # Serialize cache fills; prevent duplicate work when a bank is shared.
        with _FEATURE_LOCK:parts=_reference(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
        bank.extend((reference,a,b,vector,rms) for a,b,vector,rms in parts)
    matches=[]
    for a,b in groups:
        best=match_span(values,rate,a,b,bank)
        matches.append({'start':a/rate,'end':b/rate,'matched':bool(best and best['matched']),'best_match':best})
    report['edge_span_searches']=match_edge_spans(values,rate,groups,bank,matches)
    report['reference_parts']=len(bank)
    report['protected_references']=sum(bool(r.get('protected')) for r in references)
    report['utterances']=matches
    distinct=[i for i,m in enumerate(matches) if not m['matched']]
    if not distinct:return {**report,'action':'keep','reason':'no_distinct_remaining_utterance','trimmed':False}
    first,last=distinct[0],distinct[-1]
    if first==0 and last==len(groups)-1:return {**report,'action':'keep','reason':'no_verified_edge_replay','trimmed':False}
    lo=0;hi=len(values)
    if first:
        lo=quiet_cut(values,rate,groups[first-1][1]/rate,groups[first][0]/rate)
        if lo is None:return {**report,'action':'keep','reason':'no_safe_prefix_gap','trimmed':False}
    if last<len(groups)-1:
        hi=quiet_cut(values,rate,groups[last][1]/rate,groups[last+1][0]/rate)
        if hi is None:return {**report,'action':'keep','reason':'no_safe_suffix_gap','trimmed':False}
    if hi<=lo:return {**report,'action':'keep','reason':'invalid_crop','trimmed':False}
    destination=Path(destination);temporary=destination.with_suffix('.partial.wav')
    sf.write(temporary,values[lo:hi],rate,subtype='PCM_24');temporary.replace(destination)
    return {**report,'action':'cropped','reason':'acoustic_edge_replay_and_distinct_remaining_audio',
        'trimmed':True,'path':str(destination),'keep_start':lo/rate,'keep_end':hi/rate,
        'output_frames':hi-lo,'removed_seconds':(len(values)-hi+lo)/rate}

def check_batch(prepared,cfg,folder,progress=lambda *args:None):
    for p in prepared:
        if 'cached' in p:continue
        try:
            segment=p['segment'];generation=p['generation'];indices=segment.get('_shared_reference_indices',generation.get('shared_reference_indices',[p['index']]))
            directory=Path(p.get('reference_directory',p['folder']));texts=segment.get('_guard_reference_texts',[])
            # The bank is target-first. Donor text follows the same remaining order.
            donor_text=iter(texts);references=[]
            for index in indices:
                own=index==p['index'];text=segment['original_text'] if own else next(donor_text,{}).get('original_text','')
                path=Path(p['reference']) if own else directory/f'{index:04d}-reference.wav'
                references.append({'index':index,'name':str(index),'path':str(path),'own':own,
                    'original_text':text,'translated_text':segment['translated_text'],
                    'protected':protected_text(text,segment['translated_text'])})
            identity={'policy':POLICY,'generated_sha256':hashlib.sha256(Path(p['generated']).read_bytes()).hexdigest(),
                'references':[{**r,'sha256':hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()} for r in references]}
            digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            receipt=Path(folder)/digest/'result.json';destination=Path(p['folder'])/f'{p["index"]:04d}-guarded.wav'
            if receipt.exists():
                saved=json.loads(receipt.read_text('utf8'));report=saved['report']
                if report.get('action')=='cropped' and not Path(report['path']).exists():report=inspect_combined(p['generated'],references,destination)
            else:report=inspect_combined(p['generated'],references,destination)
            atomic_json(receipt,{'identity':identity,'report':report})
        except Exception as error:
            # Inspection failure must not discard a generated line or halt video.
            report={'policy':POLICY,'action':'keep_uncertain','trimmed':False,'content_verified':False,
                'reason':'acoustic_comparison_unavailable','error_type':type(error).__name__,'asr_requests':0}
        generation=p['generation'];generation['tts_output_guard_policy']=POLICY;generation['tts_output_check']=report
        p.pop('_guard_error',None)
        if report.get('action')=='cropped':p['generated']=Path(report['path'])
        atomic_json(Path(p['folder'])/f'{p["index"]:04d}-output-check.json',report)
    return prepared


def inspect_combined(source,references,destination):
    from phonetic_output_guard import inspect
    from leading_replay_guard import inspect as inspect_leading
    previous=inspect(source,references,destination,inspect_output(source,references,destination))
    return inspect_leading(source,references,destination,previous)
