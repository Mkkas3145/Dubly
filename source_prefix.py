"""Recover text-supported omitted onsets; never extend every reference by padding.

Scribe locates dialogue. Full Gemini text is aligned to local audio only when
its opening is missing from Scribe. Qwen proposals must survive acoustic and
neighbor checks; unsupported words are never assigned invented durations.
"""
import difflib,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from core import ROOT,atomic_json
from source_alignment import match_key

VERSION=7
# Inference proposals can be reused when only acceptance checks change.
WORKER_VERSION=5
LANGUAGES={'ja':'Japanese','jpn':'Japanese','japanese':'Japanese','en':'English','eng':'English','english':'English',
    'ko':'Korean','kor':'Korean','korean':'Korean','zh':'Chinese','zho':'Chinese','chinese':'Chinese',
    'yue':'Cantonese','cantonese':'Cantonese','fr':'French','fra':'French','french':'French',
    'de':'German','deu':'German','german':'German','it':'Italian','ita':'Italian','italian':'Italian',
    'pt':'Portuguese','por':'Portuguese','portuguese':'Portuguese','ru':'Russian','rus':'Russian','russian':'Russian',
    'es':'Spanish','spa':'Spanish','spanish':'Spanish'}


def prefix_jobs(segments,rows,duration,neighbor_measurements=None):
    jobs=[]
    for i,row in enumerate(rows):
        target=match_key(segments[row['source_index']]['original_text'])
        recognized=''.join(match_key(w['word']) for w in row['words'])
        blocks=difflib.SequenceMatcher(None,target,recognized,autojunk=False).get_matching_blocks()
        first=next((block for block in blocks if block.size),None)
        if first is None or first.a==0:continue
        # Neighbor ASR anchors bound the search, independent of LLM timestamps.
        measured=neighbor_measurements is not None
        floor=neighbor_measurements[i-1]['core_end'] if measured and i else rows[i-1]['end'] if i else 0
        lo=max(0,floor)
        hi=min(duration,rows[i+1]['start'] if i+1<len(rows) else duration,row['end']+.5)
        # Bound inference context, not the eventual utterance length. Very long
        # disconnected ASR gaps are explicitly recorded, not silently guessed.
        if hi-lo>29:lo=max(lo,row['start']-max(2.,29-(hi-row['start'])))
        if hi<=lo:continue
        jobs.append({'source_index':row['source_index'],'text':segments[row['source_index']]['original_text'],
            'missing_prefix_chars':first.a,'context_start':lo,'context_end':hi,
            'asr_start':row['start'],
            'neighbor_floor':floor,'neighbor_boundary_measured':measured,
            'neighbor_ceiling':rows[i+1]['start'] if i+1<len(rows) else duration})
    return jobs


def speech_support(proposal,start,end):
    values=np.asarray(proposal.get('vad',[]));step=proposal.get('vad_step',.032)
    if not len(values) or end<=start:return 0.
    lo=max(0,int((start-proposal['vad_start'])/step));hi=min(len(values),int(np.ceil((end-proposal['vad_start'])/step)))
    return float(np.mean(values[lo:hi])) if hi>lo else 0.


def voicing_support(part,sr):
    """Corroborate short voiced syllables which a general VAD can miss."""
    if not len(part):return 0.
    channel=int(np.argmax(np.mean(part**2,axis=0)))
    mono=part[:,channel];width=round(sr*.03);hop=round(sr*.01);scores=[]
    for start in range(0,max(1,len(mono)-width+1),hop):
        x=mono[start:start+width].astype(np.float64)
        if len(x)<round(sr*.015):continue
        x=(x-x.mean())*np.hanning(len(x))
        n=1<<(2*len(x)-1).bit_length()
        ac=np.fft.irfft(np.abs(np.fft.rfft(x,n))**2,n)[:len(x)]
        a=max(1,round(sr/1200));b=min(len(ac),round(sr/60))
        if b>a and ac[0]>1e-12:scores.append(float(np.max(ac[a:b])/ac[0]))
    return float(np.percentile(scores,75)) if scores else 0.


def apply_proposal(values,sr,row,job,proposal):
    report={'source_index':row['source_index'],'status':'unchanged','asr_start':row['start'],'asr_end':row['end']}
    def reject(reason):return row,{**report,'reason':reason}
    if proposal.get('error'):return reject('aligner_error: '+proposal['error'])
    words=proposal.get('words',[])
    if ''.join(match_key(w['word']) for w in words)!=match_key(job['text']):return reject('aligned_text_mismatch')
    if any(not np.isfinite(w['start']) or not np.isfinite(w['end']) or w['end']<w['start']
           or w['start']<job['context_start']-.02 or w['end']>job['context_end']+.02 for w in words):
        return reject('invalid_model_timestamps')
    positive=[dict(w,probability=1.,timing_source='qwen_source_text_alignment')
        for w in words if w['end']-w['start']>=.02]
    if not positive:return reject('collapsed_alignment')
    cursor=0;prefix=[];body_chars=0
    for word in words:
        size=len(match_key(word['word']))
        if cursor<job['missing_prefix_chars'] and word['end']-word['start']>=.02:prefix.append(word)
        if word['end']-word['start']>=.02:
            body_chars+=max(0,cursor+size-max(cursor,job['missing_prefix_chars']))
        cursor+=size
    # Some short words receive [t,t]. Do not fabricate an interval for them.
    # A validated first *positive* word still allows the edge detector to trace
    # its soft attack, including an adjoining omitted onset, from real samples.
    old_support=speech_support(proposal,row['start'],row['end'])
    new_support=speech_support(proposal,positive[0]['start'],positive[-1]['end'])
    report.update(aligned_start=positive[0]['start'],aligned_end=positive[-1]['end'],
        asr_speech_support=old_support,aligned_speech_support=new_support,
        zero_duration_tokens=[w['word'] for w in words if w['end']-w['start']<.02])
    # The known ASR body must remain positively aligned. A zero-duration body
    # cannot justify attaching a different earlier voice as the missing prefix.
    expected_body=max(1,len(match_key(job['text']))-job['missing_prefix_chars'])
    if body_chars/expected_body<.65:return reject('recognized_body_collapsed_in_alignment')
    if abs(positive[-1]['end']-row['end'])>.5:return reject('body_alignment_disagrees_with_asr')
    if not prefix:return reject('missing_prefix_has_no_positive_alignment')
    prefix=[dict(w,probability=1.,timing_source='qwen_source_text_alignment',
                provider_start=w['start'],provider_end=w['end'])
        for w in prefix if w['start']<row['start']-.04 and w['end']<=row['start']+.10]
    if not prefix:return reject('no_earlier_supported_prefix')
    candidate=prefix+[dict(w) for w in row['words']]
    reason='full_text_prefix_before_asr_anchor'
    first=candidate[0];lo=max(0,round(first['start']*sr));hi=min(len(values),round(first['end']*sr))
    if not job.get('neighbor_boundary_measured') and job['neighbor_floor']>0 and first['start']-job['context_start']<.08:
        return reject('prefix_stuck_at_previous_dialogue_context_edge')
    if hi<=lo:return reject('no_prefix_samples')
    part=values[lo:hi];frame=max(1,round(.01*sr))
    part=np.pad(part,((0,(-len(part))%frame),(0,0)))
    rms=np.sqrt(np.mean(part.reshape(-1,frame,values.shape[1])**2,axis=1)).max(axis=1)
    body=values[max(0,round(row['start']*sr)):min(len(values),round(row['end']*sr))]
    body_level=float(np.sqrt(np.mean(body**2))) if len(body) else 0.
    if not len(rms) or np.max(rms)<max(1e-6,body_level*.04):return reject('prefix_has_no_acoustic_energy')
    occupancy=float(np.mean(rms>max(1e-6,body_level*.04,float(rms.max())*.08)))
    report['prefix_voice_occupancy']=occupancy
    if occupancy<.45:return reject('prefix_alignment_spans_unvoiced_gap')
    if speech_support(proposal,first['start']-.05,first['end']+.05)<.2:
        # A VAD can miss high-pitched stammers. Keep a positive aligned prefix
        # if already inside the current acoustic search, but do not attach a
        # new isolated low-confidence event far before it.
        voiced=voicing_support(values[lo:hi],sr);report['prefix_voicing_support']=voiced
        if first['start']<row['start']-.32 and voiced<.4:return reject('isolated_prefix_not_confirmed_as_speech')
    if first['start']<job['neighbor_floor']-.01 or candidate[-1]['end']>job['neighbor_ceiling']+.01:
        return reject('neighbor_boundary_conflict')
    updated={**row,'start':candidate[0]['start'],'end':candidate[-1]['end'],'words':candidate,
        'onset_alignment_model':'Qwen/Qwen3-ForcedAligner-0.6B','prefix_recovery':reason,
        'scribe_start_before_prefix_recovery':row['start'],'scribe_end_before_prefix_recovery':row['end']}
    return updated,{**report,'status':'recovered','reason':reason,
        'recovered_start':updated['start'],'recovered_end':updated['end']}


def select_proposal(values,sr,row,job,proposal):
    rejected=[];accepted=[]
    for number,words in enumerate([proposal.get('words',[])]+proposal.get('alternatives',[])):
        updated,report=apply_proposal(values,sr,row,job,{**proposal,'words':words})
        if report['status']=='recovered':
            report['alignment_context']='asr_body_quiet_gap_retry' if number else 'whole_context'
            accepted.append((updated,report))
        else:rejected.append(dict(report))
    if accepted:
        # Prefer the nearest complete, positively aligned prefix to the known
        # body, rather than the earliest sound anywhere in the search window.
        updated,report=max(accepted,key=lambda pair:pair[0]['start'])
    else:updated=row;report=dict(rejected[-1])
    report['rejected_proposals']=rejected
    return updated,report


def recover_prefixes(source,values,sr,segments,rows,folder,language,progress=lambda *args:None,neighbor_measurements=None):
    jobs=prefix_jobs(segments,rows,len(values)/sr,neighbor_measurements)
    if not jobs:return rows,[]
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    language=LANGUAGES.get(str(language).lower())
    if language is None:
        return rows,[{'source_index':j['source_index'],'status':'unchanged','reason':'unsupported_or_unknown_alignment_language'} for j in jobs]
    identity=hashlib.sha256(json.dumps([WORKER_VERSION,language,jobs,hashlib.sha256(Path(source).read_bytes()).hexdigest()],
        ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    request=folder/'request.json';cache=folder/'proposals.json'
    old=json.loads(cache.read_text(encoding='utf8')) if cache.exists() else {}
    if old.get('identity')!=identity:
        atomic_json(request,{'language':language,'jobs':jobs})
        progress(0,f'ASR 누락 발성 확인 · {len(jobs)}개')
        python=ROOT/'.align-venv/Scripts/python.exe'
        if not python.is_file():raise RuntimeError('Missing vocalization alignment runtime. Run setup-alignment.ps1.')
        result=subprocess.run([str(python),str(ROOT/'source_prefix_worker.py'),str(source),str(request),str(cache)],
            cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=max(180,15*len(jobs)))
        if result.returncode:
            (folder/'worker-error.log').write_text(result.stderr,encoding='utf8')
            raise RuntimeError('Missing vocalization alignment failed. See source-prefix/worker-error.log.')
        old=json.loads(cache.read_text(encoding='utf8'));old['identity']=identity;atomic_json(cache,old)
    proposals={p['source_index']:p for p in old['proposals']};by_index={j['source_index']:j for j in jobs}
    output=[];reports=[]
    for row in rows:
        index=row['source_index']
        if index not in by_index:output.append(row);continue
        proposal=proposals.get(index,{'error':'missing proposal'})
        updated,report=select_proposal(values,sr,row,by_index[index],proposal)
        output.append(updated);reports.append(report)
    atomic_json(folder/'recovery.json',{'version':VERSION,'reports':reports})
    progress(100,f'누락 발성 확인 완료 · 복구 {sum(r["status"]=="recovered" for r in reports)}개')
    return output,reports
