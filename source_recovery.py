"""Scribe -> MAI-Transcribe 2 -> Whisper API recovery; no Gemini calls."""
from pathlib import Path
import copy,difflib,hashlib,json,re,subprocess,time,os
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import soundfile as sf
from core import ROOT,atomic_json,config
from source_alignment import match_key
from source_prefix import select_proposal,LANGUAGES
from qwen_recovery_asr import transcribe

VERSION=8
MODELS=('microsoft/mai-transcribe-2','openai/whisper-large-v3')


def leading_fragment(text):
    m=re.match(r'^(.+?[,，、!?！？…])\s*(\S.*)$',text,re.S)
    return m.group(1) if m and 0<len(match_key(m.group(1)))<=6 else None


def previous_island(values,sr,lower,upper,level):
    from scipy.ndimage import binary_closing
    from reference_edges import spans
    hop=max(1,round(sr*.01));lo=max(0,round(lower*sr));hi=min(len(values),round(upper*sr))
    if hi-lo<hop*4:return None
    x=values[lo:hi];x=np.pad(x,((0,(-len(x))%hop),(0,0)))
    rms=np.sqrt(np.mean(x.reshape(-1,hop,values.shape[1])**2,axis=1)).max(axis=1)
    mask=binary_closing(rms>max(1e-6,level*.025),structure=np.ones(7),border_value=0)
    chunks=[(lo/sr+a*.01,lo/sr+b*.01) for a,b in spans(mask) if b-a>=4]
    return chunks[-1] if chunks else None


def jobs_for(transcript,rows,measured,unmatched,values,sr):
    duration=len(values)/sr;jobs=[]
    ratios=[len(match_key(r['recognized_text']))/len(match_key(r['translated_text'])) for r in rows
        if len(match_key(r['translated_text']))>=8 and r.get('confidence',0)>=.65]
    minimum_ratio=float(np.percentile(ratios,1))*.7 if len(ratios)>=10 else .25
    for i,(row,m) in enumerate(zip(rows,measured)):
        text=row['translated_text']
        if not leading_fragment(text):continue
        floor=max(row.get('source_word_floor',0),measured[i-1]['end'] if i else 0,m['start']-3.5)
        island=previous_island(values,sr,floor,m['start']-.035,m['level'])
        if not island or island[0]>=m['start']-.04:continue
        lo=max(floor,island[0]-.18);hi=min(duration,row.get('source_word_ceiling',duration),m['end']+.08)
        jobs.append(dict(kind='prefix',row_index=i,source_index=row['source_index'],
            translated_text=text,context_start=lo,context_end=hi,asr_start=row['start'],
            neighbor_floor=floor,neighbor_ceiling=row.get('source_word_ceiling',duration),
            neighbor_boundary_measured=True,candidate_island=list(island)))
    for missing in unmatched:
        index=missing['source_index']
        previous=[(r,m) for r,m in zip(rows,measured) if r['source_index']<index]
        following=[(r,m) for r,m in zip(rows,measured) if r['source_index']>index]
        lo=previous[-1][1]['end'] if previous else 0.
        hi=following[0][1]['start'] if following else duration
        if hi<=lo:continue
        # ASR context includes a little neighboring speech, but accepted words
        # must remain in the unclaimed acoustic window.
        left=max(0,lo-.35);right=min(duration,hi+.35)
        if right-left>29:right=min(right,left+29)
        jobs.append(dict(kind='unmatched',source_index=index,translated_text=missing['translated_text'],
            context_start=left,context_end=right,asr_start=lo,
            neighbor_floor=lo,neighbor_ceiling=hi,neighbor_boundary_measured=True,minimum_text_ratio=minimum_ratio,
            isolated_ordered_gap=bool(previous and following and
                previous[-1][0]['source_index']==index-1 and following[0][0]['source_index']==index+1)))
    for number,j in enumerate(jobs):j['id']=number
    return jobs


def align_stage(source,jobs,folder,language,progress=lambda *args:None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    request=folder/'align-request.json';output=folder/'align-proposals.json'
    wanted=[dict(j,source_index=j['id'],text=j['asr_text']) for j in jobs]
    identity=hashlib.sha256(json.dumps([language,wanted],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if output.exists():
        saved=json.loads(output.read_text(encoding='utf8'))
        if saved.get('recovery_identity')==identity:return {p['source_index']:p for p in saved['proposals']}
    atomic_json(request,dict(language=language,jobs=wanted))
    status_file=output.with_suffix('.progress.json')
    atomic_json(status_file,dict(done=0,total=len(jobs),phase='모델 준비'))
    started=time.monotonic()
    with (folder/'align-error.log').open('w',encoding='utf8') as log:
        p=subprocess.Popen([str(ROOT/'.align-venv/Scripts/python.exe'),str(ROOT/'source_prefix_worker.py'),
            str(source),str(request),str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while p.poll() is None:
            try:state=json.loads(status_file.read_text(encoding='utf8'))
            except (OSError,ValueError):state={'done':0,'total':len(jobs),'phase':'정렬 준비'}
            progress(state['done']/max(1,state['total'])*100,
                f'단어 정렬 {state["done"]}/{state["total"]} · {state["phase"]} · {int(time.monotonic()-started)}초')
            if time.monotonic()-started>max(180,20*len(jobs)):
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
                else:p.kill()
                p.wait();return {j['id']:{'error':'local_alignment_timeout'} for j in jobs}
            time.sleep(1)
    if p.returncode:
        return {j['id']:{'error':'local_alignment_failed'} for j in jobs}
    saved=json.loads(output.read_text(encoding='utf8'));saved['recovery_identity']=identity;atomic_json(output,saved)
    return {p['source_index']:p for p in saved['proposals']}


def accept_prefix(values,sr,row,job,proposal):
    target=match_key(job['asr_text']);recognized=''.join(match_key(w['word']) for w in row['words'])
    blocks=difflib.SequenceMatcher(None,target,recognized,autojunk=False).get_matching_blocks()
    first=next((b for b in blocks if b.size),None)
    if first is None or first.a==0:return None,{'reason':'no_missing_opening_in_recognition'}
    spec=dict(job,text=job['asr_text'],missing_prefix_chars=first.a,source_index=row['source_index'])
    updated,report=select_proposal(values,sr,row,spec,proposal)
    if report['status']!='recovered':return None,report
    # Keep the nearest acoustic prelude only, not another earlier voice.
    if updated['start']<job['candidate_island'][0]-.10:return None,{'reason':'earlier_unrelated_island'}
    updated['recognized_text']=''.join(w['word'] for w in updated['words'])
    return updated,report


def accept_missing(job,proposal,duration,source_words,values,sr):
    from translation_alignment import locate
    candidates=[w for w in proposal.get('words',[]) if w['end']-w['start']>=.02
        and w['start']>=job['neighbor_floor']-.04 and w['end']<=job['neighbor_ceiling']+.04]
    for w in candidates:
        w['timing_source']='qwen_source_text_alignment'
        overlap=[s for s in source_words if min(s['end'],w['end'])-max(s['start'],w['start'])>0]
        if overlap:w['speaker_label']=max(overlap,key=lambda s:min(s['end'],w['end'])-max(s['start'],w['start'])).get('speaker_label')
    scopes=[candidates]
    primary=[dict(w,timing_source='scribe_translation_match') for w in source_words
        if (w['end']>w['start'] or not match_key(w['word'])) and w['start']>=job['neighbor_floor']-.04
        and w['end']<=job['neighbor_ceiling']+.04]
    while primary and not match_key(primary[0]['word']):primary.pop(0)
    if primary:scopes.append(primary)
    options=[]
    for words in scopes:
        location=locate([{'translated_text':job['translated_text']}],words,duration,
            min_similarity=.30,allow_gap_rescue=False,min_source_text_ratio=job.get('minimum_text_ratio',0))[0]
        text_ratio=len(match_key(location.get('recognized_text','')))/max(1,len(match_key(job['translated_text'])))
        if location['matched'] and min(location.get('clause_support') or [1.])>=.30 and text_ratio>=job.get('minimum_text_ratio',0):options.append(location)
    ordered_agreement=False
    if not options:
        # Idiomatic translations can have very poor cross-language embedding
        # similarity. A single complete sentence between two assigned anchors
        # can instead be identified by order AND an independent ASR confirming
        # all of Scribe's words. Never force arbitrary multi-line/voice gaps.
        from translation_alignment import speaker,clauses
        text=''.join(w['word'] for w in primary)
        key=match_key(text);native=match_key(job.get('asr_text',''))
        labels={speaker(w) for w in primary if match_key(w['word'])}
        internal_ending=any(re.search(r'[。！？!?]|\.(?=\s|$)',w['word']) and
            any(match_key(later['word']) for later in primary[i+1:]) for i,w in enumerate(primary))
        complete=not internal_ending and len(clauses(text))==1 and len(key)>=5 and len(match_key(job['translated_text']))>=5
        agreement=bool(key and any(b.size>=len(key)*.9 for b in
            difflib.SequenceMatcher(None,key,native,autojunk=False).get_matching_blocks()))
        ratio=len(key)/max(1,len(match_key(job['translated_text'])))
        if not (job.get('isolated_ordered_gap') and complete and agreement and
                len(labels)==1 and None not in labels and job.get('minimum_text_ratio',.25)<=ratio<=3.):
            return [],'recognition_did_not_cover_translation'
        matched=dict(matched=True,start=primary[0]['start'],end=primary[-1]['end'],
            recognized_text=text,words=primary,coverage=.9,asr_speaker_label=next(iter(labels)))
        ordered_agreement=True
    else:matched=max(options,key=lambda r:(min(r.get('clause_support') or [r['coverage']]),r['coverage']))
    if not matched['matched']:return [],matched.get('reason','recognition_did_not_match_translation')
    from source_prefix import speech_support,voicing_support
    start,end=matched['start'],matched['end']
    if speech_support(proposal,start,end)<.15:
        clip=values[round(start*sr):round(end*sr)]
        if not len(clip) or np.max(abs(clip))<1e-5 or voicing_support(clip,sr)<.4:
            return [],'recognized_text_without_measured_voice'
    result=[]
    for part in matched.get('parts',[matched]):
        result.append(dict(index=0,source_index=job['source_index'],start=part['start'],end=part['end'],
            located_start=part['start'],located_end=part['end'],words=part['words'],word_indices=[],
            recognized_text=part['recognized_text'],translated_text=part.get('translated_text',job['translated_text']),
            confidence=part['coverage'],backend=('ordered_gap_with_independent_asr_agreement' if ordered_agreement else 'cascaded_asr_local_forced_alignment'),
            asr_speaker_label=part.get('asr_speaker_label'),speaker_boundary_split=bool(matched.get('parts')),
            source_word_floor=job['neighbor_floor'],source_word_ceiling=job['neighbor_ceiling']))
    return result,('independent_asr_agreement_in_single_ordered_gap' if ordered_agreement else 'native_recognition_and_local_alignment')


def recover(source,values,sr,transcript,rows,measured,unmatched,source_words,folder,language,progress=lambda *args:None):
    lang=LANGUAGES.get(str(language).lower())
    neighborhood_reports=[]
    if lang and transcript.get('segments'):
        from source_neighborhood import repair
        from reference_edges import validated_boundaries
        progress(0,'미연결 문장과 앞뒤 대사 배정 재검토')
        rows,unmatched,neighborhood_reports=repair(source,values,sr,transcript,rows,unmatched,source_words,
            Path(folder)/'neighborhood',lang,config()['key'],
            transcribe,align_stage,MODELS,progress=lambda p,d:progress(p*.60,d))
        if any(r['status']=='recovered' for r in neighborhood_reports):
            rows,measured,failures=validated_boundaries(values,sr,rows,progress)
            known={m['source_index'] for m in unmatched}
            unmatched+= [dict(source_index=f['source_index'],translated_text=transcript['segments'][f['source_index']]['translated_text'],
                              reason=f['reason']) for f in failures if f['source_index'] not in known]
            atomic_json(Path(folder)/'neighborhood-boundary-review.json',{'failures':failures})
    jobs=jobs_for(transcript,rows,measured,unmatched,values,sr)
    completed={r['source_index'] for r in neighborhood_reports}
    jobs=[j for j in jobs if not (j['kind']=='unmatched' and j['source_index'] in completed)]
    if not jobs:return rows,unmatched,neighborhood_reports
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if not lang:return rows,unmatched,[{'status':'unresolved','reason':'unsupported_language'}]
    code={'Japanese':'ja','English':'en','Korean':'ko','Chinese':'zh'}.get(lang)
    key=config()['key'];pending=jobs;reports={j['id']:{'source_index':j['source_index'],'kind':j['kind'],
        'status':'unresolved','attempts':['openai/whisper-large-v3-turbo']} for j in jobs}
    output=copy.deepcopy(rows);additions=[];resolved=set()
    for stage,model in enumerate(MODELS,2):
        if not pending:break
        root=folder/f'stage-{stage}';root.mkdir(exist_ok=True)
        base=60+(stage-2)*18
        progress(base,f'누락 발성·미연결 복구 {stage}/3 · ASR 응답 0/{len(pending)}')
        def request(job):
            job={k:v for k,v in job.items() if k not in ('error','asr_text')}
            local=root/str(job['id']);local.mkdir(exist_ok=True)
            clip=local/'source.wav'
            x=values[round(job['context_start']*sr):round(job['context_end']*sr)]
            sf.write(clip,x[:,int(np.argmax(np.mean(x**2,axis=0)))],sr,subtype='PCM_24')
            try:
                text=transcribe(clip,local,key,code,model=model)['text']
                if lang=='Japanese' and re.search(r'[\u4e00-\u9fff]',text) and not re.search(r'[\u3040-\u30ff]',text):
                    known=''.join(w['word'] for w in source_words if w['start']>=job['context_start'] and w['end']<=job['context_end'])
                    if not set(re.findall(r'[\u4e00-\u9fff]',text))&set(known):
                        return dict(job,error='recognition_not_in_requested_source_language',asr_text=text)
                return dict(job,asr_text=text)
            except RuntimeError as e:return dict(job,error=str(e))
        from source_progress import Progress
        recognized=[]
        with Progress(lambda p,d:progress(base+p*.08,d)).running() as activity:
            activity.update(0,f'누락 발성·미연결 복구 {stage}/3 · ASR 응답 0/{len(pending)} · 최대 8개 병렬')
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(request,j) for j in pending]
                for done,future in enumerate(as_completed(futures),1):
                    recognized.append(future.result())
                    activity.update(done/len(pending)*100,f'누락 발성·미연결 복구 {stage}/3 · ASR 응답 {done}/{len(pending)}')
        recognized.sort(key=lambda j:j['id'])
        candidates=[j for j in recognized if not j.get('error') and j.get('asr_text','').strip()]
        proposals=align_stage(source,candidates,root,lang,
            progress=lambda p,d:progress(base+8+p*.08,f'복구 {stage}/3 · {d}')) if candidates else {}
        remaining=[]
        for job in recognized:
            report=reports[job['id']];report['attempts'].append(model)
            proposal=proposals.get(job['id'],{})
            if job.get('error') or proposal.get('error'):
                report['reason']=job.get('error') or proposal.get('error');remaining.append(job);continue
            if job['kind']=='prefix':
                updated,check=accept_prefix(values,sr,output[job['row_index']],job,proposal)
                success=updated is not None
                if success:output[job['row_index']]=updated
                report['reason']=check.get('reason');report['validation']=check
            else:
                repaired,reason=accept_missing(job,proposal,len(values)/sr,source_words,values,sr)
                success=bool(repaired);report['reason']=reason
                if success:additions.extend(repaired);resolved.add(job['source_index'])
            report['recognition']=job.get('asr_text','')
            if success:report.update(status='recovered',selected_model=model)
            else:remaining.append(job)
        pending=remaining
        atomic_json(folder/'recovery.json',dict(version=VERSION,reports=neighborhood_reports+list(reports.values()),gemini_requests=0))
    output.extend(additions);output.sort(key=lambda r:(r['source_index'],r['start']))
    return output,[r for r in unmatched if r['source_index'] not in resolved],neighborhood_reports+list(reports.values())
