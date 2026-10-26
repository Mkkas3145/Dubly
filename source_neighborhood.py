"""Repair missing complete lines by reconsidering neighboring word ownership."""
from pathlib import Path
import json,difflib,math,re,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from source_progress import Progress
import numpy as np
import soundfile as sf
from source_alignment import match_key
from translation_alignment import clauses,locate,speaker
from core import atomic_json

VERSION=3


def verified_native_words(native,lo,hi,source_words):
    """Validate timed native ASR against independent nearby Scribe anchors."""
    words=[]
    for item in native.get('words',[]):
        a,b=item.get('start'),item.get('end');text=item.get('word',item.get('text',''))
        if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not isinstance(text,str):continue
        if not math.isfinite(a+b) or a<0 or b<=a or b>hi-lo+.2:continue
        words.append(dict(word=text,start=lo+a,end=min(hi,lo+b),timing_source='verified_native_asr'))
    if len(words)<4:return None
    def sequence(items):
        text='';owners=[]
        for i,w in enumerate(items):
            part=match_key(w['word']);text+=part;owners.extend([i]*len(part))
        return text,owners
    text,owners=sequence(words)
    expected=match_key(native.get('text',''))
    if not expected or len(text)/len(expected)<.8:return None
    primary=[w for w in source_words if lo<=w['start'] and w['end']<=hi and w['end']>w['start']]
    known,kowners=sequence(primary)
    pairs=set()
    for block in difflib.SequenceMatcher(None,text,known,autojunk=False).get_matching_blocks():
        pairs.update((owners[block.a+j],kowners[block.b+j]) for j in range(block.size))
    if len(pairs)<4:return None
    deltas=np.asarray([(primary[j]['start']+primary[j]['end']-words[i]['start']-words[i]['end'])/2 for i,j in pairs])
    offset=float(np.median(deltas));good=[(i,j) for (i,j),d in zip(pairs,deltas) if abs(d-offset)<=.45]
    if abs(offset)>.75 or len(good)<4:return None
    anchor_times=[(words[i]['start']+words[i]['end'])/2 for i,j in good]
    if max(anchor_times)-min(anchor_times)<(hi-lo)*.5:return None
    for w in words:
        w['start']=max(lo,w['start']+offset);w['end']=min(hi,w['end']+offset)
        if w['end']<=w['start']:return None
    # A provider may omit punctuation between words even when Scribe retains
    # it. Transfer only sentence marks attached to text matched in both ASRs.
    all_primary=[w for w in source_words if lo<=w['start'] and w['end']<=hi]
    full,kowners=sequence(all_primary);mapping={}
    for block in difflib.SequenceMatcher(None,text,full,autojunk=False).get_matching_blocks():
        mapping.update((block.b+j,owners[block.a+j]) for j in range(block.size))
    cursor=0
    for w in all_primary:
        cursor+=len(match_key(w['word']));marks=re.findall(r'[。！？!?]',w['word'])
        if marks and cursor-1 in mapping:
            target=words[mapping[cursor-1]]
            if not re.search(r'[。！？!?]',target['word']):target['word']+=marks[-1]
    return words


def assign_speakers(words,source_words):
    for w in words:
        overlap=[s for s in source_words if min(s['end'],w['end'])>max(s['start'],w['start'])]
        if overlap:w['speaker_label']=max(overlap,key=lambda s:min(s['end'],w['end'])-max(s['start'],w['start'])).get('speaker_label')
    # Missing ASR text inside a turn does not create a new voice. Propagate
    # only when independent labels on both sides agree.
    for i,w in enumerate(words):
        if speaker(w) is not None:continue
        left=next((speaker(p) for p in reversed(words[:i]) if speaker(p) is not None),None)
        right=next((speaker(p) for p in words[i+1:] if speaker(p) is not None),None)
        if left is not None and left==right:w['speaker_label']=left


def match_neighborhood(targets,words,duration,minimum_ratio=.4):
    """Require every translated phrase, then regroup by source line and voice."""
    queries=[];owners=[]
    for index,text in targets:
        for phrase in clauses(text) or [text]:
            queries.append({'translated_text':phrase});owners.append(index)
    found=locate(queries,words,duration,min_similarity=.30,allow_gap_rescue=False,sentence_boundaries=True)
    if len(found)!=len(queries) or any(not r['matched'] for r in found):return None
    result=[]
    for (index,text) in targets:
        selected=[(q,r) for q,r,owner in zip(queries,found,owners) if owner==index]
        recognized=''.join(r['recognized_text'] for q,r in selected)
        if len(match_key(text))>=8 and len(match_key(recognized))/len(match_key(text))<minimum_ratio:return None
        # Adjacent phrases by the same voice form one continuous reference,
        # including ASR-missed speech between their independently matched anchors.
        groups=[]
        for query,location in selected:
            label=location.get('asr_speaker_label')
            if groups and groups[-1]['asr_speaker_label']==label:
                group=groups[-1];group['last']=location['word_indices'][-1]
                group['translated_text']+=' '+query['translated_text']
                group['confidence']=min(group['confidence'],location['coverage'])
            else:
                groups.append(dict(first=location['word_indices'][0],last=location['word_indices'][-1],
                    translated_text=query['translated_text'],confidence=location['coverage'],asr_speaker_label=label))
        for group in groups:
            a,b=group.pop('first'),group.pop('last');ws=words[a:b+1];positive=[w for w in ws if w['end']>w['start']]
            result.append(dict(group,source_index=index,index=0,start=positive[0]['start'],end=positive[-1]['end'],
                located_start=positive[0]['start'],located_end=positive[-1]['end'],words=ws,word_indices=[],
                recognized_text=''.join(w['word'] for w in ws),backend='neighbor_ownership_and_complete_phrase_alignment',
                speaker_boundary_split=len(groups)>1))
    return result


def repair(source,values,sr,transcript,rows,unmatched,source_words,folder,language,key,transcribe,align_stage,models,progress=lambda *args:None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    segments=transcript.get('segments',[])
    if not segments:return rows,unmatched,[]
    output=list(rows);reports=[];resolved=set();duration=len(values)/sr;jobs=[]
    with Progress(progress).running() as status:
        for number,missing in enumerate(unmatched,1):
            status.update(10*number/max(1,len(unmatched)),f'앞뒤 재검토 · 대상 확인 {number}/{len(unmatched)}')
            index=missing['source_index'];text=missing['translated_text']
            if len(clauses(text))<2:continue
            previous=[r for r in rows if r['source_index']<index];following=[r for r in rows if r['source_index']>index]
            if not previous or not following:continue
            left_index=previous[-1]['source_index'];right_index=following[0]['source_index']
            if right_index-left_index>4:continue
            targets=[(i,segments[i]['translated_text']) for i in range(left_index,right_index+1)]
            group=[r for r in rows if left_index<=r['source_index']<=right_index]
            lo=min(r['start'] for r in group);hi=max(r['end'] for r in group)
            if hi-lo>29:continue
            floor=max((r['end'] for r in rows if r['source_index']<left_index),default=0.)
            ceiling=min((r['start'] for r in rows if r['source_index']>right_index),default=duration)
            lo=max(floor,lo-.35);hi=min(ceiling,hi+.35)
            root=folder/str(index+1);root.mkdir(exist_ok=True)
            jobs.append(dict(index=index,text=text,targets=targets,left=left_index,right=right_index,
                             lo=lo,hi=hi,floor=floor,ceiling=ceiling,root=root,
                             repaired=None,attempts=['openai/whisper-large-v3-turbo'],selected=None))
        for number,job in enumerate(jobs,1):
            status.update(10+15*number/max(1,len(jobs)),f'앞뒤 재검토 · Scribe 구절 연결 {number}/{len(jobs)}')
            primary=[dict(w) for w in source_words if w['start']>=job['lo'] and w['end']<=job['hi']]
            job['repaired']=match_neighborhood(job['targets'],primary,duration)
            if job['repaired']:job['selected']='openai/whisper-large-v3-turbo'
        for stage,model in enumerate(models,2):
            pending=[j for j in jobs if not j['repaired']]
            if not pending:break
            label='MAI-Transcribe 2' if model=='microsoft/mai-transcribe-2' else 'Whisper Large V3'
            base=25+(stage-2)*27
            status.update(base,f'앞뒤 재검토 · {label} 재인식 0/{len(pending)} · 최대 8개 병렬')
            def recognize(job):
                local=job['root']/f'stage-{stage}';local.mkdir(exist_ok=True)
                clip=local/'source.wav';x=values[round(job['lo']*sr):round(job['hi']*sr)]
                sf.write(clip,x[:,int(np.argmax(np.mean(x*x,axis=0)))],sr,subtype='PCM_24')
                started=time.monotonic();job['attempts'].append(model)
                try:
                    native=transcribe(clip,local,key,{'Japanese':'ja','English':'en','Korean':'ko','Chinese':'zh'}.get(language),model=model)
                    job['native']=native;job['last_stage']=stage
                    ws=verified_native_words(native,job['lo'],job['hi'],source_words)
                    if ws is not None:
                        assign_speakers(ws,source_words)
                        job['repaired']=match_neighborhood(job['targets'],ws,duration)
                        if job['repaired']:
                            job['selected']=model
                            atomic_json(local/'verified-native-timestamps.json',{'words':ws,'method':'independent_scribe_anchors'})
                except (RuntimeError,ValueError,KeyError,IndexError) as error:
                    atomic_json(local/'repair-error.json',{'error':str(error)})
                job.setdefault('stage_seconds',{})[model]=round(time.monotonic()-started,3)
                return job
            with ThreadPoolExecutor(max_workers=8,thread_name_prefix='neighbor-asr') as pool:
                futures=[pool.submit(recognize,j) for j in pending]
                for done,future in enumerate(as_completed(futures),1):
                    job=future.result()
                    status.update(base+27*done/len(pending),f'앞뒤 재검토 · {label} 응답 {done}/{len(pending)} · 대사 {job["index"]+1} '+('연결 확인' if job['repaired'] else '검증 중'))
        needs=[j for j in jobs if not j['repaired'] and j.get('native')]
        if needs:
            status.update(80,f'앞뒤 재검토 · 단어 정렬 {len(needs)}개 · GPU 4개씩 묶음 처리')
            requests=[dict(id=i,source_index=j['index'],kind='neighborhood',context_start=j['lo'],context_end=j['hi'],
                           asr_start=j['lo'],asr_text=j['native']['text'],translated_text=j['text'],skip_context_retries=True)
                      for i,j in enumerate(needs)]
            proposals=align_stage(source,requests,folder/'batched-alignment',language,
                progress=lambda p,d:status.update(80+p*.13,'앞뒤 재검토 · '+d))
            for number,job in enumerate(needs):
                ws=[dict(w,timing_source='qwen_source_text_alignment') for w in proposals.get(number,{}).get('words',[]) if w['end']>=w['start']]
                assign_speakers(ws,source_words);job['repaired']=match_neighborhood(job['targets'],ws,duration)
                if job['repaired']:job['selected']=models[job['last_stage']-2]
        claimed=set()
        for number,job in enumerate(jobs,1):
            repaired=job['repaired'];scope=set(range(job['left'],job['right']+1))
            reason='complete_phrases_and_neighbor_ownership' if repaired else 'incomplete_neighborhood'
            if repaired and scope&claimed:repaired=None;reason='overlapping_neighborhood_review'
            report=dict(source_index=job['index'],kind='neighborhood',attempts=job['attempts'],status='recovered' if repaired else 'unresolved',
                        selected_model=job['selected'] if repaired else None,context_start=job['lo'],context_end=job['hi'],
                        reason=reason,stage_seconds=job.get('stage_seconds',{}))
            if repaired:
                for i,r in enumerate(repaired):
                    r['source_word_floor']=repaired[i-1]['end'] if i else job['floor']
                    r['source_word_ceiling']=repaired[i+1]['start'] if i+1<len(repaired) else job['ceiling']
                output=[r for r in output if not job['left']<=r['source_index']<=job['right']]+repaired
                output.sort(key=lambda r:(r['source_index'],r['start']))
                resolved.update(scope);claimed.update(scope)
                atomic_json(job['root']/'repaired.json',dict(report=report,rows=repaired))
            reports.append(report)
            status.update(93+7*number/max(1,len(jobs)),f'앞뒤 재검토 · 결과 적용 {number}/{len(jobs)}')
        status.update(100,f'앞뒤 재검토 완료 · 복구 {sum(r["status"]=="recovered" for r in reports)}/{len(jobs)}개')
    atomic_json(folder/'reports.json',{'version':VERSION,'reports':reports,'gemini_requests':0,'asr_concurrency':8})
    return output,[m for m in unmatched if m['source_index'] not in resolved],reports
