"""Verify native MAI turns acoustically before same-speaker pause merging."""
from pathlib import Path
from local_resources import local_stage
import copy,hashlib,json,subprocess
import numpy as np
import soundfile as sf
from core import ROOT,SR,atomic_json
from reference_bank_guard import MODEL,MODEL_SHA,EMBEDDING_POLICY

POLICY={'version':3,'model':MODEL,'sha256':MODEL_SHA,
    'native_margin_seconds':.04,'minimum_clip_seconds':.2,
    'minimum_anchor_seconds':1.5,'witness_count':5,'minimum_witnesses':3,
    'exclude_nearby_seconds':10.,'adjacent_gap_seconds':.8,
    'voice_change_max_cosine':.35,'minimum_candidate_median':.55,
    'minimum_three_witness_cosine':.55,'minimum_runner_up_margin':.04,
    'minimum_original_margin':.08,'reciprocal_minimum_original_margin':.04,
    'units':'native_segment_fences_plus_word_speaker_changes_and_pause_gaps',
    'corrections':'same_id_voice_conflict_or_two_independent_reciprocal_swaps',
    'post_reassignment_merge_check':True,
    'uncertain_identity':'preserve_asr_id_but_block_conflicting_merge',
    'text_and_time_changes':False,'additional_asr_requests':0}


def source_units(raw,duration):
    from mai_pipeline import normalize,word_regions
    fences=[r['start'] for r in normalize(raw,duration)[1:]]
    return [{**r,'source_index':i} for i,r in enumerate(word_regions(raw,duration,merge_breaks=fences))]


def decide(native,anchors,vectors):
    vectors=np.asarray(vectors,dtype=np.float32)
    all_rows=[*native,*anchors]
    if vectors.ndim!=2 or len(vectors)!=len(all_rows) or not np.isfinite(vectors).all():
        raise RuntimeError('Invalid utterance speaker embedding format.')
    norms=np.linalg.norm(vectors,axis=1)
    if np.any(norms<.5):raise RuntimeError('Invalid utterance speaker embedding values.')
    vectors=vectors/norms[:,None];scores=vectors[:len(native)]@vectors.T
    decisions=[];n=POLICY['witness_count'];minimum=POLICY['minimum_witnesses']
    for i,row in enumerate(native):
        groups=[]
        for voice in sorted({r['speaker'] for r in anchors}-{'unknown'}):
            peers=[j for j,a in enumerate(anchors) if a['speaker']==voice
                and a['seconds']>=POLICY['minimum_anchor_seconds']
                and (a['end']+POLICY['exclude_nearby_seconds']<=row['start']
                     or a['start']-POLICY['exclude_nearby_seconds']>=row['end'])]
            peers=sorted(peers,key=lambda j:float(scores[i,len(native)+j]),reverse=True)[:n]
            if len(peers)<minimum:continue
            values=[float(scores[i,len(native)+j]) for j in peers]
            groups.append({'speaker':voice,'median':float(np.median(values)),
                'third_cosine':values[minimum-1],
                'witnesses':[anchors[j]['source_index'] for j in peers],'cosines':values})
        groups.sort(key=lambda g:g['median'],reverse=True)
        decisions.append({'source_index':row['source_index'],'original_speaker':row['speaker'],
            'speaker':row['speaker'],'start':row['start'],'end':row['end'],
            'ranking':groups,'action':'keep'})
    # All comparisons use immutable original embeddings/labels, never prior repairs.
    conflicts=set();breaks=[]
    for i in range(1,len(native)):
        left,right=native[i-1],native[i]
        if right['source_index']!=left['source_index']+1:continue
        gap=right['start']-left['end']
        if left['speaker']!=right['speaker'] or not 0<=gap<=POLICY['adjacent_gap_seconds']+1e-7:continue
        a,b=decisions[i-1]['ranking'],decisions[i]['ranking']
        if not a or not b or a[0]['speaker']==b[0]['speaker']:continue
        cosine=float(scores[i,i-1])
        if cosine>=POLICY['voice_change_max_cosine'] or max(a[0]['median'],b[0]['median'])<POLICY['minimum_candidate_median']:continue
        conflicts.update((i-1,i));breaks.append(right['start'])
        for j in (i-1,i):decisions[j].setdefault('adjacent_conflicts',[]).append({'with':native[i if j==i-1 else i-1]['source_index'],'cosine':cosine})
    for i in conflicts:
        entry=decisions[i];rank=entry['ranking'];best=rank[0]
        if best['speaker']==entry['original_speaker']:continue
        own=next((r['median'] for r in rank if r['speaker']==entry['original_speaker']),None)
        if (own is not None and best['median']>=POLICY['minimum_candidate_median']
            and best['third_cosine']>=POLICY['minimum_three_witness_cosine']
            and (len(rank)==1 or best['median']-rank[1]['median']>=POLICY['minimum_runner_up_margin'])
            and best['median']-own>=POLICY['minimum_original_margin']):
            entry.update(speaker=best['speaker'],action='reassign_native_turn')
    def reciprocal_evidence(entry):
        rank=entry['ranking']
        if len(rank)<2:return False
        best=rank[0];own=next((r['median'] for r in rank if r['speaker']==entry['original_speaker']),None)
        return (best['speaker']!=entry['original_speaker'] and own is not None
            and best['median']>=POLICY['minimum_candidate_median']
            and best['third_cosine']>=POLICY['minimum_three_witness_cosine']
            and best['median']-rank[1]['median']>=POLICY['minimum_runner_up_margin']
            and best['median']-own>=POLICY['reciprocal_minimum_original_margin'])
    for i in range(1,len(native)):
        left,right=native[i-1],native[i];a,b=decisions[i-1],decisions[i]
        if right['source_index']!=left['source_index']+1 or left['speaker']==right['speaker']:continue
        if not 0<=right['start']-left['end']<=POLICY['adjacent_gap_seconds']+1e-7:continue
        if not reciprocal_evidence(a) or not reciprocal_evidence(b):continue
        if a['ranking'][0]['speaker']!=right['speaker'] or b['ranking'][0]['speaker']!=left['speaker']:continue
        for entry in (a,b):
            entry.update(speaker=entry['ranking'][0]['speaker'],action='reassign_native_turn',reason='reciprocal_independent_voice_witnesses')
        breaks.append(right['start'])
    # Relabeling can make formerly distinct IDs equal. Recheck those new merge
    # candidates using the unchanged acoustic evidence, not the repaired labels.
    for i in range(1,len(native)):
        left,right=native[i-1],native[i];a,b=decisions[i-1],decisions[i]
        if right['source_index']!=left['source_index']+1 or a['speaker']!=b['speaker']:continue
        if not 0<=right['start']-left['end']<=POLICY['adjacent_gap_seconds']+1e-7:continue
        x,y=a['ranking'],b['ranking']
        if not x or not y or x[0]['speaker']==y[0]['speaker']:continue
        if (float(scores[i,i-1])<POLICY['voice_change_max_cosine']
            and max(x[0]['median'],y[0]['median'])>=POLICY['minimum_candidate_median']):
            breaks.append(right['start'])
            b['post_reassignment_merge_blocked']=True
    return decisions,sorted(set(breaks))


def apply_decisions(raw,native,decisions):
    """Repair whole native turns and their own word labels, preserving all text/times."""
    result=copy.deepcopy(raw)
    changes={d['source_index']:d['speaker'] for d in decisions if d['action']=='reassign_native_turn'}
    def owner(item):
        a,b=item.get('start'),item.get('end')
        if not isinstance(a,(int,float)) or not isinstance(b,(int,float)):return None
        overlaps=[max(0.,min(b,r['end'])-max(a,r['start'])) for r in native]
        if not overlaps or max(overlaps)<=0:return None
        return native[int(np.argmax(overlaps))]['source_index']
    for kind in ('segments','words'):
        for item in result.get(kind,[]):
            if kind=='segments':
                parts=[r for r in native if min(r['end'],item['end'])>max(r['start'],item['start'])]
                # A provider segment can span multiple voices and long pauses.
                # Never relabel that entire segment from its largest child.
                if len(parts)!=1:continue
            index=owner(item)
            if index in changes:item['speaker']=changes[index]
    return result


@local_stage
def prepare(clean,raw,folder,progress=lambda *a:None):
    from mai_pipeline import normalize,word_regions,correct_edges
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    clean=Path(clean);duration=sf.info(clean).duration
    native=source_units(raw,duration)
    baseline=word_regions(raw,duration)
    anchors,_=correct_edges(clean,[{**r,'id':i} for i,r in enumerate(baseline)])
    anchors=[{**r,'source_index':i,'seconds':r['end']-r['start']} for i,r in enumerate(anchors)
        if r['end']-r['start']>=POLICY['minimum_anchor_seconds'] and r['speaker']!='unknown']
    identity={'policy':POLICY,'segments':raw.get('segments',[]),'words':raw.get('words',[])}
    with clean.open('rb') as f:identity['audio_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    cache=folder/f'{signature}.json'
    if cache.exists():
        report=json.loads(cache.read_text('utf8'));progress(100,'병합 전 화자 후보정 캐시 재사용')
        return apply_decisions(raw,native,report['decisions']),report['merge_breaks'],report
    eligible=[r for r in native if r['end']-r['start']>=POLICY['minimum_clip_seconds'] and r['speaker']!='unknown']
    report={'policy':POLICY,'signature':signature,'decisions':[],'merge_breaks':[],'changes':[]}
    if len({r['speaker'] for r in anchors})<2 or not eligible:
        report['reason']='insufficient_independent_anchor_groups'
        atomic_json(cache,report);return raw,[],report
    progress(0,f'병합 전 발화 화자 비교 · {len(eligible)}개')
    run=folder/signature;run.mkdir(exist_ok=True);embedding_rows=[]
    with sf.SoundFile(clean) as audio:
        for i,r in enumerate([*eligible,*anchors]):
            a,b=r['start'],r['end']
            if i<len(eligible):
                index=r['source_index'];left=native[index-1]['end'] if index else 0.
                right=native[index+1]['start'] if index+1<len(native) else duration
                a=max((left+a)/2 if left<=a else a,a-POLICY['native_margin_seconds'])
                b=min((b+right)/2 if right>=b else b,b+POLICY['native_margin_seconds'])
            lo=max(0,round(a*audio.samplerate));hi=min(len(audio),round(b*audio.samplerate))
            audio.seek(lo);x=audio.read(hi-lo,dtype='float32',always_2d=True)
            path=run/f'{i:04d}.wav';sf.write(path,x.mean(axis=1),audio.samplerate,subtype='FLOAT')
            embedding_rows.append({'index':i,'speaker':r['speaker'],'seconds':len(x)/audio.samplerate,'path':str(path)})
    manifest=run/'embedding-input.json'
    atomic_json(manifest,{'rows':embedding_rows,'policy':EMBEDDING_POLICY,'cache':str(ROOT/'work/reference-speaker-embeddings-v1')})
    with (run/'worker.log').open('w',encoding='utf8') as log,subprocess.Popen(
        [str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'reference_bank_worker.py'),str(manifest),str(run/'vectors.npy')],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace') as proc:
        for line in proc.stdout:
            log.write(line);log.flush()
            try:event=json.loads(line)
            except ValueError:continue
            if 'done' in event:progress(90*event['done']/len(embedding_rows),f'발화 화자 임베딩 {event["done"]}/{len(embedding_rows)}')
        if proc.wait():raise RuntimeError('Utterance speaker correction failed. See worker.log.')
    decisions,breaks=decide(eligible,anchors,np.load(run/'vectors.npy',allow_pickle=False))
    report.update(decisions=decisions,merge_breaks=breaks,changes=[d for d in decisions if d['action']!='keep'])
    atomic_json(cache,report);atomic_json(folder/'audit.json',report)
    progress(100,f'병합 전 화자 후보정 · 재배정 {len(report["changes"])}개 · 병합 차단 {len(breaks)}곳')
    return apply_decisions(raw,native,decisions),breaks,report
