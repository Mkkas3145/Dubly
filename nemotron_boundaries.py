"""Production adapter for the pinned Nemotron EEND-VC worker."""
from pathlib import Path
import json,subprocess
from core import ROOT
from nemotron_packed_policy import identity_for

BOUNDARY_POLICY={'version':1,'same_speaker_gap_seconds':.8,'intervening_speaker':'never_bridge'}

def merge_same_speaker(timeline,folder=None):
    """Bridge short model dropouts only between adjacent turns of the same voice."""
    from copy import deepcopy
    from core import atomic_json
    result=deepcopy(timeline);turns=[];joins=[]
    for turn in sorted(timeline['turns'],key=lambda t:(t['start'],t['end'])):
        if (turns and turns[-1]['speaker']==turn['speaker']
                and 0<=turn['start']-turns[-1]['end']<=BOUNDARY_POLICY['same_speaker_gap_seconds']+1e-7
                and not any(t['speaker']!=turn['speaker'] and t['start']<turn['start']
                            and t['end']>turns[-1]['end'] for t in timeline['turns'])):
            joins.append({'start':turns[-1]['end'],'end':turn['start'],'speaker':turn['speaker']})
            turns[-1]['end']=max(turns[-1]['end'],turn['end'])
        else:turns.append(dict(turn))
    result['raw_turns']=deepcopy(timeline['turns'])
    result['turns']=turns;result['boundary_merges']=joins
    result['boundary_merge_policy']=BOUNDARY_POLICY
    result['identity']={**timeline['identity'],'boundary_merge_policy':BOUNDARY_POLICY}
    if folder is not None:atomic_json(Path(folder)/'merged-timeline.json',result)
    return result

def prepare(clean,folder,progress=lambda *a:None):
    clean,folder=Path(clean),Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cached=folder/'timeline.json'
    if cached.exists():
        try: timeline=json.loads(cached.read_text('utf8'))
        except (ValueError,OSError): timeline={}
        if timeline.get('boundary_authority')=='nemotron_eend_vc' and timeline.get('identity')==identity_for(clean,ROOT):
            progress(100,'Nemotron EEND-VC 경계 캐시 재사용')
            return merge_same_speaker(timeline,folder)
    log=folder/'worker.log'
    with log.open('w',encoding='utf8') as record,subprocess.Popen(
        [str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'nemotron_packed_worker.py'),str(clean),str(folder)],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)) as process:
        for line in process.stdout:
            record.write(line);record.flush()
            try:event=json.loads(line)
            except ValueError:continue
            stage=event.get('stage')
            if stage=='load':progress(3,'Nemotron 단독 분석 준비 · 여러 참조 후보 동시 비교')
            elif stage=='session':progress(5+75*event['fraction'],f'Nemotron 세션 {event["session"]} · {event["position"]:.1f}초 · 지역 화자 {event["speakers"]}명')
            elif stage=='embeddings':progress(80+14*event['done']/max(1,event['total']),f'ReDimNet2+ 화자 특징 추출 · {event["done"]}/{event["total"]}')
            elif stage=='clustering':progress(95,f'화자 전역 연결 · {event["sessions"]}세션 · {event["nodes"]}개 지역 ID · 같은 세션 다른 화자 병합 금지')
            elif stage=='window':progress(5+80*event['fraction'],f'Nemotron 구간 분석 · {event["start"]:.0f}–{event["end"]:.0f}초 · 지역 화자 {event["local_speakers"]}명')
            elif stage=='capacity_split':progress(10,'8화자 슬롯 포화 · 해당 분석 구간을 더 짧게 재분석')
            elif stage=='linking':progress(88,f'겹치는 동일 발화로 전역 화자 연결 · {event["windows"]}구간')
            elif stage=='local_views':progress(85+3*event['done']/max(1,event['total']),f'Nemotron 짧은 문맥 교차 검증 · {event["done"]}/{event["total"]}')
            elif stage=='verification':progress(88+11*event['done']/max(1,event['total']),f'Nemotron 자체 참조 비교 · {event["done"]}/{event["total"]} · 연결 {event["verified_merges"]}개')
            elif stage=='verification_detail':progress(88+11*event['done']/max(1,event['total']),f'Nemotron 여러 참조 동시 비교 · {event["done"]}/{event["total"]} · 공유 입력 {event["candidate"]}/{event["candidates"]} · 연결 {event["verified_merges"]}개')
            elif stage=='repair':progress(90,f'Nemotron 재비교로 혼합된 지역 화자 분리 · {event["windows"]}구간 · 전역 연결 재검증')
            elif stage=='round':progress(90,f'Nemotron 전역 재비교 {event["round"]}차 · 갱신된 참조로 {event["components"]}개 화자 묶음 재검증')
            elif stage=='cached':progress(100,'Nemotron EEND-VC 경계 캐시 재사용')
            elif stage=='complete':progress(100,f'Nemotron EEND-VC · 전역 {event["speakers"]}화자')
        if process.wait():raise RuntimeError(f'Nemotron analysis failed. See {folder.name}/worker.log.')
    timeline=json.loads((folder/'timeline.json').read_text('utf8'))
    if timeline.get('boundary_authority')!='nemotron_eend_vc':raise RuntimeError('Nemotron boundary cache format mismatch.')
    return merge_same_speaker(timeline,folder)
