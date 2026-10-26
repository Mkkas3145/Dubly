"""Detect strongly contradicted donor assignments; never rewrite ASR dialogue IDs."""
from pathlib import Path
from local_resources import local_stage
import hashlib,json,subprocess
import numpy as np
import soundfile as sf
from core import ROOT,atomic_json

MODEL='lab260/redimnet2-plus'
MODEL_SHA='57056ea3ff1afcf6d825f0f44abc0a4c68fcdd6738ec36e23017df060712f02b'
BASE_SHA='287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868'
EMBEDDING_POLICY={'version':1,'model':MODEL,'sha256':MODEL_SHA,'precision':'float32',
    'embedding':'up_to_three_real_four_second_views_normalized_mean',
    'minimum_candidate_seconds':1.2,'minimum_witness_seconds':1.5,'witness_count':3,
    'alternative_witness_min_cosine':.70,'minimum_median_margin':.25,
    'maximum_own_median_cosine':.45,'minimum_own_peer_cohesion':.35,
    'action':'reference_bank_only_consensus_reassignment','short_or_uncertain':'keep_original_bank',
    'source_speaker_id':'unchanged','regeneration':False}
# Keep feature identities stable when changing decisions, not the encoder.
POLICY={**EMBEDDING_POLICY,'version':2,'minimum_witness_pair_cosine':.60,
    'minimum_own_top_witness_margin':.08,'maximum_own_top_witness_cosine':.55,
    'original_bank_cohesion_required':False}
DONOR_POLICY={'version':2,'model':MODEL,'donors':5,'minimum_anchor_seconds':1.5,
    'nearby_exclusion_seconds':10.,'ranking':'cosine_within_corrected_speaker_bank',
    'minimum_target_cosine':.30,'maximum_drop_from_best':.25,'minimum_pair_cosine':.30,
    'current_utterance':'always_first','inserted_silence':False}


def select_donors(rows,vectors,decisions,segments):
    """Rank independent utterances inside each final voice bank, not by script."""
    vectors=np.asarray(vectors,dtype=np.float32)
    vectors=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
    scores=vectors@vectors.T
    voices={d['index']:d['bank_speaker'] for d in decisions}
    selected=[]
    for i,row in enumerate(rows):
        index=row['index'];target=segments[index];voice=voices[index]
        peers=[j for j,r in enumerate(rows) if j!=i and voices[r['index']]==voice
            and (segments[r['index']]['end']<=target['start'] or segments[r['index']]['start']>=target['end'])]
        long=[j for j in peers if rows[j]['seconds']>=DONOR_POLICY['minimum_anchor_seconds']]
        distant=[j for j in long if segments[rows[j]['index']]['end']+DONOR_POLICY['nearby_exclusion_seconds']<=target['start']
            or segments[rows[j]['index']]['start']-DONOR_POLICY['nearby_exclusion_seconds']>=target['end']]
        pool=distant or long or peers
        candidates=sorted(pool,key=lambda j:(-float(scores[i,j]),rows[j]['index']))
        ranked=[];rejected=[]
        best=float(scores[i,candidates[0]]) if candidates else 0.
        floor=max(DONOR_POLICY['minimum_target_cosine'],best-DONOR_POLICY['maximum_drop_from_best'])
        for j in candidates:
            reason=None
            if scores[i,j]<floor:reason='weak_target_identity'
            elif any(scores[j,k]<DONOR_POLICY['minimum_pair_cosine'] for k in ranked):reason='conflicting_donor_identity'
            if reason:rejected.append({'index':rows[j]['index'],'reason':reason});continue
            ranked.append(j)
            if len(ranked)==DONOR_POLICY['donors']:break
        indices=[index,*[rows[j]['index'] for j in ranked]]
        segments[index]['_reference_donor_indices']=indices
        selected.append({'index':index,'speaker':voice,'reference_indices':indices,
            'donor_cosines':[float(scores[i,j]) for j in ranked],'rejected':rejected,
            'minimum_pair_cosine':min((float(scores[j,k]) for a,j in enumerate(ranked) for k in ranked[a+1:]),default=None)})
    return selected


def decide(rows,vectors):
    """All votes use the original partition; corrections cannot self-reinforce."""
    vectors=np.asarray(vectors,dtype=np.float32)
    if vectors.ndim!=2 or len(vectors)!=len(rows) or not np.isfinite(vectors).all():raise RuntimeError('Invalid speaker embedding format.')
    norms=np.linalg.norm(vectors,axis=1)
    if np.any(norms<.5):raise RuntimeError('Invalid speaker embedding values.')
    scores=(vectors/norms[:,None])@(vectors/norms[:,None]).T
    result=[];n=POLICY['witness_count']
    for i,row in enumerate(rows):
        voice=row['speaker'];entry={'index':row['index'],'original_speaker':voice,'bank_speaker':voice,'action':'keep'}
        if row.get('speaker_locked'):
            result.append({**entry,'reason':'user_selected_speaker'});continue
        peers=[j for j,r in enumerate(rows) if j!=i and r['speaker']==voice]
        if row['seconds']<POLICY['minimum_candidate_seconds'] or len(peers)<3:
            result.append({**entry,'reason':'short_utterance_or_insufficient_peers'});continue
        own=float(np.median(scores[i,peers]))
        own_witnesses=sorted([j for j in peers if rows[j]['seconds']>=POLICY['minimum_witness_seconds']],
            key=lambda j:float(scores[i,j]),reverse=True)[:n]
        own_top=float(np.median(scores[i,own_witnesses])) if len(own_witnesses)==n else own
        matrix=scores[np.ix_(peers,peers)]
        cohesion=float(np.median(matrix[np.triu_indices(len(peers),1)]))
        entry.update(own_median=own,own_peer_cohesion=cohesion,own_top_witness_median=own_top)
        alternatives=[]
        for candidate in sorted(set(r['speaker'] for r in rows)-{voice,'unknown'}):
            witnesses=sorted([j for j,r in enumerate(rows) if r['speaker']==candidate and r['seconds']>=POLICY['minimum_witness_seconds']],
                             key=lambda j:float(scores[i,j]),reverse=True)[:n]
            if len(witnesses)<n:continue
            values=[float(scores[i,j]) for j in witnesses]
            pair_min=min(float(scores[j,k]) for a,j in enumerate(witnesses) for k in witnesses[a+1:])
            alternatives.append({'speaker':candidate,'median':float(np.median(values)),'minimum':min(values),
                'witnesses':[rows[j]['index'] for j in witnesses],'cosines':values,'pair_minimum':pair_min})
        alternatives.sort(key=lambda a:a['median'],reverse=True)
        if alternatives:
            entry['alternative']=alternatives[0]
            coherent=[a for a in alternatives if a['minimum']>=POLICY['alternative_witness_min_cosine']
                and a['pair_minimum']>=POLICY['minimum_witness_pair_cosine']]
            best=coherent[0] if coherent else None
            unique=len(coherent)==1 or (len(coherent)>1 and coherent[0]['median']-coherent[1]['median']>=POLICY['minimum_median_margin'])
            if (best and unique and own<=POLICY['maximum_own_median_cosine']
                and own_top<=POLICY['maximum_own_top_witness_cosine']
                and best['median']-own>=POLICY['minimum_median_margin']
                and best['median']-own_top>=POLICY['minimum_own_top_witness_margin']):
                entry['alternative']=best
                entry.update(bank_speaker=best['speaker'],action='reassign_reference_bank',reason='three_independent_long_reference_witnesses_contradict_original_bank')
        result.append(entry)
    return result


@local_stage
def prepare(segments,clean,folder,progress=lambda *a:None,*,preserve_speaker_ids=False):
    from render import reference_audio
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,s in enumerate(segments):
        if s.get('speaker_id') in (None,'unknown') or s.get('keep_original'):continue
        path=folder/f'{i:04d}-reference.wav';reference_audio(clean,s,path)
        seconds=sf.info(path).duration
        if seconds<.2:continue
        rows.append({'index':i,'speaker':str(s['speaker_id']),'seconds':seconds,'path':str(path),'speaker_locked':preserve_speaker_ids or bool(s.get('_speaker_locked')),'start':s['start'],'end':s['end']})
    manifest=folder/'embedding-input.json'
    atomic_json(manifest,{'rows':rows,'policy':EMBEDDING_POLICY,'cache':str(ROOT/'work/reference-speaker-embeddings-v1')})
    progress(0,f'참조 묶음 화자 검증 · {len(rows)}개')
    if rows:
        with (folder/'embedding-worker.log').open('w',encoding='utf8') as log,subprocess.Popen(
            [str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'reference_bank_worker.py'),str(manifest),str(folder/'vectors.npy')],
            cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace') as proc:
            for line in proc.stdout:
                log.write(line);log.flush()
                try:event=json.loads(line)
                except ValueError:continue
                if 'done' in event:progress(90*event['done']/max(1,len(rows)),f'참조 화자 임베딩 {event["done"]}/{len(rows)}')
            if proc.wait():raise RuntimeError('Reference speaker embedding failed. See embedding-worker.log.')
        vectors=np.load(folder/'vectors.npy',allow_pickle=False)
        decisions=decide(rows,vectors)
        from acoustic_speakers import discover
        discovery=discover(rows,vectors)
        acoustic={d['index']:d for d in discovery['decisions'] if d['action']=='acoustic_recluster'}
        decisions=[acoustic.get(d['index'],d) for d in decisions]
        donors=select_donors(rows,vectors,decisions,segments)
    else:decisions=[];donors=[];discovery={}
    for item in decisions:
        if item['action'] in ('reassign_reference_bank','acoustic_recluster'):segments[item['index']]['_reference_bank_speaker']=item['bank_speaker']
    report={'policy':POLICY,'preserve_speaker_ids':preserve_speaker_ids,'acoustic_discovery':discovery,'decisions':decisions,'changes':[d for d in decisions if d['action']!='keep'],
        'donor_policy':DONOR_POLICY,'donor_selection':donors}
    atomic_json(folder/'reference-bank-audit.json',report)
    progress(100,f'참조 묶음 화자 검증 완료 · 혼입 재배정 {len(report["changes"])}개')
    return report
