"""Nemotron-only overlapping windows and shared multi-candidate context verification."""
from pathlib import Path
import sys,json,time,math,argparse,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/nemotron-transformers'))
sys.stdout.reconfigure(encoding='utf8')

def run(source,folder,*,batch_size=2,reuse_scene_results=True):
    import numpy as np,soundfile as sf,torch
    from transformers import AutoProcessor,AutoModelForAudioFrameClassification
    from core import atomic_json,command
    from nemotron_packed_policy import identity_for
    from nemotron_policy import WINDOW_POLICY
    from nemotron_linking import link
    from nemotron_samples import contiguous_samples
    from nemotron_packed_linking import verify,POLICY as PROBE_POLICY
    source,folder=Path(source),Path(folder);folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();torch.set_num_threads(4)
    def event(stage,**values):print(json.dumps({'stage':stage,**values},ensure_ascii=False),flush=True)
    identity=identity_for(source,ROOT);final=folder/'timeline.json'
    if batch_size!=2 or not reuse_scene_results:
        identity={**identity,'validation_execution':{'batch_size':batch_size,'reuse_scene_results':reuse_scene_results}}
    if final.exists() and json.loads(final.read_text('utf8')).get('identity')==identity:
        event('cached');return
    event('load',detail='Nemotron 단일 모델 준비 · 외부 음색 임베딩 없음')
    model_path=ROOT/'models/nemotron-3-diarization';device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(min(1.,4*1024**3/torch.cuda.get_device_properties(0).total_memory))
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForAudioFrameClassification.from_pretrained(model_path,local_files_only=True).to(device).eval()
    def forward(wave):
        inputs=processor(wave,sampling_rate=16000).to(device,dtype=model.dtype)
        with torch.inference_mode():logits=model(**inputs).logits
        return logits[0,:math.ceil(len(wave)/160)].float().sigmoid().cpu().numpy()
    converted=folder/f'{identity["source_sha256"][:16]}-16k.wav'
    if not converted.exists():command(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-ac','1','-ar','16000','-c:a','pcm_s16le',str(converted)])
    audio,rate=sf.read(converted,dtype='float32');assert rate==16000
    duration=len(audio)/rate;hop=160;total=math.ceil(len(audio)/hop);audio=np.pad(audio,(0,total*hop-len(audio)))
    window_frames=round(WINDOW_POLICY['window_seconds']*100);overlap_frames=round(WINDOW_POLICY['overlap_seconds']*100)
    minimum=round(WINDOW_POLICY['minimum_adaptive_seconds']*100);capacity=int(model.config.head_config.num_speakers)
    assets=folder/'native-windows';assets.mkdir(exist_ok=True)
    windows=[];warnings=[];seen=set()
    def analyze(lo,hi):
        if (lo,hi) in seen:return
        seen.add((lo,hi));stem=f'{lo:08d}-{hi:08d}';cache=assets/f'{stem}.npz';metadata=assets/f'{stem}.json'
        old=json.loads(metadata.read_text('utf8')) if metadata.exists() else {}
        if cache.exists() and old.get('identity')==identity:
            with np.load(cache,allow_pickle=False) as saved:active=saved['active']
            examples={int(k):v for k,v in old['examples'].items()}
        else:
            wave=audio[lo*hop:hi*hop];probs=forward(wave)
            if len(probs)!=hi-lo:raise RuntimeError('Nemotron frame duration differs from source audio.')
            active=probs>WINDOW_POLICY['activity_threshold']
            examples={speaker:[{'start':lo*hop+a,'end':lo*hop+b} for a,b in spans]
                for speaker,spans in contiguous_samples(wave,probs,active,rate,hop,WINDOW_POLICY).items()}
            np.savez_compressed(cache,active=active)
            atomic_json(metadata,{'identity':identity,'examples':examples})
        count=int(active.any(0).sum())
        event('window',start=lo*.01,end=hi*.01,local_speakers=count,fraction=min(1.,hi/max(1,total)))
        if count>=min(capacity,WINDOW_POLICY['split_at_local_speaker_count']) and hi-lo>minimum:
            overlap=min(overlap_frames,(hi-lo)//3);mid=(lo+hi)//2
            event('capacity_split',start=lo*.01,end=hi*.01)
            analyze(lo,min(hi,mid+(overlap+1)//2));analyze(max(lo,mid-overlap//2),hi);return
        if count>=capacity:warnings.append({'start':lo*.01,'end':hi*.01,'reason':'native_eight_slots_still_occupied'})
        windows.append({'start_frame':lo,'end_frame':hi,'active':active,'examples':examples,
            'identity_unit_gap_seconds':PROBE_POLICY['identity_unit_gap_seconds'],
            'identity_different_to_same_vote_ratio':PROBE_POLICY['identity_different_to_same_vote_ratio']})
    lo=0
    while lo<total:
        hi=min(total,lo+window_frames);analyze(lo,hi)
        if hi==total:break
        lo+=window_frames-overlap_frames
    windows.sort(key=lambda w:(w['start_frame'],w['end_frame']))
    event('linking',windows=len(windows),detail='겹친 동일 발화로 연결 후 Nemotron 자체 참조 재비교')
    def initial_groups(verified_edges=None):
        _,initial=link(windows,total,verified_edges=verified_edges,use_embeddings=False);groups={}
        for node in initial['node_mapping']:
            root=node['component'];wi=node['window'];slot=node['local_speaker']
            group=groups.setdefault(root,{'component':root,'nodes':[],'windows':set(),'examples':[]})
            group['nodes'].append([wi,slot])
            if windows[wi].get('trusted_constraints',True):group['windows'].add(wi)
            group['examples']+=windows[wi]['examples'].get(slot,[])
        # Short views are context, never independent speaker identity evidence.
        # The query masks come exclusively from ownership of the primary source
        # frames, so an auxiliary slot cannot import another person's words.
        owners=np.full(total,-1,np.int32);strength=np.full(total,-1.,float)
        for wi,w in enumerate(windows):
            lo,hi=w['start_frame'],w['end_frame'];frames=np.arange(hi-lo);quality=np.minimum(frames,hi-lo-1-frames)
            take=quality>strength[lo:hi];owners[lo:hi][take]=wi;strength[lo:hi][take]=quality[take]
        roots=sorted(groups);columns={root:i for i,root in enumerate(roots)}
        primary=np.zeros((total,len(roots)),bool)
        for node in initial['node_mapping']:
            wi,slot,root=node['window'],node['local_speaker'],node['component'];w=windows[wi];lo,hi=w['start_frame'],w['end_frame']
            primary[lo:hi,columns[root]]|=(owners[lo:hi]==wi)&w['active'][:,slot]
        contexts=list(windows)
        for group in groups.values():group['comparison_nodes']=list(group['nodes'])
        for view in views:
            lo,hi=view['start_frame'],view['end_frame'];local=primary[lo:hi];present=np.flatnonzero(local.any(0))
            if not 0<len(present)<=8:continue
            index=len(contexts)
            cannot_link={(a,b) for b in range(len(present)) for a in range(b)
                if groups[roots[present[a]]]['windows'] & groups[roots[present[b]]]['windows']}
            contexts.append({'start_frame':lo,'end_frame':hi,'active':local[:,present],'trusted_constraints':True,
                'cannot_link_slots':cannot_link})
            for slot,column in enumerate(present):
                if local[:,column].sum()>=30:groups[roots[column]]['comparison_nodes'].append([index,slot])
        return groups,contexts
    from nemotron_inference import CachedInference,forward_batch
    infer=CachedInference(folder/'native-probes',identity['revision'],model.dtype,
        lambda waves:forward_batch(model,processor,waves,device),batch_size=batch_size,
        source_identity=identity['source_sha256'],on_oom=torch.cuda.empty_cache)
    from nemotron_local_consensus import repair,short_views
    views,view_audit=short_views(windows,audio,infer,event)
    changed,view_repairs=repair(windows,view_audit)
    if changed:
        repaired=[windows[r['window']] for r in view_repairs if r['status']=='refined_by_repeated_native_scene_consensus']
        event('repair',windows=len(repaired))
    atomic_json(folder/'native-local-views.json',{'comparisons':view_audit,'repairs':view_repairs})
    def verify_contexts(groups,contexts):return verify(groups,contexts,audio,infer,event,rate,reuse_scene_results=reuse_scene_results)
    groups,contexts=initial_groups();edges,probe_audit=verify_contexts(groups,contexts)
    changed,repair_audit=repair(windows,probe_audit)
    atomic_json(folder/'local-consensus-audit.json',{'changed':changed,'repairs':repair_audit})
    if changed:
        atomic_json(folder/'reference-probe-before-repair.json',{'comparisons':probe_audit})
        event('repair',windows=sum(r['status']=='refined_by_repeated_native_scene_consensus' for r in repair_audit))
        groups,contexts=initial_groups();edges,probe_audit=verify_contexts(groups,contexts)
    atomic_json(folder/'global-refinement-rounds.json',{'rounds':[],'reason':'shared_multi_candidate_pass_no_iterative_all_pairs'})
    turns,report=link(windows,total,verified_edges=edges,use_embeddings=False)
    for turn in turns:turn['end']=min(duration,turn['end'])
    atomic_json(folder/'reference-probe-audit.json',{'policy':PROBE_POLICY,'comparisons':probe_audit,'verified_edges':edges})
    result={'identity':identity,'duration':duration,'turns':turns,'boundary_authority':'nemotron_eend_vc',
        'speakers':len({t['speaker'] for t in turns}),'linking':report,'capacity_warnings':warnings,'local_consensus_repairs':repair_audit,'native_view_repairs':view_repairs,
        'windows':[{'start':w['start_frame']*.01,'end':w['end_frame']*.01,'speakers':int(w['active'].any(0).sum())} for w in windows],
        'elapsed_seconds':time.monotonic()-started,'model_precision':str(model.dtype),'external_embedding_model':None,
        'inference_execution':{'max_batch_size':infer.batch_size,**infer.stats},
        'reidentification':'same_nemotron_packed_multi_candidate_prefix_suffix_context_consensus',
        'timing':'native_10ms_frames_owned_by_most_central_window','global_speaker_cap':None}
    atomic_json(final,result);event('complete',speakers=result['speakers'],windows=len(windows),seconds=result['elapsed_seconds'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('folder');args=parser.parse_args();run(args.source,args.folder)
