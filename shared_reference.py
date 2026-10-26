"""Target-first same-speaker audio context, with untagged reference text."""
from pathlib import Path
from local_resources import local_stage
import hashlib,json
import numpy as np
import soundfile as sf
from core import SR,atomic_json

POLICY={'version':9,'identity':'authoritative_single_speaker','max_parts':16,
    'remainder':'fill_with_previous_same_speaker_references','short_speaker':'use_all_available_without_repetition',
    'order':'target_first_then_other_references_in_video_order','speaker_tags':'none','inserted_silence_seconds':0,
    'output':'one_translated_dialogue_per_request'}

def plan(segments,timeline):
    groups=[];opened={}
    for index,segment in enumerate(segments):
        voices={t.get('speaker') for t in timeline['turns']
            if min(t['end'],segment['end'])>max(t['start'],segment['start'])}
        voice=next(iter(voices)) if len(voices)==1 else None
        if '_reference_bank_speaker' in segment:voice=segment['_reference_bank_speaker']
        # Fish token-level labels are useful for inspecting boundaries, but
        # frequent flips must not fragment a stable Nemotron voice bank.
        eligible=(voice not in (None,'unknown') and segment.get('original_text','').strip()
            and any(c.isalnum() for c in segment.get('translated_text',''))
            and not segment.get('keep_original'))
        if not eligible:
            groups.append({'speaker':None,'indices':[index]});continue
        group=opened.get(voice)
        if group is None or len(group['indices'])>=POLICY['max_parts']:
            group={'speaker':voice,'indices':[]};groups.append(group);opened[voice]=group
        group['indices'].append(index)
    by_voice={}
    for group in groups:
        if group['speaker'] is not None:by_voice.setdefault(group['speaker'],[]).extend(group['indices'])
    positions={voice:{index:p for p,index in enumerate(indices)} for voice,indices in by_voice.items()}
    for group in groups:
        voice=group['speaker']
        if voice is None:group['reference_indices']=list(group['indices']);continue
        all_indices=by_voice[voice];end=positions[voice][group['indices'][-1]]+1
        group['reference_indices']=all_indices[max(0,end-POLICY['max_parts']):end]
    return groups

def ordered_indices(reference_indices,target_index):
    if target_index not in reference_indices:raise ValueError('Target dialogue is missing from its shared reference')
    return [target_index,*[i for i in reference_indices if i!=target_index]]

@local_stage
def prepare(segments,clean,folder,timeline,progress=lambda *a:None,*,target_first=True,only_indices=None):
    from render import reference_audio
    folder=Path(folder);bank=folder/'shared-references';bank.mkdir(parents=True,exist_ok=True)
    groups=plan(segments,timeline);reports=[]
    policy={**POLICY,'order':POLICY['order'] if target_first else 'source_video_order'}
    if any('_reference_bank_speaker' in s for s in segments):
        from reference_bank_guard import POLICY as GUARD_POLICY
        policy['embedding_guard']=GUARD_POLICY
    if any('_reference_donor_indices' in s for s in segments):
        from reference_bank_guard import DONOR_POLICY
        policy['acoustic_donors']=DONOR_POLICY
    if timeline.get('boundary_authority')=='voxtral_transcribe_3':
        policy['identity']='voxtral_single_speaker_exact_native_times'
    elif timeline.get('boundary_authority')=='mai_transcribe_2':
        policy['identity']='mai_transcribe_2_single_speaker'
    elif timeline.get('boundary_authority')=='fish_transcribe_1_pro':
        policy['identity']='fish_transcribe_1_pro_single_speaker'
    elif timeline.get('boundary_authority')=='qwen_omni_audio':
        policy['identity']='qwen_omni_audio_single_speaker'
    elif timeline.get('boundary_authority')=='fish_qwen_gemini':
        policy['identity']='fish_timing_qwen_single_speaker'
    elif timeline.get('boundary_authority')=='moss_transcribe_diarize':
        policy['identity']='moss_single_speaker_exact_native_times'
    selected=None if only_indices is None else set(only_indices)
    if selected is not None and any(type(i) is not int or not 0<=i<len(segments) for i in selected):
        raise ValueError('Invalid selected reference output index')
    for number,group in enumerate(groups):
        if selected is not None:
            chosen=[i for i in group['indices'] if i in selected]
            if not chosen:continue
            group={**group,'indices':chosen}
        indices=group['indices'];reference_indices=group['reference_indices']
        custom={index:segments[index].get('_reference_donor_indices',reference_indices) for index in indices}
        for index,order in custom.items():
            if (not isinstance(order,list) or index not in order or len(order)>POLICY['max_parts']
                or len(order)!=len(set(order)) or any(type(i) is not int or not 0<=i<len(segments) for i in order)):
                raise ValueError('Invalid acoustically selected reference indices')
            target_voice=segments[index].get('_reference_bank_speaker',segments[index].get('speaker_id'))
            if any(segments[i].get('_reference_bank_speaker',segments[i].get('speaker_id'))!=target_voice for i in order):
                raise ValueError('Acoustic reference contains a different speaker')
        shared=any(len(custom[i])>1 for i in indices)
        if not shared:
            reports.append({**group,'shared':False});continue
        arrays=[];parts=[]
        try:
            pool=sorted({i for order in custom.values() for i in order})
            for index in pool:
                segment=segments[index];path=folder/f'{index:04d}-reference.wav'
                reference_audio(clean,segment,path)
                values,rate=sf.read(path,dtype='int16',always_2d=True)
                if rate!=SR or values.shape[1]!=1 or not len(values):raise RuntimeError('Invalid shared reference audio format.')
                arrays.append(values)
                parts.append({'index':index,'original_text':segment['original_text'],
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        except RuntimeError as error:
            # A bad reference must not contaminate every request for this voice
            # or abort the video. Ordinary individual failure handling remains.
            reports.append({**group,'shared':False,'reason':'unusable_member_reference','error':str(error)})
            continue
        by_index={p['index']:(p,a) for p,a in zip(parts,arrays)}
        requests=[];built={}
        for index in indices:
            chosen=custom[index]
            if len(chosen)<2:continue
            order=ordered_indices(chosen,index) if target_first else list(chosen)
            frames=sum(len(by_index[i][1]) for i in order)
            order_key=tuple(order)
            if order_key not in built:
                reordered=[by_index[i][0] for i in order]
                from rich_transcript import without_speakers
                text='\n'.join(without_speakers(p['original_text']) for p in reordered)
                identity={'policy':policy,'speaker':group['speaker'],'parts':reordered}
                digest=hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                path=bank/f'{digest}.wav';reusable=False
                if path.exists():
                    try:
                        info=sf.info(path)
                        reusable=info.samplerate==SR and info.channels==1 and info.frames==frames
                    except RuntimeError:pass
                if not reusable:
                    temporary=path.with_suffix('.partial.wav')
                    with sf.SoundFile(temporary,'w',samplerate=SR,channels=1,subtype='PCM_16') as writer:
                        for member in order:writer.write(by_index[member][1])
                    temporary.replace(path)
                receipt={'speaker':group['speaker'],'reference_indices':order,'path':str(path),
                    'reference_text':text,'seconds':frames/SR,'identity':identity}
                atomic_json(path.with_suffix('.json'),receipt)
                built[order_key]=receipt
            receipt=built[order_key];path=Path(receipt['path']);text=receipt['reference_text']
            segments[index].update(_shared_reference_path=str(path),_shared_reference_text=text,
                _shared_reference_tagged=False,
                _shared_reference_donor_tag=False,
                _shared_reference_indices=order,_shared_reference_speaker=group['speaker'],
                _guard_reference_texts=[{'original_text':segments[i]['original_text'],
                    'translated_text':segments[i]['translated_text']} for i in order if i!=index])
            requests.append({'target_index':index,'reference_indices':order,'path':str(path)})
        first=built[tuple(requests[0]['reference_indices'])]
        reports.append({**group,'shared':True,'path':first['path'],'reference_text':first['reference_text'],
            'seconds':frames/SR,'requests':requests})
        progress((number+1)/max(1,len(groups))*100,f'같은 화자 참조 준비 {number+1}/{len(groups)}')
    result={'policy':policy,'groups':reports,'shared_groups':sum(r['shared'] for r in reports),
        'individual_outputs':len(segments) if selected is None else len(selected)}
    atomic_json(folder/'shared-reference-plan.json',result)
    return result
