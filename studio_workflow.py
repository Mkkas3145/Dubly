"""URL-based editable dubbing workspace; TTS runs only as an explicit queued operation."""
from pathlib import Path
from local_resources import local_stage
import copy,hashlib,json,math,threading,time,traceback
from fastapi import APIRouter,HTTPException,Body
from fastapi.responses import FileResponse,HTMLResponse,Response
import numpy as np
import soundfile as sf
from core import ROOT,SR,atomic_json,config,video_languages,command
from storage import Store
from mix_levels import TTS_GAIN_DB,TTS_GAIN,VOLUME_VERSION
LOCK=threading.RLock()

def paths(video_id,root=ROOT):
    folder=Path(root)/'work/videos'/video_id/'editor'
    return folder,folder/'draft.json',folder/'context.json'

def checkpoint(video_id,transcript,context,root=ROOT):
    folder,draft,ctx=paths(video_id,root);folder.mkdir(parents=True,exist_ok=True)
    if not draft.exists():
        transcript=copy.deepcopy(transcript)
        for row in transcript['segments']:row.setdefault('keep_original',False);row.setdefault('audio_edit',{})
        atomic_json(draft,{'revision':1,'updated':time.time(),'transcript':transcript,'speaker_names':copy.deepcopy(transcript.get('speaker_names',{})),'generated':{},'render_revision':None})
    atomic_json(ctx,context)

def load(video_id,root=ROOT):
    _,draft,ctx=paths(video_id,root)
    if not draft.exists():return None,None
    return json.loads(draft.read_text('utf8')),json.loads(ctx.read_text('utf8'))

def source_waveform(video_id,ctx=None,root=ROOT):
    folder=Path(root)/'work/videos'/video_id
    early=folder/'waveform-source.json'
    source=ctx.get('clean') if ctx else None
    kind='clean'
    if not source and early.exists():
        value=json.loads(early.read_text('utf8'));source=value.get('path');kind=value.get('kind','original')
    if not source:source=str(folder/'original.wav');kind='original'
    path=Path(source)
    if not path.is_file():return None
    try:
        info=sf.info(path);stat=path.stat()
        if not info.frames:return None
    except (RuntimeError,OSError):return None
    key=hashlib.sha256(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:24]
    return {'path':str(path),'kind':kind,'key':key,'duration':info.duration}


def row_key(doc,index):
    rows=doc['transcript']['segments'];row=rows[index]
    bank=[{k:r.get(k) for k in ('start','end','original_text','speaker_id')} for r in rows if r.get('speaker_id')==row.get('speaker_id') and not r.get('_manual_added')]
    return hashlib.sha256(json.dumps({'row':{k:row.get(k) for k in ('start','end','original_text','translated_text','speaker_id','keep_original')},'bank':bank,'language':doc['transcript']['target_language']},sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def audio_processing_policy():
    from rich_transcript import TTS_CUE_POLICY
    from timing import FIT_VERSION
    from batch_tts import ALIGNMENT_VERSION
    from tts_alignment_review import VERSION
    from batch_tts import BATCH_REFERENCE_POLICY,CONTEXT_REFERENCE_POLICY
    from tagged_context_reference import POLICY as TAGGED_CONTEXT_POLICY
    return {'fit':FIT_VERSION,'batch_alignment':ALIGNMENT_VERSION,'audio_review':VERSION,'tts_cues':TTS_CUE_POLICY,
        'batch_generation':BATCH_REFERENCE_POLICY,'individual_reference':CONTEXT_REFERENCE_POLICY,
        'tagged_individual_reference':TAGGED_CONTEXT_POLICY}

def clip_file(video_id,index,root=ROOT):return paths(video_id,root)[0]/'tts'/f'{index:04d}-fitted.wav'
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def generation_input(doc,index):
    generation=doc.get('generated',{}).get(str(index),{})
    if generation.get('input') is not None:return copy.deepcopy(generation['input'])
    if generation.get('input_hash')==row_key(doc,index):
        row=doc['transcript']['segments'][index]
        return {k:row.get(k) for k in ('original_text','translated_text','speaker_id','keep_original')}
    return None


def archive_generation(video_id,index,doc,root=ROOT):
    import shutil
    folder=paths(video_id,root)[0];pieces=folder/'tts';files=[p for p in pieces.glob(f'{index:04d}*') if p.is_file()]
    if not files:return None
    archive=folder/'generation-history'/f'{index:04d}-{time.time_ns()}';archive.mkdir(parents=True)
    for source in files:shutil.copy2(source,archive/source.name)
    base=clip_file(video_id,index,root)
    atomic_json(archive/'editor-state.json',{'created':time.time(),'sha256':digest(base) if base.exists() else None,
        'generation':{**copy.deepcopy(doc.get('generated',{}).get(str(index),{})),'input':generation_input(doc,index)},
        'audio_edit':copy.deepcopy(doc['transcript']['segments'][index].get('audio_edit',{}))})
    return archive


def audio_versions(video_id,index,root=ROOT):
    result=[];history=paths(video_id,root)[0]/'generation-history'
    for folder in sorted(history.glob(f'{index:04d}-*'),reverse=True):
        wav=folder/f'{index:04d}-fitted.wav'
        if not wav.is_file():continue
        state_file=folder/'editor-state.json';state=json.loads(state_file.read_text('utf8')) if state_file.exists() else {}
        sha=state.get('sha256') or digest(wav)
        if any(v['audio_key']==sha for v in result):continue
        result.append({'audio_key':sha,'created':state.get('created',folder.stat().st_mtime),'duration':sf.info(wav).duration,'folder':folder.name})
    return result


def restore_generation(video_id,index,audio_key,root=ROOT):
    import shutil
    doc,ctx=load(video_id,root);row=doc['transcript']['segments'][index];base=clip_file(video_id,index,root)
    if audio_key is not None and (not isinstance(audio_key,str) or len(audio_key)!=64 or any(c not in '0123456789abcdef' for c in audio_key)):raise ValueError('Invalid audio version.')
    current=digest(base) if base.exists() else None;restored_input=generation_input(doc,index)
    if audio_key is None:
        archive_generation(video_id,index,doc,root)
        doc.setdefault('generated',{}).setdefault(str(index),{})['hidden']=True;row['audio_edit']={}
    else:
        versions=audio_versions(video_id,index,root);target=next((v for v in versions if v['audio_key']==audio_key),None)
        if current!=audio_key and target is None:raise ValueError('Previous audio not found.')
        if current!=audio_key:
            archive_generation(video_id,index,doc,root);archive=paths(video_id,root)[0]/'generation-history'/target['folder']
            for source in archive.glob(f'{index:04d}*'):
                if source.is_file():
                    temporary=base.parent/(source.name+'.restore.tmp');shutil.copy2(source,temporary);temporary.replace(base.parent/source.name)
            state_file=archive/'editor-state.json';state=json.loads(state_file.read_text('utf8')) if state_file.exists() else {}
            row['audio_edit']=state.get('audio_edit',{});restored_input=state.get('generation',{}).get('input')
        if digest(base)!=audio_key:raise RuntimeError('Previous audio validation failed.')
        doc.setdefault('generated',{})[str(index)]={'input_hash':row_key(doc,index),'sha256':audio_key,'seconds':sf.info(base).duration,'error':None,'restored':True,'input':restored_input}
    doc['revision']+=1;doc['updated']=time.time();atomic_json(paths(video_id,root)[1],doc)
    return doc,ctx

def peaks(values,count=1000):
    x=np.max(np.abs(values),axis=1) if values.ndim==2 else np.abs(values)
    if not len(x):return []
    hop=max(1,math.ceil(len(x)/count));x=np.pad(x,(0,(-len(x))%hop));return np.max(x.reshape(-1,hop),axis=1).round(5).tolist()

def source_clip(video_id,index,doc,ctx,root=ROOT):
    """Reference listening must follow source range edits, not a stale row filename."""
    from render import reference_audio
    clean=Path(ctx['clean']);stat=clean.stat()
    prepared=editor_segments(doc['transcript'],sf.info(ctx['original']).frames)[index]
    identity={'version':1,'source':str(clean.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
        'lo':prepared['lo'],'hi':prepared['hi'],'reference_lo':prepared.get('reference_lo'),
        'reference_hi':prepared.get('reference_hi'),'fade_in':prepared.get('reference_fade_in',0),
        'fade_out':prepared.get('reference_fade_out',0)}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    folder=paths(video_id,root)[0];destination=folder/f'{index:04d}-source-{key}.wav'
    if not destination.exists():
        with LOCK:
            if not destination.exists():
                temporary=folder/f'{index:04d}-source-{key}.partial.wav'
                reference_audio(clean,prepared,temporary)
                temporary.replace(destination)
    return destination


def listening_audio(video_id,index,doc,context,root=ROOT):
    """Audition the fixed mix gain without modifying base/edited TTS caches."""
    source=edited_audio(video_id,index,doc,context,root)
    key=hashlib.sha256(f'{digest(source)}:{VOLUME_VERSION}:{TTS_GAIN_DB}'.encode()).hexdigest()[:24]
    folder=paths(video_id,root)[0]/'audio-listen';folder.mkdir(exist_ok=True)
    out=folder/f'{index:04d}-{key}.wav'
    with LOCK:
        if not out.exists():
            values,rate=sf.read(source,dtype='float32',always_2d=True)
            temporary=out.with_suffix('.partial.wav')
            sf.write(temporary,values*TTS_GAIN,rate,subtype='FLOAT')
            temporary.replace(out)
    return out


def clip_info(video_id,index,root=ROOT):
    path=clip_file(video_id,index,root)
    if not path.exists():raise HTTPException(404,'Audio has not been generated yet.')
    sha=digest(path);cache=path.with_name(f'{index:04d}-wave-{sha[:16]}-v2.json')
    if cache.exists():return json.loads(cache.read_text('utf8'))
    from timing import acoustic_pause_slots
    values,sr=sf.read(path,dtype='float32',always_2d=True)
    slots=acoustic_pause_slots(values,.08) if len(values)>SR//10 else []
    info={'sha256':sha,'duration':len(values)/sr,'sampleRate':sr,'peaks':peaks(values,1600),'pauses':[{'id':i,'start':float(p['lo']/SR),'end':float(p['hi']/SR),'cut':float(p['cut']/SR)} for i,p in enumerate(slots)]}
    atomic_json(cache,info);return info

def resolve_pending_length(edit,info):
    """Turn a pre-generation range into real pause/speed edits once audio exists."""
    if 'target_duration' not in edit:return edit
    target=edit['target_duration']
    if type(target) not in (int,float) or not math.isfinite(target) or target<.01:raise ValueError('Invalid dialogue duration.')
    length=info['duration'];slots=info['pauses']
    result={'position':edit.get('position'),'audio_sha256':info['sha256'],'speed':1.,'pauses':[]}
    if target>=length:
        if slots:
            total=sum(p['end']-p['start'] for p in slots)
            result['pauses']=[{'id':p['id'],'seconds':(p['end']-p['start'])*(1+(target-length)/total)} for p in slots]
        else:result['pad_end']=target-length
    else:
        capacities=[max(0,p['end']-p['start']-.01) for p in slots];available=sum(capacities)
        shrink=min(length-target,available)
        if available:result['pauses']=[{'id':p['id'],'seconds':max(.01,p['end']-p['start']-shrink*c/available)} for p,c in zip(slots,capacities)]
        result['speed']=max(1.,(length-shrink)/target)
        if result['speed']>8:raise ValueError('The selected interval is too short. Increase its duration.')
    return result


def validate_edit(edit,info,duration):
    if isinstance(edit,dict) and 'target_duration' in edit:
        if set(edit)-{'position','target_duration'}:raise ValueError('Invalid audio edit format.')
        edit=resolve_pending_length(edit,info)
    if not isinstance(edit,dict) or set(edit)-{'trim_start','trim_end','speed','position','pauses','audio_sha256','pad_start','pad_end'}:raise ValueError('Invalid audio edit format.')
    result={}
    for field,default in [('trim_start',0.),('trim_end',None),('speed',1.),('position',None),('pad_start',0.),('pad_end',0.)]:
        value=edit.get(field,default)
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value)):raise ValueError('Invalid audio edit values.')
        result[field]=value
    if not 1<=result['speed']<=8:raise ValueError('Playback speed must be between 1x and 8x.')
    if not all(0<=result[k]<=duration for k in ('pad_start','pad_end')):raise ValueError('Invalid pause duration.')
    end=result['trim_end'] if result['trim_end'] is not None else info['duration']
    if not 0<=result['trim_start']<end<=info['duration']+.001:raise ValueError('Trim interval exceeds generated audio duration.')
    if result['position'] is not None and not 0<=result['position']<duration:raise ValueError('Placement exceeds video duration.')
    if edit.get('audio_sha256',info['sha256'])!=info['sha256']:raise ValueError('Audio was updated. Reopen the waveform.')
    slots={p['id']:p for p in info['pauses']};seen=set();result['pauses']=[]
    for p in edit.get('pauses',[]):
        if not isinstance(p,dict) or set(p)!={'id','seconds'} or type(p['id']) is not int or p['id'] not in slots or p['id'] in seen:raise ValueError('Select a valid pause interval.')
        amount=p['seconds']
        if type(amount) not in (int,float) or not math.isfinite(amount) or not .01<=amount<=duration:raise ValueError('Invalid pause duration.')
        seen.add(p['id']);result['pauses'].append({'id':p['id'],'seconds':float(amount)})
    result['audio_sha256']=info['sha256'];return result

@local_stage
def edited_audio(video_id,index,doc,context,root=ROOT):
    folder=paths(video_id,root)[0];source=clip_file(video_id,index,root);row=doc['transcript']['segments'][index]
    edit=row.get('audio_edit') or {};info=clip_info(video_id,index,root)
    if not edit:return source
    edit=validate_edit(edit,info,context['duration']);key=hashlib.sha256(json.dumps(edit,sort_keys=True).encode()).hexdigest()
    out=folder/'audio-edits'/f'{index:04d}-{key}.wav'
    if out.exists():return out
    out.parent.mkdir(exist_ok=True);values,sr=sf.read(source,dtype='float32',always_2d=True)
    lo=round(edit['trim_start']*SR);hi=round((edit['trim_end'] if edit['trim_end'] is not None else info['duration'])*SR)
    from timing import acoustic_pause_slots,edit_pause_slots
    slots=acoustic_pause_slots(values,.08);selected=[];changes=[]
    for p in edit['pauses']:
        slot=slots[p['id']]
        if not lo<=slot['lo']<slot['hi']<=hi:continue
        selected.append({**slot,'lo':slot['lo']-lo,'hi':slot['hi']-lo,'cut':slot['cut']-lo,'boundary':{}})
        changes.append(round(p['seconds']*SR)-(slot['hi']-slot['lo']))
    values=values[lo:hi]
    if selected:
        ordered=sorted(zip(selected,changes),key=lambda pair:pair[0]['cut'])
        values,_=edit_pause_slots(values,[p for p,c in ordered],[c for p,c in ordered])
    if edit['pad_start'] or edit['pad_end']:
        values=np.pad(values,((round(edit['pad_start']*SR),round(edit['pad_end']*SR)),(0,0)))
    temp=out.with_suffix('.input.wav');sf.write(temp,values,SR,subtype='FLOAT')
    if edit['speed']>1:
        from speedy import stretch
        stretch(temp,out,max(1,round(len(values)/edit['speed'])))
    else:temp.replace(out)
    return out

def public(doc,context,video_id,root=ROOT):
    rows=[];history=paths(video_id,root)[0]/'generation-history';counts={}
    for folder in history.glob('*'):
        prefix=folder.name.split('-')[0]
        if prefix.isdigit() and (folder/(prefix+'-fitted.wav')).is_file():counts[int(prefix)]=counts.get(int(prefix),0)+1
    for i,row in enumerate(doc['transcript']['segments']):
        generation=doc.get('generated',{}).get(str(i),{});available=clip_file(video_id,i,root).exists() and not generation.get('hidden')
        pending=doc.get('_generation_status',{}).get(str(i))=='pending'
        if pending and not generation:available=False
        rows.append({'id':i,**{k:row.get(k) for k in ('start','end','original_text','translated_text','speaker_id','keep_original')},
            'audio_edit':row.get('audio_edit',{}),'deleted':bool(row.get('deleted')),'audio_ready':available,'audio_stale':available and generation.get('input_hash')!=row_key(doc,i),
            'audio_seconds':generation.get('seconds'),'audio_error':generation.get('error'),'audio_key':generation.get('sha256') if available else None,'audio_history_count':counts.get(i,0),'audio_input':generation_input(doc,i),'audio_processing':pending})
    return {'revision':doc['revision'],'updated':doc['updated'],'segments':rows,'duration':context['duration'],'target_language':doc['transcript']['target_language'],
        'render_revision':doc.get('render_revision'),'render_stale':doc.get('render_revision')!=doc['revision'] or doc.get('render_volume_version')!=VOLUME_VERSION,'preview_revision':doc.get('preview_revision'),'player_revision':doc.get('player_revision'),'player_volume_version':doc.get('player_volume_version'),'volume_policy_version':VOLUME_VERSION,'tts_gain_db':TTS_GAIN_DB,'speaker_names':doc.get('speaker_names',{}),'speaker_ids':doc.get('speaker_ids',[]),'has_background':bool(context.get('background'))}

def save(video_id,payload,root=ROOT):
    doc,ctx=load(video_id,root)
    if doc is None:raise ValueError('The script is not ready yet.')
    if type(payload.get('revision')) is not int or payload['revision']!=doc['revision']:raise RuntimeError('The script was updated in another view. Refresh before editing.')
    proposed=payload.get('segments');rows=doc['transcript']['segments']
    if not isinstance(proposed,list) or not len(rows)<=len(proposed)<=10000:raise ValueError('Dialogue counts do not match.')
    for i,p in enumerate(proposed[len(rows):],len(rows)):
        if not isinstance(p,dict) or type(p.get('id')) is not int or p['id']!=i:raise ValueError('Invalid dialogue ID or edit format.')
        start,end=p.get('start'),p.get('end')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end)) or not 0<=start<end<=ctx['duration']:raise ValueError('Dialogue timestamps exceed video duration.')
        rows.append({'start':start,'end':end,'original_text':'','translated_text':None,'speaker_id':'','_manual_added':True,'_speaker_locked':True,'audio_edit':{},'keep_original':False})
    names=payload.get('speaker_names',doc.get('speaker_names',{}))
    if not isinstance(names,dict) or len(names)>500 or any(not isinstance(k,str) or not k.strip() or len(k)>64 or not isinstance(v,str) or len(v)>64 for k,v in names.items()):raise ValueError('Invalid speaker nickname. Maximum length is 64 characters.')
    names={k:v.strip() for k,v in names.items() if v.strip()}
    speaker_ids=payload.get('speaker_ids',doc.get('speaker_ids',[]))
    if not isinstance(speaker_ids,list) or len(speaker_ids)>500 or any(not isinstance(v,str) or not v.strip() or len(v)>64 for v in speaker_ids) or len(set(speaker_ids))!=len(speaker_ids):raise ValueError('Invalid speaker ID.')
    changed=names!=doc.get('speaker_names',{}) or speaker_ids!=doc.get('speaker_ids',[])
    doc['speaker_names']=names;doc['speaker_ids']=speaker_ids
    for i,(p,row) in enumerate(zip(proposed,rows)):
        if not isinstance(p,dict) or type(p.get('id')) is not int or p['id']!=i or set(p)-{'id','original_text','translated_text','speaker_id','keep_original','audio_edit','deleted','start','end'}:raise ValueError('Invalid dialogue ID or edit format.')
        if type(p.get('deleted',False)) is not bool:raise ValueError('Invalid dialogue deletion setting.')
        if any(k in p and (type(p[k]) not in (int,float) or not math.isfinite(p[k]) or abs(p[k]-row[k])>.00001) for k in ('start','end')):raise ValueError('Existing reference timestamps cannot be changed.')
        for field in ['original_text','translated_text','speaker_id']:
            if field not in p:raise ValueError('Required dialogue data is missing.')
        if not isinstance(p['original_text'],str) or not p['original_text'].strip() or len(p['original_text'])>8000:raise ValueError('Invalid source transcript.')
        if p['translated_text'] is not None and (not isinstance(p['translated_text'],str) or len(p['translated_text'])>8000):raise ValueError('Invalid translated dialogue.')
        if not isinstance(p['speaker_id'],str) or not p['speaker_id'].strip() or len(p['speaker_id'])>64:raise ValueError('Invalid speaker ID.')
        if type(p.get('keep_original',False)) is not bool:raise ValueError('Invalid source audio preservation setting.')
        update={k:p[k] for k in ('original_text','translated_text','speaker_id')};update['keep_original']=p.get('keep_original',False) or not bool((p['translated_text'] or '').strip());update['deleted']=p.get('deleted',False)
        edit=p.get('audio_edit',{})
        if edit and not clip_file(video_id,i,root).exists():
            if not isinstance(edit,dict) or set(edit)-{'position','target_duration'}:raise ValueError('Only placement intervals can be edited before audio generation.')
            position=edit.get('position',row['start'])
            if type(position) not in (int,float) or not math.isfinite(position) or not 0<=position<ctx['duration']:raise ValueError('Placement exceeds video duration.')
            update['audio_edit']={'position':float(position)}
            if 'target_duration' in edit:
                length=edit['target_duration']
                if type(length) not in (int,float) or not math.isfinite(length) or length<.01 or position+length>ctx['duration']+.00001:raise ValueError('Dialogue interval exceeds video duration.')
                update['audio_edit']['target_duration']=float(length)
        elif edit:update['audio_edit']=validate_edit(edit,clip_info(video_id,i,root),ctx['duration'])
        else:update['audio_edit']={}
        if any(row.get(k,{} if k=='audio_edit' else False if k in ('keep_original','deleted') else None)!=v for k,v in update.items()):
            if row.get('speaker_id')!=update['speaker_id']:update['_speaker_locked']=True
            if update['translated_text']!=row.get('translated_text'):row.pop('keep_original_reason',None)
            row.update(update);changed=True
    if changed:
        doc['revision']+=1;doc['updated']=time.time();atomic_json(paths(video_id,root)[1],doc)
    return doc,ctx


def editor_segments(transcript,frames):
    """Stable editor indices allow additions/deletions without renaming paid TTS caches."""
    from render import prepare_segments
    segments=[];windows=transcript.get('reference_windows')
    for i,row in enumerate(transcript['segments']):
        single={**transcript,'segments':[row],'reference_windows':[windows[i]] if windows and i<len(windows) else None}
        segment=prepare_segments(single,frames,{})[0]
        segment.update(natural_reference_margin=False,reference_margin_left=segment['lo'],reference_margin_right=segment['hi'])
        if row.get('deleted'):segment['keep_original']=True
        segments.append(segment)
    return segments

def import_published(video_id,job,root=ROOT):
    _,draft,_=paths(video_id,root)
    if draft.exists():return
    output=Path(root)/'outputs'/video_id;source=output/'segments.json';render=output/'render.json'
    if not source.exists() or not render.exists():return
    report=json.loads(render.read_text('utf8'));transcript=json.loads(source.read_text('utf8'))
    from separation import cache_directory
    base=Path(root)/'work/videos'/video_id;stage=cache_directory(base,report.get('separation_model_id','mel-band-roformer-deux'))
    clean=stage/'dialogue.wav';original=base/'original.wav';background=stage/'background.wav'
    if not clean.exists() or not original.exists():return
    checkpoint(video_id,transcript,{'source':str(base/'source.mp4'),'clean':str(clean),'original':str(original),'background':str(background) if background.exists() else None,'duration':sf.info(original).duration,'language':transcript['target_language']},root)
    # Import cached speech without any generation. Never overwrite the published MP4.
    candidates=sorted(stage.glob('full-audio-*/tts-*'),key=lambda p:p.stat().st_mtime,reverse=True)
    if candidates:
        import shutil
        pieces=paths(video_id,root)[0]/'tts';pieces.mkdir(exist_ok=True);doc,ctx=load(video_id,root)
        for i,row in enumerate(transcript['segments']):
            old=candidates[0]/f'{i:04d}-fitted.wav';meta=candidates[0]/f'{i:04d}.json'
            if not old.exists() or not meta.exists():continue
            saved=json.loads(meta.read_text('utf8'))
            if abs(saved.get('start',-10)-row['start'])>.001 or abs(saved.get('end',-10)-row['end'])>.001:continue
            shutil.copy2(old,pieces/old.name);doc['generated'][str(i)]={'input_hash':row_key(doc,i),'seconds':sf.info(old).duration,'sha256':digest(old),'imported':True}
        doc['render_revision']=doc['revision'];doc['render_volume_version']=VOLUME_VERSION;atomic_json(paths(video_id,root)[1],doc)

def apply_verified_speakers(doc,segments,audit):
    """Expose the same verified voice identity used by TTS; preserve user choices."""
    changed=[]
    for decision in audit.get('changes',[]):
        i=decision['index']
        if not 0<=i<len(segments):continue
        row=doc['transcript']['segments'][i]
        if row.get('_speaker_locked') or str(row['speaker_id'])!=str(decision['original_speaker']):continue
        if decision.get('action') not in ('reassign_reference_bank','acoustic_recluster'):continue
        corrected=str(decision['bank_speaker'])
        row['_asr_speaker_id']=row.get('_asr_speaker_id',row['speaker_id'])
        row['speaker_id']=corrected;segments[i]['speaker_id']=corrected
        changed.append(i)
    if changed:doc['revision']+=1
    return changed


def materialize_ungenerated_ranges(video_id,doc,context,indices,root=ROOT):
    """An ungenerated timeline block selects its source reference at generation time."""
    changed=False;transcript=doc['transcript'];windows=transcript.get('reference_windows')
    for i in indices:
        row=transcript['segments'][i];generation=doc.get('generated',{}).get(str(i),{})
        if row.get('deleted') or row.get('keep_original'):continue
        if clip_file(video_id,i,root).exists() and not generation.get('hidden'):continue
        edit=row.get('audio_edit') or {}
        if not edit or not set(edit)<={'position','target_duration'}:continue
        start=edit.get('position',row['start']);length=edit.get('target_duration',row['end']-row['start']);end=start+length
        if not 0<=start<end<=context['duration']+.00001:raise RuntimeError('Reference audio interval exceeds video duration.')
        end=min(end,context['duration'])
        if abs(start-row['start'])<1e-7 and abs(end-row['end'])<1e-7:continue
        row.setdefault('_original_reference_range',{'start':row['start'],'end':row['end']})
        row.update(start=start,end=end,_reference_range_source='ungenerated_timeline_selection')
        if windows is not None and i<len(windows):
            windows[i]={**windows[i],'start':start,'end':end,'start_frame':round(start*SR),'end_frame':round(end*SR),
                'fade_in_frames':0,'fade_out_frames':0}
        changed=True
    if changed:doc['revision']+=1;doc['updated']=time.time()
    return changed


def generate(video_id,action,index=None):
    from render import prepare_segments,synthesize_all,generate_speech,fit_speech,mix,mux
    from reference_bank_guard import prepare as guard
    from shared_reference import prepare as shared
    import shutil
    store=Store();folder=paths(video_id)[0];doc,ctx=load(video_id);job=store.get(video_id)
    if doc is None:store.update(video_id,state='error',phase='실패',error='먼저 대본 분석을 완료해 주세요.');return 1
    from studio_concurrency import commit,draft_lock
    baseline=copy.deepcopy(doc)
    def persist():
        nonlocal baseline
        if index is None:atomic_json(paths(video_id)[1],doc)
        else:baseline=commit(folder,doc,baseline,index)
    resume_file=folder/'resume-completed.json'
    resume_completed=resume_file.exists() and json.loads(resume_file.read_text('utf8')).get('preserve_completed') is True
    resume_file.unlink(missing_ok=True)
    def progress(percent,detail=''):store.update(video_id,phase='영상 재생 준비' if action=='watch' else '음성 생성' if action!='render' else '최종 영상 생성',phase_progress=round(percent,1),progress=round(60+percent*.4,1),detail=detail)
    try:
        from tts_prompt import with_transcript_languages
        cfg=with_transcript_languages(video_languages(config(),job),doc['transcript'])
        if action not in {'mix','watch'}:
            selection=[index] if index is not None else range(len(doc['transcript']['segments']))
            if index is not None and not 0<=index<len(doc['transcript']['segments']):raise RuntimeError('Invalid dialogue ID.')
            if materialize_ungenerated_ranges(video_id,doc,ctx,selection):persist()
        transcript=copy.deepcopy(doc['transcript'])
        segments=editor_segments(transcript,sf.info(ctx['original']).frames)
        pieces=folder/'tts';pieces.mkdir(exist_ok=True)
        if index is not None and not 0<=index<len(segments):raise RuntimeError('Invalid dialogue ID.')
        targets=[index] if index is not None else range(len(segments))
        processing_policy=audio_processing_policy()
        generation_keys={i:row_key(doc,i) for i in targets}
        generation_texts={i:{k:doc['transcript']['segments'][i].get(k) for k in ('original_text','translated_text','speaker_id','keep_original')} for i in targets}
        def record_report(i,report,publish=False):
            if report.get('deleted'):return False
            with LOCK:
                base=clip_file(video_id,i);previous=doc.get('generated',{}).get(str(i),{})
                sha=digest(base);changed=previous.get('sha256')!=sha
                row=doc['transcript']['segments'][i];edit=row.get('audio_edit') or {}
                if 'target_duration' in edit:
                    row['audio_edit']=validate_edit(edit,clip_info(video_id,i),ctx['duration']);doc['revision']+=1
                elif previous.get('sha256') and changed and edit:
                    row['audio_edit']={k:edit[k] for k in ('position',) if k in edit};doc['revision']+=1
                doc.setdefault('generated',{})[str(i)]={'input_hash':generation_keys.get(i,row_key(doc,i)) if index is not None else row_key(doc,i),'sha256':sha,'seconds':sf.info(base).duration,'error':report.get('reason') if report.get('keep_original') else None,'input':generation_input(doc,i) if report.get('cached') else (generation_texts[i] if index is not None else {k:row.get(k) for k in ('original_text','translated_text','speaker_id','keep_original')}),'processing_policy':previous.get('processing_policy') if report.get('cached') else processing_policy}
                doc.get('_generation_status',{}).pop(str(i),None)
                if publish:
                    doc['revision']+=1;doc['updated']=time.time();persist()
                return changed
        needs_generation=index is not None or any(not s.get('keep_original') and not s.get('deleted') and
            (not clip_file(video_id,i).exists() or doc.get('generated',{}).get(str(i),{}).get('hidden') or doc.get('generated',{}).get(str(i),{}).get('input_hash')!=row_key(doc,i) or doc.get('generated',{}).get(str(i),{}).get('error') or (not resume_completed and doc.get('generated',{}).get(str(i),{}).get('processing_policy')!=processing_policy)) for i,s in enumerate(segments) if i in targets)
        needs_generation=needs_generation or any(not s.get('deleted') and not s.get('keep_original') and not s.get('audio_edit') and clip_file(video_id,i).exists() and s['lo']+sf.info(clip_file(video_id,i)).frames>round(ctx['duration']*SR) for i,s in enumerate(segments) if i in targets)
        if action not in {'mix','watch'} and needs_generation:
            doc['_generation_status']={str(i):'pending' for i in targets if not segments[i].get('deleted') and not segments[i].get('keep_original')}
            doc['updated']=time.time();persist()
        embeddings=cfg.get('speaker_embeddings',False)
        needs_discovery=False
        if embeddings:
            from acoustic_speakers import POLICY as ACOUSTIC_POLICY
            audit_path=folder/'reference-bank-guard/reference-bank-audit.json'
            saved_audit=json.loads(audit_path.read_text('utf8')) if audit_path.exists() else {}
            needs_discovery=(saved_audit.get('acoustic_discovery',{}).get('policy')!=ACOUSTIC_POLICY or bool(saved_audit.get('preserve_speaker_ids'))!=bool(transcript.get('preserve_source_speaker_ids')))
        else:
            for segment in segments:
                for key in ('_reference_bank_speaker','_reference_donor_indices'):segment.pop(key,None)
        if action not in {'mix','watch'} and (needs_generation or needs_discovery):
            if embeddings:
                options={'preserve_speaker_ids':True} if transcript.get('preserve_source_speaker_ids') else {}
                audit=guard(segments,Path(ctx['clean']),folder/'reference-bank-guard',lambda p,d:progress(p*.1,d),**options)
                apply_verified_speakers(doc,segments,audit)
            timeline={'boundary_authority':transcript.get('boundary_authority','mai_transcribe_2'),'turns':[{'start':s['start'],'end':s['end'],'speaker':s['speaker_id']} for s in segments]}
            if cfg.get('tts_reference_mode')=='same_speaker_tagged_context':
                from tagged_context_reference import prepare
                prepare(segments,Path(ctx['clean']),pieces,only_indices=[index] if index is not None else None)
            elif cfg.get('tts_reference_mode')=='same_speaker_context':
                from batch_tts import prepare_context_references
                prepare_context_references(segments,Path(ctx['clean']),pieces,only_indices=[index] if index is not None else None)
            elif cfg.get('tts_reference_mode','individual').startswith('same_speaker_shared_'):
                shared(segments,Path(ctx['clean']),pieces,timeline,only_indices=[index] if index is not None else None)
        if action in {'mix','watch'}:
            ids=[];reports=[]
            for i,s in enumerate(segments):
                if s.get('deleted'):continue
                if not clip_file(video_id,i).exists() or doc.get('generated',{}).get(str(i),{}).get('hidden'):raise RuntimeError('Some audio has not been generated. Generate the missing dialogue first.')
                if doc.get('generated',{}).get(str(i),{}).get('error'):s['keep_original']=True
        elif cfg.get('tts_mode')=='same_speaker_batches':
            from batch_tts import synthesize_speaker_batches,plan_speaker_batches,speaker_request_groups
            pending=[];pending_ids=[];by_id={}
            for i,segment in enumerate(segments):
                if segment.get('deleted'):
                    by_id[i]={'index':i,'deleted':True};continue
                prior=doc.get('generated',{}).get(str(i),{})
                within_video=clip_file(video_id,i).exists() and (bool(segment.get('audio_edit')) or segment['lo']+sf.info(clip_file(video_id,i)).frames<=round(ctx['duration']*SR))
                if (index is not None and i!=index) or (index is None and prior.get('input_hash')==row_key(doc,i) and not prior.get('hidden') and not prior.get('error') and (resume_completed or prior.get('processing_policy')==processing_policy) and within_video):
                    by_id[i]={'index':i,'cached':True,'keep_original':bool(prior.get('error')) or bool(segment.get('keep_original')),'reason':prior.get('error')};continue
                if segment.get('keep_original'):
                    if index is not None:raise RuntimeError('Disable source audio preservation and enter translated dialogue.')
                    sf.write(clip_file(video_id,i),np.zeros((1,2),np.float32),SR,subtype='FLOAT')
                    by_id[i]={'index':i,'keep_original':True,'reason':'user_keep_original'};continue
                archive_generation(video_id,i,doc)
                pending_ids.append(i)
            eligible=[i for i,s in enumerate(segments) if not s.get('deleted') and not s.get('keep_original')]
            wanted=set(pending_ids)
            eligible_rows=[segments[i] for i in eligible]
            groups=plan_speaker_batches(eligible_rows)
            requests=speaker_request_groups(eligible_rows,groups)
            selected=[(g,r) for g,r in zip(groups,requests) if wanted.intersection(eligible[j] for j in g)]
            context_ids=sorted({eligible[j] for g,r in selected for j in r})
            local_ids={i:j for j,i in enumerate(context_ids)}
            cfg={**cfg,'_speaker_batch_groups':[[local_ids[eligible[j]] for j in g] for g,r in selected],
                '_speaker_batch_requests':[[local_ids[eligible[j]] for j in r] for g,r in selected]}
            pending=[{**segments[i],'next_dialogue_start_frame':min((s['lo'] for s in segments if not s.get('deleted') and s['lo']>segments[i]['lo']),default=round(ctx['duration']*SR))} for i in context_ids]
            batch_folder=pieces/'speaker-batches'
            if index is not None:cfg={**cfg,'batch_regeneration_nonce':time.time_ns()}
            reports=synthesize_speaker_batches(pending,Path(ctx['clean']),batch_folder,cfg,lambda *a:None,
                status=lambda p,d:progress(10+p*.65,d)) if pending else []
            if len(reports)!=len(context_ids):raise RuntimeError('Batch generation omitted dialogue.')
            for local,(i,report) in enumerate(zip(context_ids,reports)):
                if i not in wanted:continue
                shutil.copy2(batch_folder/f'{local:04d}-fitted.wav',clip_file(video_id,i))
                by_id[i]={**report,'index':i}
                if report.get('keep_original'):segments[i]['keep_original']=True
            ids=[index] if index is not None else list(range(len(segments)));reports=[by_id[i] for i in ids]
        elif index is None:
            from render import pipelined_queue
            def fallback(i,error):
                sf.write(clip_file(video_id,i),np.zeros((1,2),np.float32),SR,subtype='FLOAT')
                return {'index':i,'keep_original':True,'reason':str(error)}
            def prepare(i,segment):
                if segment.get('deleted'):return i,{'index':i,'deleted':True},None
                row={**segment,'next_dialogue_start_frame':min((s['lo'] for s in segments if not s.get('deleted') and s['lo']>segment['lo']),default=round(ctx['duration']*SR))}
                prior=doc.get('generated',{}).get(str(i),{})
                within_video=clip_file(video_id,i).exists() and (bool(segment.get('audio_edit')) or segment['lo']+sf.info(clip_file(video_id,i)).frames<=round(ctx['duration']*SR))
                if prior.get('input_hash')==row_key(doc,i) and not prior.get('hidden') and not prior.get('error') and (resume_completed or prior.get('processing_policy')==processing_policy) and within_video:
                    report={'index':i,'start':row['start'],'end':row['end'],'cached':True,'keep_original':bool(prior.get('error')) or row.get('keep_original',False),'reason':prior.get('error')}
                elif row.get('keep_original'):report=fallback(i,'user_keep_original')
                else:
                    try:
                        archive_generation(video_id,i,doc)
                        return i,None,generate_speech(i,row,Path(ctx['clean']),pieces,cfg)
                    except RuntimeError as error:report=fallback(i,error)
                return i,report,None
            def fit(item):
                i,report,prepared=item
                if report is None:
                    try:report=fit_speech(prepared,cfg)
                    except RuntimeError as error:report=fallback(i,error)
                if report.get('keep_original'):segments[i]['keep_original']=True
                record_report(i,report,publish=True)
                return report
            reports=pipelined_queue(segments,prepare,fit,lambda done,*_:progress(10+done/max(1,len(segments))*65,f'TTS {done}/{len(segments)}'))
            ids=list(range(len(segments)))
        else:
            # A regenerate click makes one new API generation; archive its old cache.
            saved=pieces/f'{index:04d}-generation.json'
            archive=archive_generation(video_id,index,doc)
            if saved.exists():
                saved.rename(archive/'generation-cache.json')
            row={**segments[index],'next_dialogue_start_frame':min((s['lo'] for s in segments if not s.get('deleted') and s['lo']>segments[index]['lo']),default=round(ctx['duration']*SR))}
            if row.get('keep_original'):raise RuntimeError('Disable source audio preservation and enter translated dialogue.')
            reports=[fit_speech(generate_speech(index,row,Path(ctx['clean']),pieces,cfg),cfg)];ids=[index]
        changed_audio=False
        for i,report in zip(ids,reports):
            changed_audio=record_report(i,report) or changed_audio
        doc.pop('_generation_status',None)
        if index is not None or changed_audio:doc['revision']+=1
        doc['updated']=time.time();persist()
        if action in {'render','preview','mix','watch'}:
            final=folder/'render-clips';final.mkdir(exist_ok=True)
            rendered=[]
            for i,s in enumerate(segments):
                if s.get('deleted'):continue
                audio=edited_audio(video_id,i,doc,ctx);shutil.copy2(audio,final/f'{i:04d}-fitted.wav')
                position=doc['transcript']['segments'][i].get('audio_edit',{}).get('position')
                if position is not None:s['output_start_frame']=round(position*SR)
                if s.get('output_start_frame',s['lo'])+sf.info(audio).frames>round(ctx['duration']*SR):raise RuntimeError(f'Audio for dialogue {i+1} extends past the video end. Adjust trim, placement, or speed.')
                rendered.append((i,s,audio))
            rendered.sort(key=lambda item:(item[1]['lo'],item[0]));render_dir=folder/'render-ordered';render_dir.mkdir(exist_ok=True)
            replacements=[]
            for j,(i,s,audio) in enumerate(rendered):
                shutil.copy2(audio,render_dir/f'{j:04d}-fitted.wav')
                if not s.get('keep_original'):
                    start=min(s['lo'],s.get('output_start_frame',s['lo']));end=max(s['hi'],s.get('output_start_frame',s['lo'])+sf.info(audio).frames)
                    replacements.append({'lo':max(0,start-round(.2*SR)),'hi':min(round(ctx['duration']*SR),end+round(.2*SR))})
            from render import replacement_intervals,speech_placements
            try:replacements=replacement_intervals([s for i,s,a in rendered],round(ctx['duration']*SR),speech_placements([s for i,s,a in rendered],render_dir))
            except RuntimeError:
                if not any(s.get('_manual_added') for i,s,a in rendered):raise
            mixed=folder/'mixed.wav';mix(Path(ctx['original']),Path(ctx['clean']),[s for i,s,a in rendered],render_dir,mixed,lambda p,d:progress(75+p*.15,'배경음과 편집 음성 합치는 중'),excluded_intervals=transcript.get('excluded_intervals',[]),background=Path(ctx['background']) if ctx.get('background') else None,replacement_override=sorted(replacements,key=lambda r:r['lo']))
            progress(91,'미리듣기 오디오 저장 중')
            preview=folder/'preview.partial.mp3';command(['ffmpeg','-nostdin','-v','error','-y','-i',str(mixed),'-c:a','libmp3lame','-b:a','320k',str(preview)]);preview.replace(folder/'preview.mp3')
            doc['preview_revision']=doc['revision'];doc['preview_volume_version']=VOLUME_VERSION;doc['updated']=time.time();persist()
            if action=='watch':
                player_dir=folder/'player';player_dir.mkdir(exist_ok=True)
                progress(94,'재생용 영상 준비 중')
                mux(Path(ctx['source']),mixed,player_dir/'video.mp4',cfg['language'],ctx['duration'])
                doc['player_revision']=doc['revision'];doc['player_volume_version']=VOLUME_VERSION;doc['updated']=time.time()
                persist()
            if action=='render':
                output=ROOT/'outputs'/video_id;output.mkdir(exist_ok=True);progress(93,'MP4 저장 중')
                mux(Path(ctx['source']),mixed,output/'final.mp4',cfg['language'],ctx['duration'])
                atomic_json(output/'segments.json',doc['transcript']);atomic_json(output/'editor-render.json',{'revision':doc['revision'],'edits':[s.get('audio_edit',{}) for s in doc['transcript']['segments']],'tts_reports':reports,'duration':ctx['duration']})
                doc['render_revision']=doc['revision'];doc['render_volume_version']=VOLUME_VERSION;persist()
                store.update(video_id,state='done',phase='완료',progress=100,phase_progress=100,detail='편집한 MP4가 준비되었습니다.',error=None)
            else:store.update(video_id,state='ready',phase='편집 준비 완료',progress=100,phase_progress=100,detail='대사와 파형을 편집할 수 있습니다.',error=None)
        else:store.update(video_id,state='ready',phase='편집 준비 완료',progress=100,phase_progress=100,detail='대사와 파형을 편집할 수 있습니다.',error=None)
        return 0
    except Exception as error:
        traceback.print_exc();store.update(video_id,state='error',phase='실패',detail='',error=str(error)[:1000]);return 1
    finally:
        with LOCK,draft_lock(folder):
            latest,_=load(video_id)
            if latest and '_generation_status' in latest:
                latest.pop('_generation_status',None);latest['updated']=time.time();atomic_json(paths(video_id)[1],latest)
        from mfa_hint import close_worker
        close_worker()

def router(api):
    routes=APIRouter()
    def root():return api.ROOT
    def get(video_id):return api.get_video(video_id)
    def document(video_id):
        job=get(video_id)
        with LOCK:import_published(video_id,job,root());doc,ctx=load(video_id,root())
        return job,doc,ctx
    def enqueue(video_id,action,payload,index=None):
        with api.queue.lock,LOCK:
            job,doc,ctx=document(video_id)
            if action!='analyze':
                if doc is None:raise HTTPException(409,'Analyze the script first.')
                if payload.get('confirmed') is not True:raise HTTPException(400,'Generation confirmation is required.')
                if payload.get('revision')!=doc['revision']:raise HTTPException(409,'Save the script before retrying.')
                if index is not None and not 0<=index<len(doc['transcript']['segments']):raise HTTPException(404,'No dialogue was found.')
                if index is not None and doc['transcript']['segments'][index].get('deleted'):raise HTTPException(404,'No dialogue was found.')
            elif doc is not None:return {'ok':True,'ready':True}
            if action=='analyze':
                from audio_tracks import selected
                try:selected(video_id,root())
                except RuntimeError as e:raise HTTPException(409,str(e)) from None
            if action=='tts' and index is not None and (job['state'] in {'running','queued'} or video_id in api.queue.active_jobs):
                from studio_concurrency import enqueue as enqueue_line
                if job.get('workflow_action')!='tts' or job.get('editor_index') is None:raise HTTPException(409,'Wait for full processing to finish before regenerating dialogue.')
                added=enqueue_line(paths(video_id,root())[0],index,job.get('editor_index'))
                return {'ok':True,'queued':added,'duplicate':not added}
            try:api.store.schedule_editor(video_id,action,index)
            except RuntimeError as e:raise HTTPException(409,str(e)) from None
            api.queue.start(editor_only=True,automatic=not (payload.get('confirmed') is True and api.queue.failure_id==video_id))
        return {'ok':True}
    @routes.get('/videos/{video_id}/edit')
    @routes.get('/videos/{video_id}/loading')
    def editor(video_id:str):
        get(video_id);return FileResponse(root()/'static/studio.html',headers={'Cache-Control':'no-cache'})
    @routes.get('/videos/{video_id}/watch')
    def watch_page(video_id:str):
        get(video_id)
        return FileResponse(root()/'static/studio-player.html',headers={'Cache-Control':'no-cache'})
    @routes.post('/api/studio/{video_id}/watch')
    def prepare_watch(video_id:str,payload:dict=Body(...)):
        job,doc,ctx=document(video_id)
        if doc and payload.get('revision')==doc['revision'] and doc.get('player_revision')==doc['revision'] and doc.get('player_volume_version')==VOLUME_VERSION:
            from media_output import resolve
            if resolve(paths(video_id,root())[0]/'player/video.mp4').exists():return {'ok':True,'ready':True}
        return enqueue(video_id,'watch',payload)
    @routes.get('/api/studio/{video_id}/watch-video')
    def watch_video(video_id:str):
        _,doc,_=document(video_id)
        if not doc or doc.get('player_revision')!=doc['revision'] or doc.get('player_volume_version')!=VOLUME_VERSION:raise HTTPException(409,'Prepare a video using the current edits.')
        from media_output import resolve
        path=resolve(paths(video_id,root())[0]/'player/video.mp4')
        if not path.is_file():raise HTTPException(404,'Playback video is not available yet.')
        return FileResponse(path,media_type='video/mp4',headers={'Cache-Control':'no-cache'})
    @routes.get('/api/studio/{video_id}/playlist')
    def playlist(video_id:str):
        get(video_id)
        from video_playlist import members
        group,items=members(video_id,api.store,root())
        if not group:return {'playlist':None}
        result=[]
        for job,part in items:
            identifier=job['id'];doc,ctx=load(identifier,root())
            if doc:
                rows=[(i,row) for i,row in enumerate(doc['transcript']['segments']) if not row.get('deleted')]
                def ready(i,row):
                    generation=doc.get('generated',{}).get(str(i),{})
                    return row.get('keep_original') or (clip_file(identifier,i,root()).is_file() and not generation.get('hidden') and not (doc.get('_generation_status',{}).get(str(i))=='pending' and not generation))
                playable=bool(rows) and all(ready(i,row) for i,row in rows)
            else:
                from media_output import resolve
                try:playable=resolve(root()/'outputs'/identifier/'final.mp4').is_file()
                except (OSError,ValueError,RuntimeError):playable=False
            result.append({'id':identifier,'filename':job['filename'],'part':part['part'],
                'duration':ctx['duration'] if ctx else part['end']-part['start'],
                'state':job['state'],'phase':job['phase'],'playable':playable,
                'watch_url':f'/videos/{identifier}/watch','edit_url':f'/videos/{identifier}/edit'})
        return {'playlist':{'group_id':group['group_id'],'title':group['title'],'parts':group['parts'],'items':result}}
    @routes.get('/static/{filename}')
    def asset(filename:str):
        if filename not in {'video-playlist.js','studio-player.js','studio-player.css','studio.js','studio-media.js','studio-preview.js','editor-history.js','studio.css','history.js','project-share.js','grid.js','progress.js','i18n.js','locales.js'}:raise HTTPException(404)
        return FileResponse(root()/'static'/filename,headers={'Cache-Control':'no-cache'})
    @routes.post('/api/studio/history')
    def history(payload:dict=Body(...)):
        ids=payload.get('ids',[])
        if not isinstance(ids,list) or len(ids)>200:raise HTTPException(400,'Invalid job ID.')
        result=[]
        for id in dict.fromkeys(x for x in ids if isinstance(x,str)):
            try:job=get(id)
            except HTTPException:continue
            doc,ctx=load(id,root())
            from video_playlist import origin
            split=origin(id,root())
            result.append({k:job.get(k) for k in ['id','filename','state','phase','phase_progress','detail','progress','created','updated','source_language','target_language','detected_source_language','error','resource_state']}|{'duration':ctx['duration'] if ctx else split['end']-split['start'] if split else None,'split':split})
        return {'videos':result}
    @routes.get('/api/studio/{video_id}')
    def snapshot(video_id:str):
        job,doc,ctx=document(video_id)
        wave=source_waveform(video_id,ctx,root())
        from studio_concurrency import pending
        job=dict(job,regenerating_indices=list(dict.fromkeys(([job['editor_index']] if job.get('editor_index') is not None and job['state'] in {'queued','running'} else [])+pending(paths(video_id,root())[0]))))
        return {'video':job,'source_waveform':{k:v for k,v in wave.items() if k!='path'} if wave else None,'document':public(doc,ctx,video_id,root()) if doc else None,'has_output':(root()/'outputs'/video_id/'current-output.json').exists() or (root()/'outputs'/video_id/'final.mp4').exists()}
    @routes.put('/api/studio/{video_id}/draft')
    def update(video_id:str,payload:dict=Body(...)):
        from studio_concurrency import draft_lock
        with api.queue.lock,LOCK,draft_lock(paths(video_id,root())[0]):
            job=get(video_id)
            if job['state']=='deleting' or (job['state'] in {'running','queued'} and not (job.get('workflow_action')=='tts' and job.get('editor_index') is not None)):raise HTTPException(409,'Editing is unavailable during processing.')
            try:doc,ctx=save(video_id,payload,root())
            except ValueError as e:raise HTTPException(400,str(e)) from None
            except RuntimeError as e:raise HTTPException(409,str(e)) from None
        return public(doc,ctx,video_id,root())
    @routes.post('/api/studio/{video_id}/resume')
    def resume(video_id:str):
        with api.queue.lock,LOCK:
            job=get(video_id)
            if job['state']=='uploaded':return enqueue(video_id,'analyze',{})
            if job['state'] in {'error','cancelled'}:
                try:
                    if job.get('workflow_action')=='full':api.store.schedule_editor(video_id,'analyze')
                    else:api.queue.retry(video_id)
                except RuntimeError as e:raise HTTPException(409,str(e)) from None
            elif job['state'] not in {'queued','running'}:return {'ok':True,'ready':True}
            api.queue.start(editor_only=True)
        return {'ok':True}
    @routes.post('/api/studio/{video_id}/analyze')
    def analyze(video_id:str):return enqueue(video_id,'analyze',{})
    @routes.post('/api/studio/{video_id}/generate')
    def render(video_id:str,payload:dict=Body(...)):return enqueue(video_id,'render',payload)
    @routes.post('/api/studio/{video_id}/generate-stale')
    def generate_stale(video_id:str,payload:dict=Body(...)):return enqueue(video_id,'tts',payload)
    @routes.post('/api/studio/{video_id}/preview')
    def preview(video_id:str,payload:dict=Body(...)):return enqueue(video_id,'mix',payload)
    @routes.get('/api/studio/{video_id}/preview-audio')
    def preview_audio(video_id:str):
        get(video_id);path=paths(video_id,root())[0]/'preview.mp3'
        if not path.exists():raise HTTPException(404,'Audio preview is not available yet.')
        return FileResponse(path,media_type='audio/mpeg',headers={'Cache-Control':'no-cache'})
    @routes.post('/api/studio/{video_id}/segments/{index}/generate')
    def segment(video_id:str,index:int,payload:dict=Body(...)):return enqueue(video_id,'tts',payload,index)
    @routes.get('/api/studio/{video_id}/segments/{index}/versions')
    def versions(video_id:str,index:int):
        _,doc,_=document(video_id)
        if doc is None or not 0<=index<len(doc['transcript']['segments']):raise HTTPException(404,'No dialogue was found.')
        return {'versions':[{k:v for k,v in item.items() if k!='folder'} for item in audio_versions(video_id,index,root())]}
    @routes.post('/api/studio/{video_id}/segments/{index}/restore')
    def restore(video_id:str,index:int,payload:dict=Body(...)):
        with api.queue.lock,LOCK:
            job,doc,ctx=document(video_id)
            if job['state'] in {'running','queued','deleting'}:raise HTTPException(409,'Editing is unavailable during processing.')
            if doc is None or not 0<=index<len(doc['transcript']['segments']):raise HTTPException(404,'No dialogue was found.')
            if payload.get('revision')!=doc['revision']:raise HTTPException(409,'Save the script before retrying.')
            if 'audio_key' not in payload:raise HTTPException(400,'Invalid audio version.')
            try:doc,ctx=restore_generation(video_id,index,payload['audio_key'],root())
            except ValueError as e:raise HTTPException(400,str(e)) from None
        return public(doc,ctx,video_id,root())
    @routes.get('/api/studio/{video_id}/source')
    def source(video_id:str):
        get(video_id)
        preview=root()/'outputs/speed-comparison/browser-previews'/video_id/'video.mp4'
        source=preview if preview.exists() else root()/'work/videos'/video_id/'source.mp4'
        return FileResponse(source,media_type='video/mp4',headers={'Cache-Control':'no-cache'})
    @routes.get('/api/studio/{video_id}/preview-source')
    def preview_source(video_id:str,kind:str,start:float,end:float):
        _,doc,ctx=document(video_id)
        if not ctx:raise HTTPException(409,'Analysis is in progress.')
        if kind not in {'original','clean','background'} or not ctx.get(kind):raise HTTPException(400,'Unknown audio type.')
        if not math.isfinite(start) or not math.isfinite(end) or not 0<=start<end<=ctx['duration']+.01 or end-start>18:
            raise HTTPException(400,'Invalid waveform query interval.')
        import io
        with sf.SoundFile(ctx[kind]) as audio:
            lo=round(start*audio.samplerate);hi=min(len(audio),round(end*audio.samplerate));audio.seek(lo)
            values=audio.read(max(0,hi-lo),dtype='float32',always_2d=True);rate=audio.samplerate
        buffer=io.BytesIO();sf.write(buffer,values,rate,format='WAV',subtype='FLOAT')
        return Response(buffer.getvalue(),media_type='audio/wav',headers={'Cache-Control':'private, max-age=3600'})
    @routes.get('/api/studio/{video_id}/thumbnail')
    def thumbnail(video_id:str):
        get(video_id);folder=root()/'work/videos'/video_id;path=folder/'thumbnail.jpg'
        if not path.exists():
            try:command(['ffmpeg','-nostdin','-v','error','-y','-i',str(folder/'source.mp4'),'-frames:v','1','-vf','scale=160:-2',str(path)])
            except RuntimeError:raise HTTPException(404) from None
        return FileResponse(path,media_type='image/jpeg')
    @routes.get('/api/studio/{video_id}/waveform')
    def waveform(video_id:str,start:float|None=None,end:float|None=None,points:int=4000):
        job,doc,ctx=document(video_id)
        selected=source_waveform(video_id,ctx,root())
        if not selected:raise HTTPException(409,'Audio is being prepared.')
        duration=selected['duration']
        if start is not None or end is not None:
            if start is None or end is None or not math.isfinite(start) or not math.isfinite(end) or not 0<=start<end<=duration+.01 or end-start>240 or not 100<=points<=24000:
                raise HTTPException(400,'Invalid waveform query interval.')
            with sf.SoundFile(selected['path']) as audio:
                lo=max(0,min(len(audio),math.floor(start*audio.samplerate)))
                hi=max(lo,min(len(audio),math.ceil(end*audio.samplerate)))
                audio.seek(lo)
                values=[];hop=max(1,math.ceil((hi-lo)/points));remaining=hi-lo
                while remaining>0:
                    block=audio.read(min(remaining,hop*500),dtype='float32',always_2d=True)
                    if not len(block):break
                    remaining-=len(block);values.extend(peaks(block,math.ceil(len(block)/hop)))
                return {'start':lo/audio.samplerate,'end':hi/audio.samplerate,'duration':(hi-lo)/audio.samplerate,'peaks':values}
        folder=paths(video_id,root())[0];cache=folder/f"source-waveform-{selected['key']}.json"
        if not cache.exists():
            audio=sf.SoundFile(selected['path']);values=[]
            with audio:
                hop=max(1,math.ceil(len(audio)/12000))
                while True:
                    block=audio.read(hop*500,dtype='float32',always_2d=True)
                    if not len(block):break
                    values.extend(peaks(block,math.ceil(len(block)/hop)))
            atomic_json(cache,{'duration':duration,'peaks':values})
        return {**json.loads(cache.read_text('utf8')),'kind':selected['kind'],'key':selected['key']}
    @routes.get('/api/studio/{video_id}/segments/{index}/waveform')
    def clip_wave(video_id:str,index:int):
        job,doc,ctx=document(video_id)
        if doc is None or not 0<=index<len(doc['transcript']['segments']):raise HTTPException(404)
        info=clip_info(video_id,index,root());row=doc['transcript']['segments'][index]
        with sf.SoundFile(ctx['clean']) as audio:
            lo=round(row['start']*audio.samplerate);hi=min(len(audio),round(row['end']*audio.samplerate));audio.seek(lo)
            values=audio.read(max(0,hi-lo),dtype='float32',always_2d=True)
        return {**info,'reference_rms':float(np.sqrt(np.mean(values**2))) if values.size else 0.}
    @routes.get('/api/studio/{video_id}/segments/{index}/audio')
    def clip(video_id:str,index:int,kind:str='edited'):
        job,doc,ctx=document(video_id)
        if doc is None or not 0<=index<len(doc['transcript']['segments']):raise HTTPException(404)
        try:
            if kind=='base':path=clip_file(video_id,index,root())
            elif kind=='edited':path=edited_audio(video_id,index,doc,ctx,root())
            elif kind=='playback':path=listening_audio(video_id,index,doc,ctx,root())
            elif kind=='reference':
                path=source_clip(video_id,index,doc,ctx,root())
            else:raise HTTPException(400,'Unknown audio type.')
        except (ValueError,RuntimeError) as e:raise HTTPException(400,str(e)) from None
        if not path.exists():raise HTTPException(404)
        return FileResponse(path,media_type='audio/wav',headers={'Cache-Control':'no-cache'})
    return routes
