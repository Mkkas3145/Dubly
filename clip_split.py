"""Sample-exact clip splitting with aligned target words and lossless inline cues."""
from pathlib import Path
import copy,json,math,re,time,uuid,shutil
import soundfile as sf
from core import ROOT,SR,GEMINI,atomic_json,config
from rich_transcript import spoken


def split_tagged(text,cut):
    text=text or '';plain=spoken(text);cut=max(0,min(len(plain),cut))
    # Map plain-text character positions without treating [] directions as speech.
    positions=[];tags=list(re.finditer(r'\[[^\]\n]*\]',text));end=0
    for tag in tags:
        positions.extend(range(end,tag.start()));end=tag.end()
    positions.extend(range(end,len(text)))
    while positions and text[positions[0]].isspace():positions.pop(0)
    while positions and text[positions[-1]].isspace():positions.pop()
    raw_cut=positions[cut] if cut<len(positions) else len(text)
    before=text[:raw_cut]
    pending=re.search(r'(?:\s*\[[^\]\n]*\]\s*)+\s*$',before)
    if pending:raw_cut=pending.start()
    left,right=text[:raw_cut].strip(),text[raw_cut:].strip()
    active=list(re.finditer(r'\[[^\]\n]*\]',text[:raw_cut]))
    if right and not right.startswith('[') and active:right=active[-1].group()+' '+right
    # A cue without any spoken text is not an empty speech fragment's dialogue.
    if not spoken(left):left=''
    if not spoken(right):right=''
    return left,right


def word_cuts(text,words):
    plain=spoken(text);cursor=0;parts=[]
    for word in words:
        token=word['word'];start=plain.find(token,cursor)
        if start<0:raise RuntimeError('Aligned words do not match the dialogue being split.')
        end=start+len(token);parts.append((start,end,word));cursor=end
    return plain,parts


def cut_from_audio(text,words,seconds):
    plain,parts=word_cuts(text,words)
    for start,end,word in parts:
        if seconds<(word['start']+word['end'])/2:return start
    return len(plain)


def source_cut_index(source,left,right,language,key):
    plain=spoken(source)
    if not spoken(left):return 0
    if not spoken(right):return len(plain)
    from speech import http_json
    schema={'type':'json_schema','json_schema':{'name':'source_text_cut','strict':True,'schema':{
        'type':'object','properties':{'cut':{'type':'integer','minimum':0,'maximum':len(plain)}},'required':['cut'],'additionalProperties':False}}}
    payload={'model':GEMINI,'max_tokens':1024,'reasoning':{'effort':'low'},'response_format':schema,
        'messages':[{'role':'system','content':'Choose one character index in ORIGINAL that divides it into the meanings of LEFT and RIGHT translations. Preserve order; choose a natural word/phrase boundary when possible. Return only {"cut":integer}. Never rewrite or add any text.'},
            {'role':'user','content':json.dumps({'ORIGINAL':plain,'LEFT':spoken(left),'RIGHT':spoken(right),'translation_language':language},ensure_ascii=False)}]}
    def parse(value):
        value=json.loads(value);index=value.get('cut')
        if type(index) is not int or not 0<=index<=len(plain):raise ValueError('Invalid source split character index.')
        return index
    index,_=http_json(payload,key,parser=parse)
    return index


def text_plan(row,context,playback,seconds,folder,cfg):
    from mms_alignment import align_batch
    from local_resources import local_resources
    source_language=context.get('source_language') or next(iter(context.get('source_languages') or []),None)
    source_language=source_language or row.get('_source_language')
    if not source_language:raise RuntimeError('Source language is required to align split dialogue text.')
    source=Path(folder)/'source.wav'
    with sf.SoundFile(context['clean']) as audio:
        rate=audio.samplerate;audio.seek(round(row['start']*rate))
        values=audio.read(round((row['end']-row['start'])*rate),dtype='float32',always_2d=True)
    sf.write(source,values,rate,subtype='FLOAT')
    jobs=[]
    for name,audio,text in [('target',playback,row['translated_text']),('source',source,row['original_text'])]:
        if not spoken(text):continue
        jobs.append({'audio':str(Path(audio).resolve()),'texts':[spoken(text)],'cache':str((Path(folder)/(name+'.json')).resolve()),
                     'identity':{'split':1,'text':spoken(text)},'language':cfg['language'] if name=='target' else source_language,'name':name})
    # A single aligner load handles the two languages, with per-job tokenization.
    with local_resources():result=align_batch(jobs,cfg['language'],folder)
    aligned={job['name']:result['words'] for job,result in zip(jobs,result['results'])}
    target_cut=cut_from_audio(row['translated_text'],aligned.get('target',[]),seconds)
    left,right=split_tagged(row['translated_text'],target_cut)
    original_cut=source_cut_index(row['original_text'],left,right,cfg['language'],cfg['key'])
    original_left,original_right=split_tagged(row['original_text'],original_cut)
    plain,parts=word_cuts(row['original_text'],aligned.get('source',[]))
    before=[w for a,b,w in parts if b<=original_cut];after=[w for a,b,w in parts if a>=original_cut]
    if before and after:relative=(before[-1]['end']+after[0]['start'])/2
    elif before:relative=before[-1]['end']
    elif after:relative=after[0]['start']
    else:relative=0.
    source_time=max(row['start']+1/SR,min(row['end']-1/SR,row['start']+relative))
    return {'original':[original_left,original_right],'translated':[left,right],'source_time':source_time}


def split_clip(video,index,payload,root=ROOT):
    from studio_workflow import paths,load,edited_audio,clip_file,archive_generation,row_key,digest,generation_input,public,audio_processing_policy
    from studio_concurrency import draft_lock
    folder,draft,_=paths(video,root)
    with draft_lock(folder):
        doc,ctx=load(video,root)
        if doc is None or not 0<=index<len(doc['transcript']['segments']):raise ValueError('Dialogue not found.')
        if payload.get('revision')!=doc['revision']:raise RuntimeError('The script changed. Save and retry splitting.')
        row=copy.deepcopy(doc['transcript']['segments'][index]);revision=doc['revision']
        if row.get('deleted') or row.get('keep_original'):raise ValueError('Select a generated dialogue clip.')
        source=edited_audio(video,index,doc,ctx,root)
        snapshot=copy.deepcopy(doc)
    cut=payload.get('time')
    if type(cut) not in (int,float) or not math.isfinite(cut):raise ValueError('Invalid split position.')
    values,rate=sf.read(source,dtype='float32',always_2d=True)
    position=(row.get('audio_edit') or {}).get('position',row['start'])
    frame=round((cut-position)*rate)
    if not 0<frame<len(values):raise ValueError('Click inside the audio clip to split it.')
    work=folder/'splits'/uuid.uuid4().hex;work.mkdir(parents=True)
    cfg=config();cfg['language']=ctx['language']
    context={**ctx,'source_languages':doc['transcript'].get('source_languages',[])}
    plan=text_plan(row,context,source,frame/rate,work,cfg)
    staged=[]
    for n,part in enumerate([values[:frame],values[frame:]]):
        path=work/f'{n}.wav';sf.write(path,part,rate,subtype='FLOAT');staged.append(path)
    with draft_lock(folder):
        doc,ctx=load(video,root)
        if doc['revision']!=revision:raise RuntimeError('The script changed during splitting. No clips were replaced; save and retry.')
        backup=copy.deepcopy(doc);right_index=len(doc['transcript']['segments'])
        archive_generation(video,index,doc,root)
        left=copy.deepcopy(row);right=copy.deepcopy(row)
        original_range=[row['start'],row['end']];split_range=[row['start'],plan['source_time'],row['end']]
        for n,part in enumerate([left,right]):
            part.update(start=split_range[n],end=split_range[n+1],original_text=plan['original'][n],translated_text=plan['translated'][n],
                audio_edit={'position':position if n==0 else position+frame/rate},_split_from=index,_source_range_history=[*row.get('_source_range_history',[]),original_range,[split_range[n],split_range[n+1]]])
            part.pop('_manual_added',None)
        rows=doc['transcript']['segments'];windows=doc['transcript'].get('reference_windows')
        if windows is None:windows=[{'start_frame':round(r['start']*SR),'end_frame':round(r['end']*SR),'fade_in_frames':0,'fade_out_frames':0} for r in rows]
        rows[index]=left;rows.append(right)
        def window(r):return {'start':r['start'],'end':r['end'],'start_frame':round(r['start']*SR),'end_frame':round(r['end']*SR),'fade_in_frames':0,'fade_out_frames':0}
        windows[index]=window(left);windows.append(window(right));doc['transcript']['reference_windows']=windows
        for n,i in enumerate([index,right_index]):
            destination=clip_file(video,i,root);temporary=destination.with_suffix('.split.tmp');shutil.copy2(staged[n],temporary);temporary.replace(destination)
            part=rows[i]
            doc.setdefault('generated',{})[str(i)]={'sha256':digest(destination),'seconds':sf.info(destination).duration,'input_hash':row_key(doc,i),
                'input':{k:part.get(k) for k in ('original_text','translated_text','speaker_id','keep_original')},'split_from':index,'edge_fades':{'start':n==0,'end':n==1},'error':None,'processing_policy':audio_processing_policy()}
        # Splitting one clip must not make all same-speaker generations stale.
        for i in range(len(backup['transcript']['segments'])):
            prior=backup.get('generated',{}).get(str(i))
            if i!=index and prior and prior.get('input_hash')==row_key(backup,i):doc['generated'][str(i)]['input_hash']=row_key(doc,i)
        doc['revision']+=1;doc['updated']=time.time();doc['render_revision']=None;doc['preview_revision']=None
        atomic_json(work/'before.json',backup);atomic_json(work/'split.json',{'index':index,'right_index':right_index,'time':position+frame/rate,'plan':plan})
        atomic_json(draft,doc)
    return {'document':public(doc,ctx,video,root),'left':index,'right':right_index,'time':position+frame/rate}
