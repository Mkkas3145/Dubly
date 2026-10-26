"""Split long uploads at quiet/scene boundaries into independently editable videos."""
from pathlib import Path
import json,math,re,subprocess,uuid,shutil
from core import probe,atomic_json

MAX_PART_SECONDS=30*60

def _run(args,timeout=600):
    p=subprocess.run(args,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
    if p.returncode:raise RuntimeError('Media processing failed during automatic video splitting.')
    return p

def _audio_index(info,language):
    from tts_prompt import language_code
    tracks=[s for s in info['streams'] if s.get('codec_type')=='audio']
    wanted=language_code(language)
    matching=[s for s in tracks if wanted and language_code(s.get('tags',{}).get('language'))==wanted]
    return int(next((s for s in matching or tracks if s.get('disposition',{}).get('default')),(matching or tracks)[0]).get('index',0))

def _quiet(source,lo,hi,audio):
    text=_run(['ffmpeg','-nostdin','-hide_banner','-ss',str(lo),'-i',str(source),'-t',str(hi-lo),'-map',f'0:{audio}','-vn','-af','silencedetect=noise=-35dB:d=0.25','-f','null','-']).stderr
    spans=[];start=None
    for match in re.finditer(r'silence_(start|end):\s*([-\d.]+)',text):
        value=max(lo,min(hi,lo+float(match[2])))
        if match[1]=='start':start=value
        elif start is not None:
            if value>start:spans.append((start,value))
            start=None
    if start is not None and hi>start:spans.append((start,hi))
    return spans

def _keys(source,lo,hi):
    text=_run(['ffprobe','-v','error','-read_intervals',f'{lo}%{hi}','-select_streams','v:0','-skip_frame','nokey','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(source)]).stdout
    return sorted({float(f['best_effort_timestamp_time']) for f in json.loads(text).get('frames',[]) if 'best_effort_timestamp_time' in f and lo<=float(f['best_effort_timestamp_time'])<=hi})

def _scenes(source,lo,hi):
    text=_run(['ffmpeg','-nostdin','-hide_banner','-ss',str(lo),'-i',str(source),'-t',str(hi-lo),'-an','-vf','scale=160:-2,select=gt(scene\\,0.35),showinfo','-f','null','-']).stderr
    return [lo+float(v) for v in re.findall(r'pts_time:([\d.]+)',text) if lo<lo+float(v)<hi]

def plan(source,duration,info,language=None,max_seconds=MAX_PART_SECONDS):
    if duration<=max_seconds:return [{'start':0.,'end':duration,'start_key':True,'end_key':True,'boundary':'whole'}]
    # Leave room for codec packet rounding, and balance parts to avoid tiny tails.
    ceiling=max_seconds-min(.5,max_seconds*.02)
    count=math.ceil(duration/ceiling);target=duration/count;audio=_audio_index(info,language)
    cuts=[{'time':0.,'key':True,'method':'start'}]
    for n in range(1,count):
        ideal=n*target;radius=min(120.,target*.15)
        lo=max(cuts[-1]['time']+target*.25,duration-(count-n)*ceiling,ideal-radius)
        hi=min(cuts[-1]['time']+ceiling,duration-(count-n)*target*.25,ideal+radius)
        if hi<=lo:raise RuntimeError('Failed to calculate video split boundaries.')
        keys=_keys(source,lo,hi);quiet=_quiet(source,lo,hi,audio)
        silent_keys=[t for t in keys if any(a+.04<=t<=b-.04 for a,b in quiet)]
        if silent_keys:cut=min(silent_keys,key=lambda t:abs(t-ideal));key=True;method='silence_keyframe'
        elif quiet:
            cut=min(((a+b)/2 for a,b in quiet),key=lambda t:abs(t-ideal));key=False;method='silence'
        else:
            scenes=_scenes(source,lo,hi)
            if scenes:cut=min(scenes,key=lambda t:abs(t-ideal));key=any(abs(t-cut)<.025 for t in keys);method='scene'
            elif keys:cut=min(keys,key=lambda t:abs(t-ideal));key=True;method='keyframe'
            else:cut=(lo+hi)/2;key=False;method='duration_fallback'
        cuts.append({'time':cut,'key':key,'method':method})
    cuts.append({'time':duration,'key':True,'method':'end'})
    return [{'start':a['time'],'end':b['time'],'start_key':a['key'],'end_key':b['key'],'boundary':b['method']} for a,b in zip(cuts,cuts[1:])]

def _write(source,destination,part,max_seconds):
    base=['ffmpeg','-nostdin','-v','error','-y','-ss',str(part['start']),'-i',str(source),'-t',str(part['end']-part['start']),'-map','0','-map_metadata','0','-map_chapters','-1']
    copy=part['start_key'] and part['end_key']
    for mode in (['copy','encode'] if copy else ['encode']):
        codec=['-c','copy'] if mode=='copy' else ['-c','copy','-c:v','libx264','-preset','veryfast','-crf','18','-c:a','aac','-b:a','192k']
        _run(base+codec+['-avoid_negative_ts','make_zero','-movflags','+faststart',str(destination)],timeout=None)
        info=probe(destination);seconds=float(info.get('format',{}).get('duration',0))
        if 0<seconds<=max_seconds:return info,mode
    raise RuntimeError('Split video exceeds the 30-minute limit and was not saved.')

def cleanup(folders,root):
    parent=(Path(root)/'work/videos').resolve()
    for folder in folders:
        folder=Path(folder).resolve()
        if folder.parent!=parent or str(uuid.UUID(folder.name))!=folder.name:raise RuntimeError('Invalid video cleanup path.')
        if folder.exists():shutil.rmtree(folder)

def prepare(source,video_id,filename,info,root,language=None,max_seconds=MAX_PART_SECONDS):
    source=Path(source);duration=float(info['format']['duration'])
    if duration<=max_seconds:return [{'id':video_id,'filename':filename,'folder':source.parent,'info':info}]
    ranges=plan(source,duration,info,language,max_seconds);items=[];folders=[]
    try:
        for n,part in enumerate(ranges,1):
            identifier=video_id if n==1 else str(uuid.uuid4());folder=Path(root)/'work/videos'/identifier
            if n>1:folder.mkdir();folders.append(folder)
            dest=folder/('split-first.mp4' if n==1 else 'source.mp4')
            part_info,mode=_write(source,dest,part,max_seconds)
            expected=len([s for s in info['streams'] if s.get('codec_type')=='audio'])
            if len([s for s in part_info['streams'] if s.get('codec_type')=='audio'])!=expected:raise RuntimeError('An audio track was lost during splitting; the result was not saved.')
            items.append({'id':identifier,'filename':f'{Path(filename).stem} ({n}){Path(filename).suffix}','folder':folder,'info':part_info})
            atomic_json(folder/'split-origin.json',{'group_id':video_id,'original_filename':filename,'original_duration':duration,'part':n,'parts':len(ranges),'start':part['start'],'end':part['end'],'boundary':part['boundary'],'encoding':mode})
        # Commit only after every part has passed validation.
        source.unlink();(source.parent/'split-first.mp4').replace(source)
        group={'ids':[p['id'] for p in items]}
        for p in items:atomic_json(p['folder']/'split-group.json',group)
        return items
    except Exception:
        cleanup(folders,root);(source.parent/'split-first.mp4').unlink(missing_ok=True);raise
