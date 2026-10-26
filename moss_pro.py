"""Official MOSS Pro streaming transcription; preserves native speaker boundaries."""
import hashlib,json,time
from pathlib import Path
import requests
import soundfile as sf
from core import atomic_json,command

MODEL='moss-transcribe-diarize-pro'
URL='https://api.mosi.cn/v1/audio/transcriptions'
POLICY={'version':1,'model':MODEL,'endpoint':URL,'diarize':True,'stream':True,
        'input':'whole_separated_dialogue_flac_original_clock','boundary_correction':False}


def keyterms(value):
    if not value:return []
    try:terms=json.loads(value) if isinstance(value,str) else value
    except ValueError:raise RuntimeError('MOSS_KEYTERMS must be a JSON array of strings.') from None
    if (not isinstance(terms,list) or len(terms)>20 or
        any(not isinstance(t,str) or not t.strip() or len(t)>30 for t in terms)):
        raise RuntimeError('MOSS hotwords must contain at most 20 nonempty strings, each up to 30 characters.')
    return list(dict.fromkeys(terms))


def parse_events(lines,raw_path,progress):
    segments=[];text='';done=False;usage={};started=time.monotonic()
    with Path(raw_path).open('w',encoding='utf8') as log:
        for line in lines:
            if not line:continue
            if isinstance(line,bytes):line=line.decode('utf8')
            log.write(line+'\n');log.flush()
            if not line.startswith('data:'):continue
            value=line[5:].strip()
            if value=='[DONE]':continue
            try:event=json.loads(value)
            except ValueError:continue
            kind=event.get('type')
            if kind=='error' or event.get('error'):raise RuntimeError('MOSS Pro returned a streaming error. The raw response was preserved.')
            if kind in ('transcript.segment.done','transcript.text.segment'):
                # The event itself carries the segment in the official schema.
                segments.append({k:event.get(k) for k in ('start','end','speaker','text')})
                progress(min(95,10+len(segments)*.5),f'MOSS Pro 화자·시간 · {len(segments)}개 수신 · {time.monotonic()-started:.0f}초')
            elif kind=='transcript.text.delta':text+=event.get('delta','')
            elif kind=='transcript.text.done':
                text=event.get('text',text);usage=event.get('usage',{});done=True
    if not done:raise RuntimeError('MOSS Pro response ended prematurely. The raw stream and successful caches were preserved.')
    if not segments and text.strip():raise RuntimeError('MOSS Pro returned a transcript without speaker or timing segments. The raw response was preserved.')
    return {'text':text,'segments':segments,'usage':usage}


def source(clean,folder,key,terms,progress=lambda *a:None):
    if not key:raise RuntimeError('MOSS_API_KEY is not configured.')
    clean=Path(clean);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    terms=keyterms(terms)
    signature={'policy':POLICY,'audio_sha256':hashlib.sha256(clean.read_bytes()).hexdigest(),'keyterms':terms}
    run=folder/hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();run.mkdir(exist_ok=True)
    cache=run/'result.json'
    if cache.exists():
        data=json.loads(cache.read_text('utf8'))
        if data.get('signature')==signature:progress(100,'MOSS Pro 화자·시간 캐시 재사용');return {**data,'_cache_reused':True}
    info=sf.info(clean)
    if info.duration>3600:raise RuntimeError('MOSS Pro supports at most 60 minutes per request.')
    audio=run/'input.flac'
    if not audio.exists():
        temp=run/'input.partial.flac'
        command(['ffmpeg','-nostdin','-v','error','-y','-i',str(clean),'-ac','1','-c:a','flac','-sample_fmt','s16',str(temp)])
        temp.replace(audio)
    if audio.stat().st_size>512*1024**2:raise RuntimeError('MOSS Pro input exceeds 512 MB.')
    fields={'model':MODEL,'diarize':'true','stream':'true','response_format':'verbose_json'}
    if terms:fields['keyterms']=json.dumps(terms,ensure_ascii=False)
    atomic_json(run/'request.json',{'endpoint':URL,**fields,'audio_seconds':info.duration,'bytes':audio.stat().st_size})
    progress(5,'MOSS Pro 전체 대사 오디오 요청 · 화자·시간 스트리밍 대기')
    try:
        with audio.open('rb') as f,requests.post(URL,headers={'Authorization':f'Bearer {key}','Accept':'text/event-stream'},
            files={'file':('dialogue.flac',f,'audio/flac')},data=fields,stream=True,timeout=(30,300)) as response:
            if response.status_code!=200:
                atomic_json(run/'http-error.json',{'status':response.status_code,'body':response.text.replace(key,'[REDACTED]')})
                raise RuntimeError(f'MOSS Pro HTTP {response.status_code}. Check account credits and key permissions. The response was preserved.')
            result=parse_events(response.iter_lines(),run/'raw.sse',progress)
    except requests.RequestException:
        raise RuntimeError('MOSS Pro connection was interrupted or timed out. The raw response was preserved.') from None
    data={**result,'signature':signature,'model':MODEL}
    # Validate before caching, retaining the exact native endpoint values.
    from moss_pipeline import regions
    regions(data,info.duration)
    atomic_json(cache,data);progress(100,f'MOSS Pro {len(data["segments"])}개 구간 완료');return data
