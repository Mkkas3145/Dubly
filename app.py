from __future__ import annotations
import asyncio
from html import escape
from contextlib import asynccontextmanager
from pathlib import Path
import uuid
import math
import json
from typing import Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Form, Request
from fastapi.responses import FileResponse, HTMLResponse,Response
from fastapi.staticfiles import StaticFiles
from core import ROOT, config, probe,LANGUAGE_CATALOG,SOURCE_SELECTIONS,TARGET_LANGUAGES
from storage import Store
from queue_worker import VideoQueue
from separation import model_ready
from stretch import ready as speed_ready

store=Store()
queue=VideoQueue(store)
for directory in ['work/videos','outputs','outputs/speed-comparison','logs']:(ROOT/directory).mkdir(parents=True,exist_ok=True)


@asynccontextmanager
async def lifespan(app):
    store.recover()
    yield
    queue.stop()


app=FastAPI(title='Dubly',lifespan=lifespan)
MAX_UPLOAD_BYTES=10*1024**3
from upload_parts import MAX_PART_SECONDS
app.mount('/samples/speed',StaticFiles(directory=ROOT/'outputs/speed-comparison',html=True),name='speed-samples')
@app.get('/')
def index():return FileResponse(ROOT/'static/index.html',headers={'Cache-Control':'no-cache'})


@app.get('/static/app.js')
def script():return FileResponse(ROOT/'static/app.js',media_type='text/javascript',headers={'Cache-Control':'no-cache'})


@app.get('/static/languages.js')
def language_catalog():
    return Response('window.DublyLanguages='+json.dumps(LANGUAGE_CATALOG,ensure_ascii=False)+';',media_type='text/javascript',headers={'Cache-Control':'no-cache'})


@app.get('/static/pretendard.woff2')
def pretendard():return FileResponse(ROOT/'assets/fonts/PretendardVariable.woff2',media_type='font/woff2',headers={'Cache-Control':'public, max-age=31536000, immutable'})


@app.get('/static/tts-presets.js')
def tts_presets_catalog():
    from tts_presets import PRESETS,DEFAULT
    return Response('window.DublyTTSPresets='+json.dumps({'presets':PRESETS,'default':DEFAULT},ensure_ascii=False)+';',media_type='text/javascript',headers={'Cache-Control':'no-cache'})


@app.get('/api/state')
def state():
    return {'running':queue.running,'stopping':queue.stop_requested and queue.running,
            'key_ready':bool(config()['key']),'model_ready':model_ready(config()['separator']),
            'videos':store.list()}


@app.post('/api/videos',status_code=201)
async def upload(request:Request,file:UploadFile=File(...),source_language:str|None=Form(None),
                 target_language:str|None=Form(None),tts_preset:str=Form('creative')):
    from tts_presets import PRESETS
    if tts_preset not in PRESETS:
        await file.close()
        raise HTTPException(422,'Unsupported TTS configuration.')
    form=await request.form()
    if len(form.getlist('file'))!=1:
        await file.close()
        raise HTTPException(400,'Upload one video at a time.')
    if (source_language is None)!=(target_language is None):
        await file.close()
        raise HTTPException(400,'Select both source and target languages.')
    if source_language is not None and (source_language not in SOURCE_SELECTIONS or target_language not in TARGET_LANGUAGES):
        await file.close()
        raise HTTPException(422,'Unsupported language.')
    if source_language is not None and source_language==target_language:
        await file.close()
        raise HTTPException(400,'Source and target languages must differ.')
    filename=Path(file.filename or '').name
    if Path(filename).suffix.lower()!='.mp4':raise HTTPException(400,'Select an MP4 file.')
    video_id=str(uuid.uuid4())
    folder=ROOT/'work/videos'/video_id
    folder.mkdir()
    source=folder/'source.mp4'
    size=0;upload_folders=[folder]
    try:
        with source.open('wb') as target:
            while block:=await file.read(4*1024*1024):
                size+=len(block)
                if size>MAX_UPLOAD_BYTES:raise HTTPException(413,'Each file must be 10 GB or smaller.')
                await asyncio.to_thread(target.write,block)
        info=await asyncio.to_thread(probe,source)
        kinds={s['codec_type'] for s in info.get('streams',[])}
        if not {'audio','video'}<=kinds:raise HTTPException(400,'Both video and audio tracks are required.')
        duration=float(info.get('format',{}).get('duration',0))
        if not math.isfinite(duration) or duration<=0:raise HTTPException(400,'Cannot determine video duration.')
        from upload_parts import prepare
        from audio_tracks import initialize
        parts=await asyncio.to_thread(prepare,source,video_id,filename,info,ROOT,source_language,MAX_PART_SECONDS)
        upload_folders=[part['folder'] for part in parts]
        if await request.is_disconnected():raise HTTPException(499,'Upload cancelled.')
        track_info=[initialize(part['id'],part['info'],ROOT) for part in parts]
        records=store.add_uploaded(parts,source_language=source_language,target_language=target_language,tts_preset=tts_preset)
        results=[{**record,'editor_url':f'/videos/{record["id"]}/edit','audio_tracks':tracks} for record,tracks in zip(records,track_info)]
        return {**results[0],'videos':results,'split':len(results)>1}
    except Exception as error:
        from upload_parts import cleanup
        cleanup(upload_folders,ROOT)
        if isinstance(error,HTTPException):raise
        if isinstance(error,RuntimeError):raise HTTPException(400,str(error)) from None
        raise HTTPException(400,'Failed to read the MP4 file.') from None
    finally:
        await file.close()


@app.get('/api/videos/{video_id}/audio-tracks')
def list_audio_tracks(video_id:str):
    get_video(video_id)
    from audio_tracks import inspect
    return inspect(video_id,ROOT)


@app.get('/api/videos/{video_id}/audio-track-preview')
def audio_track_preview(video_id:str,index:int):
    get_video(video_id)
    from audio_tracks import inspect
    data=inspect(video_id,ROOT)
    if index not in [r['index'] for r in data['tracks']]:raise HTTPException(422,'Audio track not found.')
    folder=ROOT/'work/videos'/video_id;path=folder/f'audio-track-{index}-sample.mp3'
    if not path.exists():
        from core import command
        from studio_workflow import LOCK
        with LOCK:
            if not path.exists():
                temp=folder/f'audio-track-{index}-sample.partial.mp3'
                command(['ffmpeg','-nostdin','-v','error','-y','-i',str(folder/'source.mp4'),'-map',f'0:{index}','-t','20','-vn','-c:a','libmp3lame','-b:a','128k',str(temp)])
                temp.replace(path)
    return FileResponse(path,media_type='audio/mpeg',headers={'Cache-Control':'no-cache'})


@app.post('/api/videos/{video_id}/audio-track')
def choose_audio_track(video_id:str,payload:dict=Body(...)):
    with queue.lock:
        job=get_video(video_id)
        folder=ROOT/'work/videos'/video_id
        if job['state']!='uploaded' or (folder/'original.wav').exists() or (folder/'editor/draft.json').exists():
            raise HTTPException(409,'Select the audio track before starting analysis.')
        from audio_tracks import select
        try:
            result=select(video_id,payload.get('index'),ROOT)
            group=ROOT/'work/videos'/video_id/'split-group.json'
            if group.exists():
                from audio_tracks import inspect
                ordinal=next(i for i,t in enumerate(result['tracks']) if t['index']==result['selected'])
                for sibling in json.loads(group.read_text('utf8'))['ids']:
                    sibling_job=store.get(sibling)
                    if sibling==video_id or not sibling_job or sibling_job['state']!='uploaded':continue
                    other=inspect(sibling,ROOT)
                    if other['selection_required'] and ordinal<len(other['tracks']):select(sibling,other['tracks'][ordinal]['index'],ROOT)
            return result
        except ValueError as e:raise HTTPException(422,str(e)) from None


@app.post('/api/start')
def start():
    if not config()['key']:raise HTTPException(400,'OPENROUTER_API_KEY is not configured.')
    if not model_ready(config()['separator']):raise HTTPException(503,'The selected separation model is not installed.')
    if not speed_ready():raise HTTPException(503,'Google Speedy is not installed. Run install_speedy.py.')
    queue.start()
    return {'ok':True}


@app.post('/api/queue/reset')
def reset_queue():
    queue.reset()
    return {'ok':True}


@app.post('/api/pause')
def pause():
    queue.pause()
    return {'ok':True}


@app.post('/api/stop')
def stop(confirmed:Literal[True]=Body(...,embed=True)):
    try:queue.stop()
    except RuntimeError as error:raise HTTPException(409,str(error)) from None
    return {'ok':True}


def get_video(video_id):
    try:
        if str(uuid.UUID(video_id))!=video_id:raise ValueError()
    except ValueError:raise HTTPException(404,'Video not found.')
    job=store.get(video_id)
    if job is None:raise HTTPException(404,'Video not found.')
    return job


@app.post('/api/videos/{video_id}/cancel')
def cancel_video(video_id:str,confirmed:Literal[True]=Body(...,embed=True)):
    get_video(video_id)
    try:state=queue.cancel(video_id)
    except KeyError:raise HTTPException(404,'Video not found.') from None
    except RuntimeError as error:raise HTTPException(409,str(error)) from None
    return {'ok':True,'state':state}


@app.post('/api/videos/{video_id}/retry')
def retry(video_id:str):
    get_video(video_id)
    try:queue.retry(video_id)
    except KeyError:raise HTTPException(404,'Video not found.') from None
    except RuntimeError as error:raise HTTPException(409,str(error)) from None
    return {'ok':True}


@app.delete('/api/videos/{video_id}')
def delete_video(video_id:str,confirmed:Literal[True]=Body(...,embed=True)):
    get_video(video_id)
    try:queue.delete(video_id)
    except KeyError:raise HTTPException(404,'Video not found.') from None
    except (ValueError,RuntimeError) as error:raise HTTPException(409,str(error)) from None
    return {'ok':True}


def result_file(video_id):
    job=get_video(video_id)
    from media_output import resolve
    path=resolve(ROOT/'outputs'/video_id/'final.mp4')
    if not path.is_file():raise HTTPException(404,'Output file not found.')
    return path,Path(job['filename']).stem+'_dubbed.mp4'


@app.get('/videos/{video_id}/player',response_class=HTMLResponse)
def player(video_id:str):
    _,filename=result_file(video_id)
    page=(ROOT/'static/player.html').read_text(encoding='utf-8')
    return HTMLResponse(page.replace('__VIDEO_ID__',video_id).replace('__VIDEO_NAME__',escape(filename)),
        headers={'Cache-Control':'no-cache'})


@app.get('/api/videos/{video_id}/stream')
def stream(video_id:str):
    path,filename=result_file(video_id)
    return FileResponse(path,media_type='video/mp4',filename=filename,content_disposition_type='inline',headers={'Cache-Control':'no-cache'})


@app.get('/api/videos/{video_id}/download')
def download(video_id:str):
    path,filename=result_file(video_id)
    return FileResponse(path,media_type='video/mp4',filename=filename)


import sys
from studio_workflow import router as studio_router
app.include_router(studio_router(sys.modules[__name__]))
from project_sharing import router as project_sharing_router
app.include_router(project_sharing_router(sys.modules[__name__]))
from server_runtime import router as launcher_router
app.include_router(launcher_router())


if __name__=='__main__':
    from server_runtime import run
    run(app)
