"""Canonical browser MP4, without altering the uploaded file or dubbing tracks."""
from pathlib import Path
import json
from core import ROOT,command,probe,atomic_json

VERSION=2

def prepare(video_id,root=ROOT):
    root=Path(root);source=root/'work/videos'/video_id/'source.mp4'
    folder=root/'outputs/speed-comparison/browser-previews'/video_id
    folder.mkdir(parents=True,exist_ok=True)
    from audio_tracks import selected
    audio_index=selected(video_id,root)
    identity={'audio_stream_index':audio_index,'version':VERSION,'size':source.stat().st_size,'mtime_ns':source.stat().st_mtime_ns}
    target=folder/'video.mp4';manifest=folder/'input.json'
    saved=json.loads(manifest.read_text('utf8')) if manifest.exists() else {}
    if target.exists() and saved.get('source')==identity:return target
    temporary=folder/'video.partial.mp4'
    original=probe(source)
    audio=next(s for i,s in enumerate(original['streams']) if int(s.get('index',i))==audio_index)
    audio_options=[] if audio.get('codec_name') in ('aac','mp3') else ['-c:a','aac','-b:a','192k']
    command(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-map','0:v:0','-map',f'0:{audio_index}',
        '-c','copy',*audio_options,'-movflags','+faststart',str(temporary)])
    original=probe(source);result=probe(temporary)
    before=next(s for s in original['streams'] if s['codec_type']=='video')
    after=next(s for s in result['streams'] if s['codec_type']=='video')
    if before['codec_name']!=after['codec_name'] or abs(float(before['duration'])-float(after['duration']))>.01:
        raise RuntimeError('Preview video validation failed.')
    temporary.replace(target)
    atomic_json(manifest,{'source':identity,'method':'stream_copy_faststart','video_duration':after['duration']})
    return target
