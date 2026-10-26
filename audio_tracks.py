"""Persist the chosen input audio stream before any analysis starts."""
from pathlib import Path
import json
from core import ROOT,probe,atomic_json


def tracks(info):
    result=[]
    for ordinal,s in enumerate(info.get('streams',[])):
        if s.get('codec_type')!='audio':continue
        tags=s.get('tags',{});result.append({'index':int(s.get('index',ordinal)),
            'language':tags.get('language','und'),'title':tags.get('title') or tags.get('handler_name') or '',
            'codec':s.get('codec_name',''),'channels':s.get('channels'),
            'default':bool(s.get('disposition',{}).get('default'))})
    return result


def path(video_id,root=ROOT):return Path(root)/'work/videos'/video_id/'audio-tracks.json'


def inspect(video_id,root=ROOT):
    p=path(video_id,root)
    if p.exists():return json.loads(p.read_text('utf8'))
    items=tracks(probe(p.parent/'source.mp4'))
    default=next((r['index'] for r in items if r['default']),items[0]['index'] if items else None)
    return {'tracks':items,'selected':default,'selection_required':False}


def initialize(video_id,info,root=ROOT):
    items=tracks(info);selected=items[0]['index'] if len(items)==1 else None
    data={'tracks':items,'selected':selected,'selection_required':len(items)>1}
    atomic_json(path(video_id,root),data);return data


def select(video_id,index,root=ROOT):
    data=inspect(video_id,root)
    if type(index) is not int or index not in [r['index'] for r in data['tracks']]:raise ValueError('Audio track not found.')
    data.update(selected=index,selection_required=False);atomic_json(path(video_id,root),data);return data


def selected(video_id,root=ROOT):
    data=inspect(video_id,root)
    if data['selected'] is None or data['selection_required']:raise RuntimeError('Select an audio track first.')
    return data['selected']


def extract(source,destination,index,duration):
    from core import command
    destination=Path(destination);temporary=destination.with_name(destination.stem+'.partial.wav')
    command(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-map',f'0:{index}','-vn','-ac','2',
        '-af','aresample=44100:async=1:first_pts=0,apad','-t',str(duration),'-c:a','pcm_f32le','-rf64','auto',str(temporary)])
    temporary.replace(destination)
