"""Read explicit upload-split provenance; never group videos by their filenames."""
from pathlib import Path
import json, math, uuid


def _identifier(value):
    try:return isinstance(value,str) and str(uuid.UUID(value))==value
    except (ValueError,AttributeError):return False


def _read(path):
    try:
        value=json.loads(path.read_text('utf8'))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):return {}


def origin(video_id,root):
    if not _identifier(video_id):return None
    data=_read(Path(root)/'work/videos'/video_id/'split-origin.json')
    if not _identifier(data.get('group_id')):return None
    part,parts=data.get('part'),data.get('parts')
    if type(part) is not int or type(parts) is not int or not 1<=part<=parts or parts<2:return None
    name=data.get('original_filename')
    if not isinstance(name,str) or not name.strip():return None
    start,end=data.get('start'),data.get('end')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end)) or not 0<=start<end:return None
    return {'group_id':data['group_id'],'title':name,'part':part,'parts':parts,'start':start,'end':end}


def members(video_id,store,root):
    current=origin(video_id,root)
    if not current:return None,[]
    group=_read(Path(root)/'work/videos'/video_id/'split-group.json')
    ids=group.get('ids')
    if not isinstance(ids,list) or video_id not in ids:return None,[]
    result=[];seen=set();numbers=set()
    for identifier in ids:
        if not _identifier(identifier) or identifier in seen:continue
        seen.add(identifier)
        item=origin(identifier,root)
        if not item or item['group_id']!=current['group_id'] or item['parts']!=current['parts'] or item['title']!=current['title']:continue
        job=store.get(identifier)
        if not job or job['state']=='deleting' or item['part'] in numbers:continue
        numbers.add(item['part']);result.append((job,item))
    if not any(job['id']==video_id for job,_ in result):return None,[]
    return current,sorted(result,key=lambda pair:pair[1]['part'])
