"""Optional SRT translation reference. Never a transcript or alignment authority."""
from pathlib import Path
import html,json,re,uuid

MAX_BYTES=2*1024**2
MAX_TEXT=500_000
_TIME=r'(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})'
_CUE=re.compile(r'^[ \t]*(?:\d+[ \t]*\n[ \t]*)?'+_TIME+r'[ \t]*-->[ \t]*'+_TIME+r'[^\n]*\n',re.M)


def _seconds(parts):
    h,m,s,ms=parts
    if int(m)>=60 or int(s)>=60:raise ValueError('Invalid SRT timestamp.')
    return int(h)*3600+int(m)*60+int(s)+int(ms.ljust(3,'0'))/1000


def parse_srt(raw,name):
    if Path(name).suffix.lower()!='.srt':raise ValueError('Select an SRT subtitle file.')
    if len(raw)>MAX_BYTES:raise ValueError('SRT files must be 2 MB or smaller.')
    try:
        if raw.startswith((b'\xff\xfe',b'\xfe\xff')):text=raw.decode('utf-16')
        else:text=raw.decode('utf-8-sig')
    except UnicodeError:
        from charset_normalizer import from_bytes
        match=from_bytes(raw).best()
        if match is None:raise ValueError('Cannot decode the SRT file. Save it as UTF-8 and retry.') from None
        text=str(match)
    text=text.replace('\r\n','\n').replace('\r','\n')
    markers=list(_CUE.finditer(text));cues=[];total=0
    for i,match in enumerate(markers):
        start=_seconds(match.groups()[:4]);end=_seconds(match.groups()[4:])
        if end<=start:raise ValueError('SRT subtitle end must be after its start.')
        body=text[match.end():markers[i+1].start() if i+1<len(markers) else len(text)].strip()
        body=html.unescape(re.sub(r'</?(?:i|b|u|font|span|ruby|rt)(?:\s[^>]*)?>','',body,flags=re.I)).strip()
        if not body:continue
        total+=len(body)
        if total>MAX_TEXT or len(cues)>=20_000:raise ValueError('SRT reference text is too large.')
        cues.append({'start':start,'end':end,'text':body})
    if not cues:raise ValueError('The SRT file contains no readable subtitle cues.')
    filename=name.replace('\\','/').split('/')[-1][:240]
    return {'version':1,'name':filename,'cues':sorted(cues,key=lambda c:(c['start'],c['end']))}


def save_for_parts(reference,parts):
    from core import atomic_json
    for part in parts:
        folder=Path(part['folder']);origin=folder/'split-origin.json'
        bounds=json.loads(origin.read_text('utf8')) if origin.exists() else {'start':0.,'end':float(part['info']['format']['duration'])}
        lo,hi=bounds['start'],bounds['end']
        cues=[{**cue,'start':max(0.,cue['start']-lo),'end':min(hi-lo,cue['end']-lo)}
              for cue in reference['cues'] if cue['start']<hi and cue['end']>lo]
        atomic_json(folder/'translation-reference.json',{'version':1,'name':reference['name'],'cues':cues})


def load_for_video(video_id,root):
    if not isinstance(video_id,str):return None
    try:
        if str(uuid.UUID(video_id))!=video_id:return None
    except ValueError:return None
    path=Path(root)/'work/videos'/video_id/'translation-reference.json'
    if not path.exists():return None
    reference=json.loads(path.read_text('utf8'))
    if reference.get('version')!=1 or not isinstance(reference.get('cues'),list):
        raise RuntimeError('Invalid saved SRT translation reference.')
    return reference if reference['cues'] else None
