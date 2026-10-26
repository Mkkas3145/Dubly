"""Cached OpenRouter recovery transcription (legacy cache filenames retained)."""
from pathlib import Path
import base64,hashlib,json,time,threading,urllib.request,urllib.error
import soundfile as sf
from core import atomic_json,command

MODEL='microsoft/mai-transcribe-2'
VERSION=1
URL='https://openrouter.ai/api/v1/audio/transcriptions'
_CACHE_LOCK=threading.RLock()


def read_cache(path):
    # Windows does not allow replacing a file while another thread opens it
    # without delete sharing. Serialize only the short local cache operation;
    # network requests remain parallel.
    with _CACHE_LOCK:
        try:return json.loads(path.read_text(encoding='utf8'))
        except (OSError,ValueError):return {}


def write_cache(path,data):
    with _CACHE_LOCK:
        for attempt in range(5):
            try:atomic_json(path,data);return
            except PermissionError:
                if attempt==4:raise
                time.sleep(.025*2**attempt)


def transcribe(source,folder,key,language=None,model=MODEL):
    source=Path(source);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if not key:raise RuntimeError('OpenRouter API key is missing.')
    options={'azure':{'diarization':{'enabled':True},'enhancedMode':{'modelOptions':{'transcribeStyle':'verbatim'}}}} if model=='microsoft/mai-transcribe-2' else {}
    identity={'model':model,'version':VERSION,'audio_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'language':language,'response_format':'verbose_json','timestamp_granularities':['word']}
    if options:identity['provider_options']=options
    cache=folder/'qwen.json'
    if cache.exists():
        old=read_cache(cache)
        if old.get('identity')==identity:return old
    # Job numbering can change after a neighboring line is repaired. Reuse
    # identical paid recognition by audio content rather than that number.
    search=folder.parent
    for other in search.glob('*/qwen.json'):
        if other==cache:continue
        # Another parallel recovery job may be atomically replacing this
        # optional cache entry on Windows. A transient sharing violation must
        # not abort the whole video's recognition batch.
        old=read_cache(other)
        if old.get('identity')==identity:
            write_cache(cache,old);return old
    converted=folder/'input.flac'
    command(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-ar','16000','-ac','1',
        '-sample_fmt','s16','-c:a','flac',str(converted)])
    payload={'model':model,'input_audio':{'data':base64.b64encode(converted.read_bytes()).decode(),'format':'flac'},
        'response_format':'verbose_json','timestamp_granularities':['word']}
    if options:payload['provider']={'options':options}
    if language:payload['language']=language
    request=urllib.request.Request(URL,data=json.dumps(payload).encode(),method='POST',
        headers={'Authorization':f'Bearer {key}','Content-Type':'application/json','X-Title':'Fast Dubbing source recovery'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=180) as response:data=json.load(response)
            atomic_json(folder/'raw.json',data)
            if not isinstance(data.get('text'),str):raise RuntimeError('Recovery ASR response contains no text.')
            result={'identity':identity,'duration':sf.info(converted).duration,'text':data['text'],
                'words':data.get('words',[]),'segments':data.get('segments',[]),'usage':data.get('usage',{})}
            write_cache(cache,result);return result
        except urllib.error.HTTPError as error:
            code=error.code;body=error.read().decode('utf8',errors='replace');error.close()
            (folder/f'http-{code}.json').write_text(body,encoding='utf8')
            if (code==429 or code>=500) and attempt<2:time.sleep(2**attempt);continue
            raise RuntimeError(f'Recovery ASR {model} HTTP {code}. The raw response was preserved.') from None
        except (urllib.error.URLError,TimeoutError):
            if attempt<2:time.sleep(2**attempt);continue
            raise RuntimeError(f'Recovery ASR {model} request timed out.') from None
