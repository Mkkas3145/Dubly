"""OpenRouter Scribe v2 word timestamps, cached by actual audio content."""
from __future__ import annotations
import base64,hashlib,json,math,time,urllib.error,urllib.request
from pathlib import Path
import soundfile as sf
from core import atomic_json,command

MODEL='elevenlabs/scribe-v2'
VERSION=1
# Official range is 0.1–0.4: its lowest threshold maximizes sensitivity.
# Keep num_speakers unset so the threshold is honored.
DIARIZATION_THRESHOLD=0.10
URL='https://openrouter.ai/api/v1/audio/transcriptions'
# Leave room for base64 expansion and the JSON envelope as well as the file cap.
INLINE_REQUEST_LIMIT=25*1024**2
AUDIO_BYTE_BUDGET=(INLINE_REQUEST_LIMIT-8192)//4*3
MP3_BITRATES=(320,256,192,160,128,96,64,48,32,24,16)


def prepare_input(source,folder,digest,*,audio_byte_budget=None):
    budget=AUDIO_BYTE_BUDGET if audio_byte_budget is None else audio_byte_budget
    converted=folder/'scribe-input-native.flac';encoded_identity=folder/'input-native.json'
    old=json.loads(encoded_identity.read_text(encoding='utf-8')) if encoded_identity.exists() else {}
    if not converted.exists() or old.get('source_sha256')!=digest:
        temp=folder/'scribe-input-native.partial.flac'
        command(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-c:a','flac','-compression_level','12',str(temp)])
        temp.replace(converted);atomic_json(encoded_identity,{'source_sha256':digest})
    info=sf.info(converted);duration=info.frames/info.samplerate
    if converted.stat().st_size<=budget:
        return converted,'flac',duration
    native=sf.info(source)
    for bitrate in MP3_BITRATES:
        rate=native.samplerate if native.samplerate in (8000,11025,12000,16000,22050,24000,32000,44100,48000) else 48000
        if bitrate<32 and rate>=32000:rate=24000
        if rate<32000 and bitrate>160:continue
        channels=min(native.channels,2)
        if duration*bitrate*1000/8+8192>budget:continue
        compressed=folder/f'scribe-input-{digest[:16]}-{rate//1000}k-{channels}ch-{bitrate}k.mp3'
        if not compressed.exists():
            temp=compressed.with_suffix('.partial.mp3')
            command(['ffmpeg','-nostdin','-v','error','-y','-i',str(converted),'-map','0:a:0',
                '-ar',str(rate),'-ac',str(channels),'-c:a','libmp3lame','-b:a',f'{bitrate}k',str(temp)])
            audio=sf.info(temp)
            if audio.channels!=channels or audio.samplerate!=rate or abs(audio.duration-duration)>.15:
                temp.unlink(missing_ok=True)
                raise RuntimeError('Compressed ASR audio duration or format differs from the source.')
            temp.replace(compressed)
        if compressed.stat().st_size<=budget:
            atomic_json(folder/'scribe-input-compression.json',{'source_sha256':digest,'format':'mp3',
                'sample_rate':rate,'channels':channels,'bitrate_kbps':bitrate,'duration_seconds':duration,
                'bytes':compressed.stat().st_size,'flac_bytes':converted.stat().st_size,
                'split':False,'silence_removed':False,'compression_order':'native_flac_then_highest_fitting_mp3_then_required_rate_reduction'})
            return compressed,'mp3',duration
    raise RuntimeError('ASR audio still exceeds the input size limit after compression to 16 kbps MP3. Audio was not trimmed or automatically split.')

def parse_words(data,duration):
    words=[];events=[]
    for item in data.get('words',[]):
        text=item.get('word',item.get('text',''))
        start,end=item.get('start'),item.get('end')
        if not isinstance(text,str) or not isinstance(start,(int,float)) or not isinstance(end,(int,float)):continue
        if not math.isfinite(start) or not math.isfinite(end) or end<start or start<0 or start>=duration or end>duration+.2:continue
        word={'word':text,'start':float(start),'end':min(float(end),duration),
            'probability':float(item['confidence']) if item.get('confidence') is not None else 1.}
        for name in ['speaker','speaker_id','speaker_label','type']:
            if name in item:word[name]=item[name]
        if item.get('type') in ['audio_event','audio_event_tag'] or (text.strip().startswith('(') and text.strip().endswith(')')):
            events.append(word)
        elif text.strip() and item.get('type')!='spacing':words.append(word)
    if not any(w['end']>w['start'] for w in words):raise RuntimeError('ASR response contains no valid word timestamps. Previous timestamps or another model were not substituted.')
    return sorted(words,key=lambda w:(w['start'],w['end'])),events

def transcribe_words(source,folder,key,language=None,progress=lambda *args:None,attempts=1,model=MODEL,*,provider_options=None):
    source=Path(source);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if not key:raise RuntimeError('OPENROUTER_API_KEY is not configured.')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    signature={'model':model,'version':VERSION,'source_sha256':digest,'language':language,
        'response_format':'verbose_json','timestamp_granularities':['word']}
    if provider_options:signature['provider_options']=provider_options
    cache=folder/'scribe.json'
    old=json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else {}
    if old.get('signature')==signature:
        raw_cache=folder/'scribe-raw.json'
        if raw_cache.exists():
            raw=json.loads(raw_cache.read_text(encoding='utf-8'))
            words,events=parse_words(raw,duration=float(old['duration']))
            old.update(words=words,audio_events=events)
            old['language']=raw.get('language') or raw.get('language_code') or old.get('language')
            if isinstance(raw.get('segments'),list):old['segments']=raw['segments']
            atomic_json(cache,old)
        progress(100,'ASR 단어 시간 캐시 재사용');return old
    progress(5,f'{model} 입력 음성 준비')
    converted,audio_format,duration=prepare_input(source,folder,digest)
    payload={'model':model,'input_audio':{'data':base64.b64encode(converted.read_bytes()).decode(),'format':audio_format},
        'response_format':'verbose_json','timestamp_granularities':['word']}
    if provider_options:payload['provider']={'options':provider_options}
    if language:payload['language']=language
    request=urllib.request.Request(URL,data=json.dumps(payload).encode(),method='POST',
        headers={'Authorization':f'Bearer {key}','Content-Type':'application/json','X-Title':'Fast Dubbing'})
    progress(15,f'{model} 전체 음성 전사 요청')
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request,timeout=240) as response:data=json.load(response)
            atomic_json(folder/'scribe-raw.json',data)
            if data.get('error'):raise RuntimeError('ASR provider returned a transcription error.')
            words,events=parse_words(data,duration)
            result={'signature':signature,'duration':duration,'words':words,'audio_events':events,
                'text':data.get('text',''),'language':data.get('language') or data.get('language_code') or language,'usage':data.get('usage',{})}
            if isinstance(data.get('segments'),list):result['segments']=data['segments']
            atomic_json(folder/'scribe-raw.json',data);atomic_json(cache,result)
            progress(100,f'{model} 전사 수신 완료');return result
        except urllib.error.HTTPError as error:
            code=error.code;error.close()
            if (code==429 or code>=500) and attempt<attempts-1:time.sleep(2**(attempt+1));continue
            raise RuntimeError(f'ASR HTTP {code}. Check OpenRouter model access, credits, and input format.') from None
        except (urllib.error.URLError,TimeoutError):
            raise RuntimeError('ASR request timed out. Successful transcription caches were preserved; no other model was substituted.') from None
