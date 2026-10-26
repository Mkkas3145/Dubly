"""One native-audio Qwen request: transcription, voice IDs and translation."""
from pathlib import Path
import base64,hashlib,json,math,re,time,urllib.request,urllib.error
import soundfile as sf
from core import SR,LANGUAGES,atomic_json,command

MODEL='qwen/qwen3.8-omni-flash'
AUTHORITY='qwen_omni_audio'
POLICY={'version':11,'model':MODEL,'reasoning':{'enabled':False},
    'max_tokens':131072,'input':'one_full_dialogue_audio_no_silence_removal',
    'audio_sample_rate':'preserve_source_when_supported','audio_channels':'preserve_source_up_to_stereo','base64_limit':9_800_000,
    'bitrates_kbps':[320,256,192,160,128,96,64,48,40,32,24,16],'speaker_ids':'from_qwen_audio_only',
    'boundaries':'qwen_times_then_neighbor_bounded_waveform_edges','extra_asr_requests':0}
PROMPT='''Listen to the entire audio through the end. Transcribe the original dialogue. Translate faithfully to the original into natural spoken {language}.
Identify speakers primarily by their audible voices, using stable integer speaker IDs across the recording. Speaker identity evidence priority: first compare stable audible voice traits across clear examples (timbre, resonance, articulation, vocal texture and habitual rhythm). Use conversational context ONLY AS A LAST RESORT when acoustic comparison remains inconclusive, and only as a cautious tie-breaker among acoustically plausible identities. Context must NEVER override clear voice evidence, merge audibly distinct people or split one clearly matching voice. Do not force an assignment from story roles, character names, assumed turn-taking or external plot knowledge. If uncertainty remains, return speaker:null rather than guessing. If voice identity is ambiguous, return speaker:null rather than guessing or assigning a shared unknown-speaker ID. Changes in acting, pitch or emotion alone do not change speaker identity; require audible voice evidence to link turns. Split when the speaker changes or a natural phrase ends; keep a hesitation with its same-speaker continuation.
Cues MUST appear in original_text itself: use concise English square-bracketed [] cues at audible changes in emotion or vocal delivery, including meaningful breaths or laughter within dialogue. The contents of every [] cue MUST be English in BOTH original_text and translated_text. Translate spoken dialogue only; never translate, localize or paraphrase the English text inside brackets. Then copy every cue unchanged to the corresponding position in translated_text. Neither field may have a cue missing from the other. Keep cues grounded in the audio; do not invent or exaggerate acting or label neutral speech unnecessarily.
Omit standalone nonverbal sounds, meaningless fragments, and sound-equivalent reactions that need no dubbing; do not turn these into translated interjections or cue-only entries. Preserve meaningful short replies, unfinished dialogue and sentence-internal hesitations. For any retained turn needing no dubbing, set translated_text to null.
Return only {{"segments":[{{"start":0.0,"end":1.0,"speaker":0,"original_text":"original dialogue","translated_text":"translation"}}]}} in audio order. Times are seconds from this audio's beginning and cover the complete utterance. Check speaker continuity, paired cues and coverage through the end. No additional fields or explanations.'''


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def encode_audio(clean,folder,source_hash,progress,*,base64_limit=None):
    native=sf.info(clean);duration=native.duration
    limit=POLICY['base64_limit'] if base64_limit is None else base64_limit
    # Keep the original clock and the entire file in one audio part. Select
    # compression by measured payload bytes, never by detected speech duration.
    for bitrate in POLICY['bitrates_kbps']:
        if duration*bitrate*1000/8*4/3+8192>limit:continue
        # MPEG-1 at 44.1 kHz silently clamps 16/24 kbps to 32 kbps.
        # MPEG-2 at 24 kHz actually supports the lower rates for long clips.
        sample_rate=native.samplerate if native.samplerate in (8000,11025,12000,16000,22050,24000,32000,44100,48000) else 48000
        if bitrate<32 and sample_rate>=32000:sample_rate=24000
        if sample_rate<32000 and bitrate>160:continue
        channels=min(native.channels,2)
        path=folder/f'full-{source_hash[:16]}-{sample_rate}-{channels}ch-{bitrate}k.mp3'
        if not path.exists():
            progress(2,'Qwen 전체 대사 오디오 압축 · 무음 유지')
            temporary=path.with_suffix('.partial.mp3')
            command(['ffmpeg','-nostdin','-v','error','-y','-i',str(clean),'-map','0:a:0',
                '-ar',str(sample_rate),'-ac',str(channels),'-c:a','libmp3lame','-b:a',f'{bitrate}k',str(temporary)])
            info=sf.info(temporary)
            if info.samplerate!=sample_rate or info.channels!=channels or abs(info.duration-duration)>.15:
                raise RuntimeError('Compressed Qwen audio duration or format differs from the source.')
            temporary.replace(path)
        if 4*((path.stat().st_size+2)//3)<=limit:
            return path,{'format':'mp3','sample_rate':sample_rate,'channels':channels,'bitrate_kbps':bitrate,
                'bytes':path.stat().st_size,'duration_seconds':duration,'sha256':digest(path),
                'transport':'raw_base64_without_data_uri','silence_removed':False,'audio_parts':1,
                'base64_bytes':4*((path.stat().st_size+2)//3),'base64_budget_bytes':limit}
    raise RuntimeError('Compressed Qwen audio still exceeds the provider input size limit. Audio was not automatically split.')


def response_format():
    return {'type':'json_schema','json_schema':{'name':'qwen_dubbing','strict':True,'schema':{
        'type':'object','properties':{'segments':{'type':'array','items':{'type':'object','properties':{
            'start':{'type':'number'},'end':{'type':'number'},'speaker':{'type':['integer','null']},
            'original_text':{'type':'string'},'translated_text':{'type':['string','null']}},
            'required':['start','end','speaker','original_text','translated_text'],'additionalProperties':False}}},
        'required':['segments'],'additionalProperties':False}}}


def parse(content,duration):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError(f'Duplicate JSON key: {k}')
            result[k]=v
        return result
    raw=json.loads(content,object_pairs_hook=unique)
    entries=raw if isinstance(raw,list) else raw.get('segments') if isinstance(raw,dict) else None
    if not isinstance(entries,list):raise ValueError('Expected segments array')
    rows=[];previous=-1.
    for i,item in enumerate(entries):
        if not isinstance(item,dict) or set(item)!={'start','end','speaker','original_text','translated_text'}:
            raise ValueError(f'Segment {i+1}: unexpected or missing fields')
        a,b=item['start'],item['end'];speaker=item['speaker'];text=item['original_text'];translated=item['translated_text']
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (a,b)) or not 0<=a<b<=duration+.15 or a<previous:
            raise ValueError(f'Segment {i+1}: invalid time/order ({a}, {b}) for {duration:.3f}s')
        if speaker is not None and (type(speaker) is not int or speaker<0):raise ValueError(f'Segment {i+1}: invalid speaker')
        if not isinstance(text,str) or not text.strip() or len(text)>4000:raise ValueError(f'Segment {i+1}: invalid original text')
        if translated is not None and (not isinstance(translated,str) or not translated.strip() or len(translated)>4000):
            raise ValueError(f'Segment {i+1}: invalid translation')
        previous=float(a)
        rows.append({'id':i,'start':float(a),'end':min(duration,float(b)),
            'speaker':'unknown' if speaker is None else f'qwen_{speaker}',
            'text':text.strip(),'translation':translated.strip() if translated is not None else None})
    return rows


def request(payload,key,folder,duration,progress,*,parser=None,label='Qwen 전사·화자·번역',repair_invalid=False):
    from speech import read_stream
    folder.mkdir(exist_ok=True,parents=True)
    encoded=json.dumps(payload).encode()
    for attempt in range(1,4):
        started=time.monotonic();stem=f'{time.time_ns()}-{attempt}'
        def activity(event):
            count=event.get('received_segments',0);elapsed=round(time.monotonic()-started)
            stage='첫 결과 대기' if not count else f'{count}개 대사 수신'
            progress(min(85,10+count*.1),f'{label} · {stage} · {elapsed}초')
        req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=encoded,
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Title':'Fast Dubbing native Qwen dubbing'})
        try:
            with urllib.request.urlopen(req,timeout=900) as stream,(folder/f'{stem}.sse').open('wb') as record:
                data=read_stream(stream,lambda n:None,record=record,on_activity=activity)
            atomic_json(folder/f'{stem}.json',data)
            choice=data.get('choices',[{}])[0]
            if data.get('error') or choice.get('error'):
                error=data.get('error') or choice['error'];code=error.get('code') if isinstance(error,dict) else None
                if str(code) in ('408','429','500','502','503','504') and attempt<3:
                    time.sleep(2**attempt);continue
                raise RuntimeError(f'Qwen provider error {code}. The raw response was preserved.')
            if choice.get('finish_reason')!='stop':raise RuntimeError(f'Qwen response is incomplete ({choice.get("finish_reason")}). The raw response and usage were preserved.')
            try:rows=(parser or parse)(choice['message']['content'],duration)
            except (ValueError,KeyError,TypeError) as error:
                atomic_json(folder/f'{stem}.validation.json',{'error':str(error),'finish_reason':choice.get('finish_reason'),'usage':data.get('usage',{})})
                if repair_invalid and attempt<3:
                    progress(10,f'{label} · 응답 형식 재확인 {attempt}/2')
                    feedback=('Your previous response failed validation: '+str(error)+'. Re-listen to the complete supplied audio and return the COMPLETE JSON again, covering every utterance through the end. '
                        'EVERY segment MUST contain all four fields: start, end, speaker, original_text. Both start AND end must be explicit numeric seconds. Never omit end. '
                        'ALL square-bracketed delivery cues MUST be ENGLISH, even when nicknames are in another language. Do not alter spoken words when changing cue language. No translation of spoken dialogue. No explanations.')
                    retry_payload={**payload,'messages':[*payload['messages'],{'role':'assistant','content':choice['message'].get('content','')},{'role':'user','content':feedback}]}
                    encoded=json.dumps(retry_payload).encode()
                    continue
                raise RuntimeError(f'Qwen dialogue JSON validation failed: {error}. The raw response was preserved.') from None
            return rows,{'usage':data.get('usage',{}),'elapsed_seconds':time.monotonic()-started,
                'response_id':data.get('id'),'provider':data.get('provider'),'successful_attempt':attempt}
        except urllib.error.HTTPError as error:
            body=error.read().decode('utf8','replace').replace(key,'[REDACTED]')
            atomic_json(folder/f'{stem}.http-error.json',{'status':error.code,'response':body})
            if (error.code==429 or error.code>=500) and attempt<3:
                time.sleep(2**attempt);continue
            raise RuntimeError(f'Qwen HTTP {error.code}. The raw provider response was preserved.') from None


def nonoverlapping(rows):
    """One replacement interval for genuinely overlapping estimated turns."""
    result=[]
    for row in rows:
        row=dict(row)
        if result and row['start']<result[-1]['end']:
            previous=result[-1];previous['end']=max(previous['end'],row['end'])
            previous['text']+=' '+row['text']
            a,b=previous['translation'],row['translation']
            previous['translation']=' '.join(x for x in (a,b) if x is not None) if a is not None or b is not None else None
            if previous['speaker']!=row['speaker']:previous['speaker']='unknown'
        else:result.append(row)
    return result


def transcribe(clean,work,output,cfg,progress):
    clean,work,output=Path(clean),Path(work),Path(output);work.mkdir(exist_ok=True,parents=True)
    duration=sf.info(clean).duration;source_hash=digest(clean)
    audio,manifest=encode_audio(clean,work,source_hash,progress)
    prompt=PROMPT.format(language=LANGUAGES[cfg['language']])
    signature={'source_sha256':source_hash,'language':cfg['language'],'prompt':prompt,'policy':POLICY,'audio':manifest}
    cache=work/'qwen-completed.json';saved=json.loads(cache.read_text('utf8')) if cache.exists() else {}
    if saved.get('signature')==signature:
        rows=parse(json.dumps(saved['segments'],ensure_ascii=False),duration);receipt=saved['receipt']
        progress(85,'Qwen 전사·화자·번역 캐시 재사용')
    else:
        atomic_json(work/'qwen-input-manifest.json',signature)
        payload={'model':MODEL,'max_tokens':POLICY['max_tokens'],'reasoning':POLICY['reasoning'],
            'provider':{'require_parameters':True},'stream':True,'stream_options':{'include_usage':True},
            'response_format':response_format(),'messages':[{'role':'user','content':[
                {'type':'text','text':prompt},{'type':'input_audio','input_audio':{'format':manifest['format'],
                    'data':base64.b64encode(audio.read_bytes()).decode()}}]}]}
        rows,receipt=request(payload,cfg['key'],work/'qwen-responses',duration,progress)
        encoded=[{'start':r['start'],'end':r['end'],'speaker':None if r['speaker']=='unknown' else int(r['speaker'].split('_')[-1]),
            'original_text':r['text'],'translated_text':r['translation']} for r in rows]
        atomic_json(cache,{'signature':signature,'segments':encoded,'receipt':receipt})
    result=assemble(clean,rows,signature,receipt,cfg,progress)
    atomic_json(output,result);progress(100,f'Qwen 전사·화자·번역 완료 · {len(result["segments"])}개 대사')
    return result


def assemble(clean,rows,signature,receipt,cfg,progress=lambda *a:None):
    from qwen_boundaries import correct_edges,POLICY as boundary_policy
    duration=sf.info(clean).duration
    raw_turns=[{'start':r['start'],'end':r['end'],'speaker':r['speaker']} for r in rows]
    bounded,audit=correct_edges(clean,nonoverlapping(rows),lambda p,d:progress(86+(p-30)*.6,d.replace('MAI','Qwen')))
    segments=[];excluded=[];windows=[]
    for row in bounded:
        translated=row['translation']
        if translated is None or not any(c.isalnum() for c in re.sub(r'\[[^\]]*\]','',translated)):
            excluded.append({'start':row['start'],'end':row['end'],'reason':'not_needing_dubbing'});continue
        segments.append({'start':row['start'],'end':row['end'],'speaker_id':row['speaker'],
            'original_text':row['text'],'translated_text':translated,'source_region_id':row['id']})
        windows.append({'start':row['start'],'end':row['end'],'start_frame':round(row['start']*SR),
            'end_frame':round(row['end']*SR),'fade_in_frames':0,'fade_out_frames':0,'boundary_policy':AUTHORITY})
    # Reference grouping sees the same corrected bounds as actual reference cuts.
    timeline={'identity':signature,'duration':duration,'boundary_authority':AUTHORITY,
        'turns':[{'start':r['start'],'end':r['end'],'speaker':r['speaker']} for r in bounded],
        'raw_turns':raw_turns,'external_embedding_model':None}
    result={'model':MODEL,'segments':segments,'excluded_intervals':excluded,'source_regions':bounded,
        'speaker_timeline':timeline,'reference_windows':windows,'boundary_measurements':audit,
        'boundary_authority':AUTHORITY,'speaker_authority':MODEL,'original_text_source':MODEL,
        'reference_speaker_boundaries':AUTHORITY,'boundary_policy':boundary_policy,'target_language':cfg['language'],'source_languages':['unknown'],
        'duration_seconds':duration,'time_unit':'seconds','llm_timestamps':True,'asr_sent_to_llm':False,
        'source_asr_model':MODEL,'source_asr_input_policy':POLICY['input'],'asr_used':True,
        'transcription_mode':'one_native_audio_request_transcription_diarization_translation',
        'request_count':1,'additional_asr_requests':0,'translation_output':'source_translation_times_voice_per_turn',
        'reasoning':POLICY['reasoning'],'usage':receipt['usage'],'request_receipt':receipt,
        'transition_policy':'standard_external_fade','reference_tail_policy':{'enabled':False,'method':'qwen_waveform_edges'},
        'unmatched_segments':[]}
    return result
