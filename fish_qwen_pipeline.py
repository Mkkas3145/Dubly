"""Fish owns transcript/time, Qwen owns voice IDs, Gemini owns translation."""
from pathlib import Path
import base64,json,math,re,time,urllib.request,urllib.error
import soundfile as sf
from core import SR,QWEN,GEMINI,LANGUAGES,atomic_json
from source_asr import transcribe_words,MODEL as ASR,CACHE_DIRECTORY,INPUT_POLICY
from rich_transcript import enrich,join_words,spoken
from qwen_audio_pipeline import encode_audio,digest
from qwen_boundaries import correct_edges,POLICY as BOUNDARY_POLICY

AUTHORITY='fish_qwen_gemini'
POLICY={'version':1,'source_asr':ASR,'speaker_model':QWEN,'translation_model':GEMINI,
    'asr_input':'whole_separated_dialogue_original_clock','same_voice_gap_seconds':.8,
    'qwen_output':'region_id_to_speaker_integer_or_null_only','qwen_reasoning':{'enabled':False},
    'gemini_output':'region_id_to_translation_or_null_only','gemini_audio':False,
    'omitted_translation':'keep_original','separate_stage_caches':True,'qwen_audio_base64_budget_bytes':8_000_000,
    'post_qwen_merge':{'version':1,'same_voice_max_gap_seconds':.8,'punctuation_does_not_block':True,'unknown_voices':'never_merge'}}
SPEAKER_PROMPT='''Identify speakers primarily by audible voice characteristics. Speaker identity evidence priority: first compare stable audible voice traits across clear examples (timbre, resonance, articulation, vocal texture and habitual rhythm). Use conversational context ONLY AS A LAST RESORT when acoustic comparison remains inconclusive, and only as a cautious tie-breaker among acoustically plausible identities. Context must NEVER override clear voice evidence, merge audibly distinct people or split one clearly matching voice. Do not force an assignment from story roles, character names, assumed turn-taking or external plot knowledge. If uncertainty remains, return speaker:null rather than guessing. Text and times locate the corresponding audio regions. There is NO limit on the number of speakers. Create as many distinct speaker IDs as the audible voices require; never merge different voices or return null merely to fit a presumed speaker-count limit. Keep one stable integer ID for the same audible voice despite acting changes. Return null if uncertain or mixed.
OUTPUT ONLY one JSON object mapping each supplied region ID to an integer speaker number or null. This is mandatory. NEVER echo input region objects, start/end times, transcript text, descriptions or explanations. No arrays, nested objects or additional fields.'''
TRANSLATION_PROMPT='''Translate faithfully to the original into natural spoken {language}. Keep each supplied region separate. Preserve every supplied [] cue verbatim in its original language and at its corresponding position. Do not translate, paraphrase, replace, or invent text inside brackets. Translate only the spoken dialogue outside brackets. Speaker IDs are fixed: do not change them. Omit unnecessary nonverbal reactions and sounds needing no dubbing by returning null or leaving out that region ID. Preserve meaningful short dialogue.
Return only a JSON object mapping region IDs to translated strings or null. No source text, timestamps, speaker IDs or explanations.'''


def regions(raw,duration):
    """Sentence/word boundaries avoid letting one mistaken Fish ID join a scene."""
    groups=[]
    words=sorted(enrich(raw).get('words',[]),key=lambda w:(w['start'],w['end']))
    for word in words:
        a,b=word['start'],word['end']
        if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (a,b)) or not 0<=a<=b<=duration+.2:continue
        word={**word,'end':min(b,duration)}
        hint=word.get('fish_speaker') or word.get('speaker',word.get('speaker_id'))
        previous=groups[-1] if groups else None
        surface=spoken(join_words(previous['words'])) if previous else ''
        completed=bool(re.search(r'[。！？!?][」』\"”’]*$',surface) or re.search(r'(?<!\.)\.[\"”’]*$',surface))
        changed=previous is not None and hint is not None and previous['hint'] is not None and str(hint)!=str(previous['hint'])
        if previous and not completed and not changed and a-previous['end']<=.8+1e-7:
            previous['words'].append(word);previous['end']=max(previous['end'],word['end'])
        else:groups.append({'start':a,'end':word['end'],'hint':hint,'words':[word]})
    rows=[]
    for group in groups:
        text=join_words(group['words'])
        if not any(c.isalnum() for c in spoken(text)):continue
        row={'start':group['start'],'end':group['end'],'text':text,'speaker':'unknown','words':group['words']}
        if rows and row['start']<rows[-1]['end']:
            rows[-1]['end']=max(rows[-1]['end'],row['end']);rows[-1]['words']+=row['words']
            rows[-1]['text']=join_words(rows[-1]['words'])
        else:rows.append(row)
    for i,row in enumerate(rows):row['id']=i
    if not rows and spoken(raw.get('text','')).strip():raise RuntimeError('Fish transcription contains no word timestamps usable for segmentation. The raw response was preserved.')
    return rows


def schema(kind):
    value={'type':['integer','null']} if kind=='speakers' else {'type':['string','null']}
    return {'type':'json_schema','json_schema':{'name':kind,'strict':True,
        'schema':{'type':'object','additionalProperties':value}}}


SPEAKER_MESSAGE_LAYOUT='system_task_data_audio_output_reminder_v1'


def speaker_payload(inputs,audio,manifest):
    return {'model':QWEN,'max_tokens':131072,'reasoning':{'enabled':False},'provider':{'require_parameters':True},
        'stream':True,'stream_options':{'include_usage':True},'response_format':schema('speakers'),
        'messages':[{'role':'system','content':SPEAKER_PROMPT},
            {'role':'user','content':[
                {'type':'text','text':'REGIONS: input data only, not the requested output.\n'+json.dumps(inputs,ensure_ascii=False,separators=(',',':'))},
                {'type':'input_audio','input_audio':{'format':manifest['format'],'data':base64.b64encode(audio.read_bytes()).decode()}},
                {'type':'text','text':'Assign speaker IDs from the audio now. Output ONLY the region-ID-to-speaker-integer-or-null JSON object. Do NOT repeat the supplied region data.'}]}]}


def merge_voice_turns(rows,measurements):
    """Qwen identity is authoritative; Fish punctuation only locates drafts."""
    merged=[];audits=[];changes=[]
    for row,measurement in zip(rows,measurements):
        row={**row,'source_region_ids':list(row.get('source_region_ids',[row['id']]))}
        previous=merged[-1] if merged else None
        eligible=(previous is not None and row['speaker']==previous['speaker'] and row['speaker']!='unknown'
            and not row.get('keep_original') and not previous.get('keep_original')
            and 0<=row['start']-previous['end']<=POLICY['post_qwen_merge']['same_voice_max_gap_seconds']+1e-7)
        if eligible:
            changes.append({'left_ids':list(previous['source_region_ids']),'right_ids':list(row['source_region_ids']),
                'gap_seconds':row['start']-previous['end'],'speaker':row['speaker']})
            previous['end']=row['end'];previous['text']+=' '+row['text']
            previous['words']=[*previous.get('words',[]),*row.get('words',[])]
            previous['source_region_ids']+=row['source_region_ids']
            audits[-1]['end']=row['end'];audits[-1]['members'].append(measurement)
            audits[-1]['source_region_ids']=list(previous['source_region_ids'])
        else:
            merged.append(row)
            audits.append({'method':'qwen_same_voice_gap_merge','start':row['start'],'end':row['end'],
                'source_region_ids':list(row['source_region_ids']),'members':[measurement]})
    for i,(row,audit) in enumerate(zip(merged,audits)):row['id']=i;audit['source_index']=i
    return merged,audits,changes


def parse_map(content,ids,kind):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError(f'Duplicate region ID {key}')
            result[key]=value
        return result
    result=json.loads(content,object_pairs_hook=unique);keys=[str(i) for i in ids]
    # Some Qwen endpoints ignore the object wrapper but preserve fixed order.
    if kind=='speakers' and isinstance(result,list):
        from speech import speaker_entry,speaker_pair
        if all(speaker_entry(item) for item in result):
            pairs=[]
            for item in result:
                # Any echoed start/end values are discarded here, regardless
                # of their content. Source/locally corrected times stay fixed.
                pairs.append(speaker_pair(item))
            result=unique(pairs)
        elif len(result)==len(keys):result=dict(zip(keys,result))
    if not isinstance(result,dict) or not set(result)<=set(keys):raise ValueError('Expected known region IDs only')
    for key,value in result.items():
        if kind=='speakers':
            if value is not None and (type(value) is not int or value<0):raise ValueError(f'Invalid speaker for region {key}')
        elif value is not None and (not isinstance(value,str) or not value.strip() or len(value)>4000):
            raise ValueError(f'Invalid translation for region {key}')
    # A missing voice remains unresolved; missing translations retain original.
    return {key:result.get(key) for key in keys}


def same_signature(left,right):
    def request_identity(value):
        value=dict(value or {})
        if isinstance(value.get('audio'),dict):
            value['audio']={k:v for k,v in value['audio'].items() if k not in ('base64_bytes','base64_budget_bytes')}
        return value
    return request_identity(left)==request_identity(right)


def stage_chat(payload,key,work,signature,ids,kind,progress):
    """Reuse a complete saved response after a parser compatibility repair."""
    label='qwen-speaker' if kind=='speakers' else 'gemini-translation'
    manifest=work/f'{label}-input.json';folder=work/f'{label}-responses'
    old=json.loads(manifest.read_text('utf8')) if manifest.exists() else {}
    if same_signature(old.get('signature'),signature) and folder.exists():
        for response in sorted(folder.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            if response.name.endswith(('.validation.json','.http-error.json')):continue
            if response.stat().st_mtime<manifest.stat().st_mtime:continue
            try:
                data=json.loads(response.read_text('utf8'));choice=data['choices'][0]
                if choice.get('finish_reason')!='stop' or data.get('error') or choice.get('error'):continue
                if data.get('model') not in (None,payload['model']):continue
                values=parse_map(choice['message']['content'],ids,kind)
            except (ValueError,KeyError,TypeError):continue
            progress(100,'보관된 원본 응답 재사용 · 추가 모델 요청 없음')
            return values,{'usage':data.get('usage',{}),'provider':data.get('provider'),
                'response_id':data.get('id'),'recovered_from_saved_response':True}
    # Preserve the original request epoch on same-input retries. Rewriting it
    # would incorrectly hide completed responses saved before a parser fix.
    if not manifest.exists() or not same_signature(old.get('signature'),signature):
        atomic_json(manifest,{'signature':signature,'response_format':payload['response_format']})
    return chat(payload,key,folder,ids,kind,progress)


def chat(payload,key,folder,ids,kind,progress):
    from speech import read_stream
    folder.mkdir(exist_ok=True,parents=True)
    label='Qwen 화자 번호' if kind=='speakers' else 'Gemini 구간 번역'
    for attempt in range(1,4):
        started=time.monotonic();stem=f'{time.time_ns()}-{attempt}'
        request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(payload).encode(),
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Title':'Fast Dubbing staged speech'})
        def activity(event):
            count=min(len(ids),event.get('received_segments',0))
            elapsed=round(time.monotonic()-started)
            chars=event.get('content_characters',0)
            invalid=event.get('unrecognized_response_regions',0)
            if kind=='speakers' and invalid and not count:
                atomic_json(folder/f'{stem}.validation.json',{'error':'Qwen returned input regions without speaker assignments',
                    'received_characters':chars,'unrecognized_regions':invalid})
                progress(0,f'{label} 응답 형식 오류 · 배정 없는 입력 복사 {invalid}개 수신 · {elapsed}초')
                raise RuntimeError('Qwen copied input intervals instead of speaker IDs. The raw stream was preserved.')
            if not count and not chars:detail=f'{label} 첫 결과 대기 · {elapsed}초'
            elif not count:detail=f'{label} 응답 수신 {chars:,}자 · 유효 배정 0/{len(ids)} · {elapsed}초'
            else:detail=f'{label} {count}/{len(ids)}개 수신 · {elapsed}초'
            progress(count/max(1,len(ids))*100,detail)
        def received(count):
            progress(min(len(ids),count)/max(1,len(ids))*100,
                f'{label} {min(len(ids),count)}/{len(ids)}개 수신 · {round(time.monotonic()-started)}초')
        try:
            with urllib.request.urlopen(request,timeout=900) as response,(folder/f'{stem}.sse').open('wb') as record:
                data=read_stream(response,received,record,on_activity=activity,expected_region_ids=ids)
            atomic_json(folder/f'{stem}.json',data)
            choice=data.get('choices',[{}])[0]
            if data.get('error') or choice.get('error'):raise RuntimeError(f'{label} provider error. The raw response was preserved.')
            if choice.get('finish_reason')!='stop':raise RuntimeError(f'{label} response is incomplete ({choice.get("finish_reason")}). The raw response was preserved.')
            try:values=parse_map(choice['message']['content'],ids,kind)
            except (ValueError,KeyError,TypeError) as error:
                atomic_json(folder/f'{stem}.validation.json',{'error':str(error),'finish_reason':choice.get('finish_reason')})
                raise RuntimeError(f'{label} JSON validation failed: {error}. The raw response was preserved.') from None
            return values,{'usage':data.get('usage',{}),'elapsed_seconds':time.monotonic()-started,
                'provider':data.get('provider'),'response_id':data.get('id')}
        except urllib.error.HTTPError as error:
            body=error.read().decode('utf8','replace').replace(key,'[REDACTED]')
            atomic_json(folder/f'{stem}.http-error.json',{'status':error.code,'response':body})
            if (error.code==429 or error.code>=500) and attempt<3:time.sleep(2**attempt);continue
            detail=''
            try:
                outer=json.loads(body).get('error',{});detail=outer.get('message','')
                native=outer.get('metadata',{}).get('raw','')
                for line in native.splitlines():
                    if line.startswith('data:'):
                        detail=json.loads(line[5:]).get('error',{}).get('message',detail);break
            except (ValueError,AttributeError,TypeError):pass
            detail=' '.join(str(detail).split())[:250]
            raise RuntimeError(f'{label} HTTP {error.code}'+(f': {detail}' if detail else '')+'. 원본 응답을 보관했습니다.') from None


def transcribe(clean,work,output,cfg,progress):
    clean,work=Path(clean),Path(work);work.mkdir(parents=True,exist_ok=True)
    info=sf.info(clean);duration=info.duration
    raw=transcribe_words(clean,clean.parent/CACHE_DIRECTORY,cfg['key'],progress=lambda p,d:progress(p*.28,d))
    rows=regions(raw,duration);ids=[r['id'] for r in rows]
    audio_folder=work/'speaker-audio';audio_folder.mkdir(exist_ok=True)
    audio,manifest=encode_audio(clean,audio_folder,digest(clean),lambda p,d:progress(29,'Qwen 화자 분석용 전체 대사 오디오 준비'),
        base64_limit=POLICY['qwen_audio_base64_budget_bytes']) if rows else (None,None)
    inputs=[{k:r[k] for k in ('id','start','end','text')} for r in rows]
    identity={'source':raw.get('signature'),'regions':inputs,'audio':manifest,'prompt':SPEAKER_PROMPT,
        'model':QWEN,'reasoning':{'enabled':False},'version':POLICY['version'],'message_layout':SPEAKER_MESSAGE_LAYOUT}
    cache=work/'qwen-speakers.json';saved=json.loads(cache.read_text('utf8')) if cache.exists() else {}
    if same_signature(saved.get('signature'),identity):
        voices=parse_map(json.dumps(saved['speakers']),ids,'speakers');qwen_receipt=saved['receipt']
        progress(60,'Qwen 화자 번호 캐시 재사용')
    elif rows:
        payload=speaker_payload(inputs,audio,manifest)
        voices,qwen_receipt=stage_chat(payload,cfg['key'],work,identity,ids,'speakers',lambda p,d:progress(30+p*.30,d))
        atomic_json(cache,{'signature':identity,'speakers':voices,'receipt':qwen_receipt})
    else:voices={};qwen_receipt={'usage':{}}
    for row in rows:
        voice=voices[str(row['id'])];row['speaker']='unknown' if voice is None else f'qwen_{voice}'
    bounded,measurements=correct_edges(clean,rows,lambda p,d:progress(61,d.replace('MAI','Fish')))
    draft_count=len(rows)
    bounded,measurements,merge_changes=merge_voice_turns(bounded,measurements)
    rows=bounded;ids=[r['id'] for r in rows]
    progress(63,f'같은 화자 0.8초 이내 병합 · {draft_count} → {len(rows)}개')
    prompt=TRANSLATION_PROMPT.format(language=LANGUAGES[cfg['language']])
    context=[{'id':r['id'],'speaker':r['speaker'],'text':r['text']} for r in rows]
    translation_identity={'source':raw.get('signature'),'regions':context,'prompt':prompt,'model':GEMINI,'language':cfg['language']}
    cache=work/'gemini-translations.json';saved=json.loads(cache.read_text('utf8')) if cache.exists() else {}
    if saved.get('signature')==translation_identity:
        translations=parse_map(json.dumps(saved['translations']),ids,'translations');gemini_receipt=saved['receipt']
        progress(95,'Gemini 번역 캐시 재사용')
    elif rows:
        from speech import MAX_OUTPUT_TOKENS
        payload={'model':GEMINI,'max_tokens':MAX_OUTPUT_TOKENS,'reasoning':{'effort':'minimal'},'provider':{'require_parameters':True},
            'stream':True,'stream_options':{'include_usage':True},'response_format':schema('translations'),
            'messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps(context,ensure_ascii=False,separators=(',',':'))}]}
        translations,gemini_receipt=stage_chat(payload,cfg['key'],work,translation_identity,ids,'translations',lambda p,d:progress(64+p*.31,d))
        atomic_json(cache,{'signature':translation_identity,'translations':translations,'receipt':gemini_receipt})
    else:translations={};gemini_receipt={'usage':{}}
    segments=[];excluded=[];windows=[]
    for row in bounded:
        translated=translations[str(row['id'])]
        if row.get('keep_original') or row['end']<=row['start'] or translated is None or not any(c.isalnum() for c in spoken(translated)):
            excluded.append({'start':row['start'],'end':row['end'],'reason':'not_selected_for_dubbing'});continue
        segments.append({'start':row['start'],'end':row['end'],'speaker_id':row['speaker'],
            'original_text':row['text'],'translated_text':translated,'source_region_id':row['id']})
        windows.append({'start':row['start'],'end':row['end'],'start_frame':round(row['start']*SR),
            'end_frame':round(row['end']*SR),'fade_in_frames':0,'fade_out_frames':0,'boundary_policy':AUTHORITY})
    timeline={'identity':identity,'duration':duration,'boundary_authority':AUTHORITY,'external_embedding_model':None,
        'turns':[{'start':r['start'],'end':r['end'],'speaker':r['speaker']} for r in bounded]}
    result={'model':GEMINI,'segments':segments,'excluded_intervals':excluded,'source_regions':bounded,
        'speaker_timeline':timeline,'reference_windows':windows,'boundary_measurements':measurements,
        'boundary_policy':BOUNDARY_POLICY,'boundary_authority':AUTHORITY,'speaker_authority':QWEN,
        'original_text_source':ASR,'source_asr_model':ASR,'source_asr_input_policy':INPUT_POLICY,
        'reference_speaker_boundaries':AUTHORITY,'word_grouping':POLICY['post_qwen_merge'],'speaker_turn_merges':merge_changes,
        'source_draft_region_count':draft_count,'target_language':cfg['language'],'source_languages':[raw.get('language') or 'unknown'],
        'duration_seconds':duration,'time_unit':'seconds','llm_timestamps':False,'asr_sent_to_llm':True,'asr_used':True,
        'transcription_mode':'fish_transcription_qwen_voice_ids_gemini_translation','translation_output':'region_id_to_translation_or_null_only',
        'models':{'source_asr':ASR,'speakers':QWEN,'translation':GEMINI},'request_counts':{'source_asr':int(bool(rows)),'speakers':int(bool(rows)),'translation':int(bool(rows))},
        'qwen_receipt':qwen_receipt,'gemini_receipt':gemini_receipt,'usage':gemini_receipt['usage'],'llm_audio_input':None,
        'transition_policy':'standard_external_fade','reference_tail_policy':{'enabled':False,'method':'fish_clock_qwen_voices_waveform_edges'},'unmatched_segments':[]}
    atomic_json(output,result);progress(100,f'Fish 전사 → Qwen 화자 → Gemini 번역 완료 · {len(segments)}개 대사')
    return result
