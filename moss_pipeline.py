"""MOSS owns transcription, speakers and boundaries; Gemini only translates."""
from pathlib import Path
import base64,hashlib,json,math,subprocess,time
import soundfile as sf
from core import ROOT,SR,GEMINI,LANGUAGES,atomic_json

MODEL='OpenMOSS-Team/MOSS-Transcribe-Diarize'
AUTHORITY='moss_transcribe_diarize'
POLICY={'version':2,'model':MODEL,'input':'whole_separated_dialogue_original_clock',
    'prompt':'native_format_with_detailed_audible_acting_cues_english_brackets','decoding':'greedy','source_boundaries':'exact_model_output',
    'boundary_correction':False,'speaker_correction':False,'post_merge':False,
    'translation_output':'region_id_to_translation_or_null_only'}


def source_signature(clean):
    model=ROOT/'models/moss-transcribe-diarize'
    return {'policy':POLICY,'audio_sha256':hashlib.sha256(Path(clean).read_bytes()).hexdigest(),
        'model_files':[[p.name,p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(model.iterdir()) if p.is_file()],
        'worker_sha256':hashlib.sha256((ROOT/'moss_source_worker.py').read_bytes()).hexdigest(),
        'prompt_sha256':hashlib.sha256((ROOT/'moss_prompt.py').read_bytes()).hexdigest()}


def source(clean,folder,progress):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);signature=source_signature(clean)
    cache=folder/'source.json'
    if cache.exists():
        data=json.loads(cache.read_text('utf8'))
        if data.get('signature')==signature:progress(100,'MOSS 전사·화자·시간 캐시 재사용');return data
    # Different input or worker versions cannot accidentally reuse a previous raw response.
    run=folder/hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    run.mkdir(exist_ok=True)
    if not (run/'result.json').exists():
        runtime=ROOT/'.moss-diarize-env/Scripts/python.exe'
        if not runtime.exists():raise RuntimeError('MOSS transcription runtime is missing.')
        with (run/'worker.log').open('w',encoding='utf8') as log,subprocess.Popen(
            [str(runtime),'-u',str(ROOT/'moss_source_worker.py'),str(clean),str(run)],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace') as proc:
            started=time.monotonic()
            for line in proc.stdout:
                log.write(line);log.flush()
                try:event=json.loads(line)
                except ValueError:continue
                stage=event.get('stage')
                if stage=='generating':progress(20,f'MOSS 전사·화자 분석 · {event["generated_tokens"]}토큰 생성 · {time.monotonic()-started:.0f}초')
                elif stage=='loading':progress(1,'MOSS 모델 로드 중')
                elif stage=='loaded':progress(4,'MOSS 전체 대사 오디오 분석 준비')
                elif stage=='input_ready':progress(8,f'MOSS 전체 대사 분석 · 입력 {event["prompt_tokens"]}토큰')
                elif stage=='done':progress(100,f'MOSS {event["segments"]}개 구간 인식 완료')
            if proc.wait():raise RuntimeError(f'MOSS transcription or speaker analysis failed. Raw logs were preserved in {run.name}/worker.log.')
    raw=json.loads((run/'result.json').read_text('utf8'))
    if raw['generated_tokens']>=raw['max_new_tokens']:raise RuntimeError('MOSS transcription reached the output limit. The raw response was preserved.')
    data={**raw,'signature':signature};atomic_json(cache,data);return data


def regions(raw,duration):
    rows=[]
    for i,s in enumerate(raw['segments']):
        a,b=s.get('start'),s.get('end');text=s.get('text');voice=s.get('speaker')
        if (type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a+b)
            or not 0<=a<b<=duration or not isinstance(text,str) or not text.strip()
            or not isinstance(voice,str) or not voice.strip()):
            raise RuntimeError(f'Invalid MOSS transcript, speaker, or timestamps for dialogue {i+1}. The raw response was preserved.')
        rows.append({'id':i,'start':a,'end':b,'speaker':voice,'text':text})
    if not rows and raw.get('text','').strip():raise RuntimeError('Could not read valid dialogue segments from the MOSS response. The raw response was preserved.')
    rows.sort(key=lambda r:(r['start'],r['end']))
    # Actual overlapping turns cannot form a clean single-speaker reference.
    # Preserve both in the source; never shorten, shift or relabel the MOSS output.
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            if right['start']>=left['end']:break
            left['keep_original']=right['keep_original']=True
    return rows


def transcribe(clean,work,output,cfg,progress,*,source_loader=None,boundary_builder=None,authority=AUTHORITY,audio_builder=None,translation_prompt=None,source_cue_builder=None,region_builder=None,source_text_builder=None):
    from fish_qwen_pipeline import TRANSLATION_PROMPT,parse_map,schema,stage_chat
    from rich_transcript import spoken
    from speech import MAX_OUTPUT_TOKENS
    clean,work=Path(clean),Path(work);work.mkdir(parents=True,exist_ok=True)
    if source_loader:
        raw=source_loader(clean,cfg,lambda p,d:progress(p*.5,d));speaker_model=raw['model']
    elif cfg.get('moss_api_key'):
        from moss_pro import source as pro_source,MODEL as speaker_model
        raw=pro_source(clean,clean.parent/'moss-pro-source-v1',cfg['moss_api_key'],cfg.get('moss_keyterms','[]'),lambda p,d:progress(p*.65,d))
    else:
        speaker_model=MODEL
        raw=source(clean,clean.parent/'moss-source-v1',lambda p,d:progress(p*.65,d))
    duration=sf.info(clean).duration;rows=(region_builder or regions)(raw,duration);ids=[r['id'] for r in rows]
    measurements=[]
    if boundary_builder:
        rows,measurements=boundary_builder(clean,rows,lambda p,d:progress(50+max(0,min(20,p-30))*.75,d))
    ids=[r['id'] for r in rows]
    source_cue_receipt=None
    source_processor=source_text_builder or source_cue_builder
    if source_processor:
        rows,source_cue_receipt=source_processor(clean,rows,work,cfg,lambda p,d:progress(65+p*.12,d))
    prompt=(translation_prompt or TRANSLATION_PROMPT).format(language=LANGUAGES[cfg['language']])+\
        '\nPreserve expressive punctuation and audible delivery cues without flattening their intensity. Preserve supplied [] cues at their corresponding positions; do not invent or intensify acting cues.'
    context=[{k:r[k] for k in ('id','speaker','text')} for r in rows]
    audio_manifest=audio_builder(clean,work,raw['signature']) if audio_builder and rows else None
    if audio_manifest:
        # Input clock hints only locate each utterance in the complete recording.
        # These values cannot be emitted or modified by the translation schema.
        context=[{**c,'start':r['start'],'end':r['end']} for c,r in zip(context,rows)]
        atomic_json(work/'gemini-audio-input.json',audio_manifest)
    identity={'source':raw['signature'],'text_authority':speaker_model,'regions':context,'prompt':prompt,'model':GEMINI,'language':cfg['language']}
    if audio_manifest:identity['audio']={k:v for k,v in audio_manifest.items() if k!='path'}
    cache=work/'gemini-translations.json';saved=json.loads(cache.read_text('utf8')) if cache.exists() else {}
    if saved.get('signature')==identity:
        translations=parse_map(json.dumps(saved['translations']),ids,'translations');receipt=saved['receipt'];requests=0
        progress(95,'Gemini 번역 캐시 재사용')
    elif rows:
        payload={'model':GEMINI,'max_tokens':MAX_OUTPUT_TOKENS,'reasoning':{'effort':'minimal'},'provider':{'require_parameters':True},
            'stream':True,'stream_options':{'include_usage':True},'response_format':schema('translations'),
            'messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps(context,ensure_ascii=False,separators=(',',':'))}]}
        if audio_manifest:
            payload['messages'][1]['content']=[{'type':'text','text':payload['messages'][1]['content']},
                {'type':'input_audio','input_audio':{'format':audio_manifest['format'],
                    'data':base64.b64encode(Path(audio_manifest['path']).read_bytes()).decode()}}]
        translations,receipt=stage_chat(payload,cfg['key'],work,identity,ids,'translations',lambda p,d:progress((78+p*.17) if source_processor else (66+p*.29),d));requests=1
        atomic_json(cache,{'signature':identity,'translations':translations,'receipt':receipt})
    else:translations={};receipt={'usage':{}};requests=0
    segments=[];windows=[];excluded=[]
    for row in rows:
        text=translations[str(row['id'])]
        if row.get('keep_original') or text is None or not any(c.isalnum() for c in spoken(text)):
            excluded.append({'start':row['start'],'end':row['end'],'reason':'moss_overlapping_turns' if row.get('keep_original') else 'not_selected_for_dubbing'});continue
        segments.append({'start':row['start'],'end':row['end'],'speaker_id':row['speaker'],
            'original_text':row['text'],'translated_text':text,'source_region_id':row['id']})
        windows.append({'start':row['start'],'end':row['end'],'start_frame':round(row['start']*SR),
            'end_frame':round(row['end']*SR),'fade_in_frames':0,'fade_out_frames':0,'boundary_policy':authority})
    timeline={'identity':raw['signature'],'duration':duration,'boundary_authority':authority,
        'external_embedding_model':None,'turns':[{k:r[k] for k in ('start','end','speaker')} for r in rows]}
    result={'model':GEMINI,'segments':segments,'excluded_intervals':excluded,'source_regions':rows,
        'speaker_timeline':timeline,'reference_windows':windows,'boundary_measurements':measurements,
        'boundary_policy':raw['signature'].get('policy',POLICY),'boundary_authority':authority,'speaker_authority':speaker_model,
        'original_text_source':speaker_model,'source_asr_model':speaker_model,'source_asr_input_policy':raw['signature'].get('policy',POLICY),
        'reference_speaker_boundaries':authority,'target_language':cfg['language'],'source_languages':[raw.get('language') or 'unknown'],
        'duration_seconds':duration,'time_unit':'seconds','llm_timestamps':False,'asr_sent_to_llm':True,'asr_used':True,
        'transcription_mode':('mai_transcript_speakers_acoustic_boundaries_gemini_audio_cues_translation' if audio_manifest else 'mai_transcript_speakers_acoustic_boundaries_gemini_translation') if source_loader else 'moss_native_transcript_speakers_timestamps_gemini_translation',
        'translation_output':POLICY['translation_output'],'models':{'source_asr':speaker_model,'speakers':speaker_model,'translation':GEMINI},
        'request_counts':{'source_asr_api':raw.get('_api_requests',int(bool(cfg.get('moss_api_key')) and not raw.get('_cache_reused',False))),'speakers_api':0,'translation':requests},
        'combined_transcription_and_speakers':True,'fish_asr_requests':0,
        'gemini_receipt':receipt,'usage':receipt.get('usage',{}),'llm_audio_input':({k:v for k,v in audio_manifest.items() if k!='path'} if audio_manifest else None),
        'transition_policy':'standard_external_fade','reference_tail_policy':{'enabled':False,'method':'mai_waveform_edges_before_translation' if source_loader else 'moss_exact_native_times'},
        'unmatched_segments':[]}
    if source_cue_receipt is not None:
        from qwen_delivery_cues import MODEL as CUE_MODEL
        role='punctuation' if source_text_builder else 'delivery_cues'
        result['models'][role]=CUE_MODEL
        result['request_counts'][role]=source_cue_receipt.get('requests',0)
        result['punctuation_receipt' if source_text_builder else 'delivery_cue_receipt']=source_cue_receipt
        result['transcription_mode']='mai_transcript_speakers_acoustic_boundaries_qwen_source_cues_gemini_text_translation'
    if raw.get('_speaker_repair') is not None:
        result['source_speaker_repair']=raw['_speaker_repair']
        result['models']['speaker_refinement']=raw['_speaker_repair']['policy']['model']
        result['speaker_timeline']['external_embedding_model']=raw['_speaker_repair']['policy']['model']
    atomic_json(output,result);progress(100,f'{"MAI 전사·화자·경계 보정" if source_loader else "MOSS 전사·화자·시간"} → {"문장부호 보정 → " if source_text_builder else "Qwen 원문 단서 → " if source_cue_builder else ""}Gemini 번역 완료 · {len(segments)}개 대사');return result
