"""Gemini hears the recording directly; no upstream ASR or Qwen review."""
from pathlib import Path
import base64,hashlib,json,math,re
import soundfile as sf
from core import GEMINI,LANGUAGES,atomic_json

MODEL=GEMINI
AUTHORITY='gemini_native_audio'
POLICY={'version':2,'model':MODEL,'input':'whole_separated_dialogue_original_clock',
    'tasks':['transcription','acting_cues','speakers','nicknames','timestamps'],
    'external_asr':False,'speaker_count_limit':None,'cue_language':'English',
    'nicknames':'target_language','cue_evidence':'audible_vocal_execution_only_no_semantic_or_scene_inference','boundaries':'gemini_audio_times_then_local_waveform_edges',
    'audio':'native_flac_then_highest_fitting_mp3_sample_rate_last_resort'}
PROMPT="""Listen carefully to the ENTIRE supplied recording, including its end. Perform transcription, acting annotation, speaker diarization and speaker nicknaming directly from the audio. No ASR transcript is supplied. Do not translate the dialogue in this response.
Transcribe intelligible spoken words faithfully in their original language, including names, repetitions, hesitation, unfinished phrases and meaningful short replies. Use expressive punctuation (?, !, ?!, ~, ..., and language-appropriate equivalents), including inside sentences when audible intonation warrants it. Do not turn background sounds or pure nonverbal events into invented words. Never summarize or rush later parts because the recording is long.
Identify speakers primarily by stable audible voice traits: timbre, resonance, articulation, vocal texture and habitual rhythm. Remember previously heard voices and compare later appearances against clear earlier examples. Context is ONLY a last-resort tie-breaker for acoustically plausible matches; names, story roles and assumptions must never override voice evidence. Acting, pitch or emotion changes alone do not create a new identity. Reuse a stable nonnegative integer ID for the same voice throughout the recording; create as many IDs as needed, without a fixed speaker limit. Do not merge audibly distinct people just because they have similar pitch, age or gender. Use null when identity is genuinely uncertain.
AFTER deciding voice identities, give every identified speaker a distinct useful nickname in TARGET_LANGUAGE (maximum 64 characters). Prefer a supported character name or a recognizable role/relationship inferred from dialogue. Mark uncertain inferred names as tentative in TARGET_LANGUAGE. Invent a useful role-based label when no name is supported. Nicknames are display labels, never evidence for speaker identity.
Put English [acting/delivery cues] directly into EVERY segment's text and at meaningful changes within a segment. Describe both communicative intent AND the actual audible vocal execution: loudness, pitch and contour, pace, rhythm, articulation, softness/roughness/strain, tremor, breath, hesitation, pauses and changes over the line, as applicable. Do not reduce rich audible performance to one or two generic emotion/intent labels; describe all salient audible traits without inventing or exaggerating them. Do not output separate redundant cue explanations.
ACTING CUE EVIDENCE IS AUDIO ONLY. Do NOT derive emotion, intent, attitude, loudness or delivery tags from dialogue meaning, translation, plot, inferred relationships, scene expectations, or how an actor ought to say the words. Semantic context may help decipher words and create display nicknames, but it is NEVER evidence for an acting cue. Describe what is actually heard, not what the sentence suggests. Threatening words spoken softly must retain the audible softness; sad content spoken evenly must not acquire crying or sadness tags. An intent label is allowed only when audible prosody or vocal behavior supports it, alongside the actual vocal execution. Do not infer movements, facial expressions or off-screen events. Re-listen to each span before tagging; omit any uncertain trait and use a plain acoustic description rather than guessing an emotion. A mandatory cue is NOT permission to invent emotion: audibly neutral/even delivery should be described as such.
Always split at a real speaker change. Within one speaker prefer inline cues for ordinary emotional or flow changes. Split only at a completed meaningful phrase boundary or when cues alone are unlikely to reproduce the performance faithfully and separate local reference segments are needed. Never split inside a word, a number, a name or a dependent ending. Keep a tiny dependent fragment or hesitation with its same-speaker continuation; preserve independent brief answers. Do not force an unrelated completed sentence to merge just because it is short.
Return source_language as an ISO language code, speakers as id/nickname records, and segments in audio order. Each segment has start/end in seconds from the beginning of this exact audio, speaker ID (or null), and text with inline English cues. Cover the complete audible onset and ending, including weak consonants, without including another speaker. Do not reset the clock at pauses. Stay within AUDIO_DURATION_SECONDS. Return only the requested JSON, no analysis or prose. Empty segments are allowed only if there is no intelligible speech."""


def response_format(duration):
    voice={'type':['integer','null'],'minimum':0}
    schema={'type':'object','additionalProperties':False,
        'properties':{'source_language':{'type':'string'},
            'speakers':{'type':'array','items':{'type':'object','additionalProperties':False,
                'properties':{'id':{'type':'integer','minimum':0},'nickname':{'type':'string','minLength':1,'maxLength':64}},
                'required':['id','nickname']}},
            'segments':{'type':'array','items':{'type':'object','additionalProperties':False,
                'properties':{'start':{'type':'number','minimum':0,'maximum':duration},
                    'end':{'type':'number','minimum':0,'maximum':duration},'speaker':voice,
                    'text':{'type':'string','minLength':1}},'required':['start','end','speaker','text']}}},
        'required':['source_language','speakers','segments']}
    return {'type':'json_schema','json_schema':{'name':'audio_dialogue_analysis','strict':True,'schema':schema}}


def parse(content,duration):
    if isinstance(content,str):
        content=content.strip()
        if content.startswith('```'):
            content=re.sub(r'^```(?:json)?\s*|\s*```$','',content,flags=re.I)
        value=json.loads(content)
    else:value=content
    if not isinstance(value,dict) or not isinstance(value.get('segments'),list) or not isinstance(value.get('speakers'),list):
        raise ValueError('Expected source_language, speakers and segments from Gemini audio analysis')
    language=value.get('source_language')
    if not isinstance(language,str) or not language.strip():raise ValueError('Missing source language')
    names={}
    for person in value['speakers']:
        identifier=person.get('id');name=person.get('nickname')
        if type(identifier) is not int or identifier<0 or str(identifier) in names or not isinstance(name,str) or not 0<len(name.strip())<=64:
            raise ValueError('Invalid or duplicate speaker nickname')
        names[str(identifier)]=name.strip()
    rows=[]
    for i,segment in enumerate(value['segments']):
        a,b=segment.get('start'),segment.get('end');voice=segment.get('speaker');text=segment.get('text')
        if (type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a+b)
            or not 0<=a<b<=duration or not isinstance(text,str) or not text.strip()):
            raise ValueError(f'Invalid transcript or audio timestamp for dialogue {i+1}')
        if voice is not None and (type(voice) is not int or str(voice) not in names):
            raise ValueError(f'Missing speaker nickname for dialogue {i+1}')
        rows.append({'id':i,'start':float(a),'end':float(b),
            'speaker':'unknown' if voice is None else str(voice),'text':text.strip()})
    rows.sort(key=lambda r:(r['start'],r['end']))
    for i,row in enumerate(rows):row['id']=i
    return {'language':language.strip(),'speaker_names':names,'segments':rows}


def source(clean,folder,cfg,progress):
    import speech
    from scribe import prepare_input  # Shared format/size compression only; no ASR request.
    from mai_native_pipeline import record_auto_language
    clean=Path(clean);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with clean.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    duration=sf.info(clean).duration
    context={'TARGET_LANGUAGE':LANGUAGES[cfg['language']],'SOURCE_LANGUAGE_HINT':cfg.get('source_language'),
        'AUDIO_DURATION_SECONDS':duration,'TIMESTAMP_ORIGIN_SECONDS':0}
    signature={'policy':POLICY,'audio_sha256':digest,'context':context,'prompt':PROMPT}
    cache=folder/'source.json'
    if cache.exists():
        saved=json.loads(cache.read_text('utf8'))
        if saved.get('signature')==signature:
            record_auto_language(saved,cfg)
            progress(100,'Gemini 전사·감정 단서·화자·별명 캐시 재사용')
            return {**saved,'_cache_reused':True}
    progress(2,'Gemini 전체 대사 오디오 준비')
    audio,format,duration=prepare_input(clean,folder,digest)
    payload={'model':MODEL,'max_tokens':speech.MAX_OUTPUT_TOKENS,'reasoning':{'effort':'low'},
        'provider':{'require_parameters':True},'response_format':response_format(duration),
        'messages':[{'role':'system','content':PROMPT},{'role':'user','content':[
            {'type':'text','text':json.dumps(context,ensure_ascii=False)},
            {'type':'input_audio','input_audio':{'format':format,'data':base64.b64encode(audio.read_bytes()).decode()}}]}]}
    manifest={'path':str(audio),'format':format,'sample_rate':sf.info(audio).samplerate,
        'channels':sf.info(audio).channels,'duration_seconds':duration,'original_clock':True}
    atomic_json(folder/'audio-input.json',manifest)
    parsed,usage=speech.http_json(payload,cfg['key'],response_directory=folder/'responses',
        parser=lambda content:parse(content,duration),
        on_segments=lambda n:progress(5,f'Gemini 전사·감정 단서·화자·별명 · {n}개 대사 수신'))
    result={**parsed,'model':MODEL,'signature':signature,'usage':usage,'audio_input':manifest}
    record_auto_language(result,cfg)
    atomic_json(cache,result);progress(100,f'Gemini {len(parsed["segments"])}개 대사 분석 완료')
    return result


def transcribe(clean,work,output,cfg,progress):
    from mai_pipeline import translate,correct_edges
    from fish_pipeline import fixed_translations
    from fish_qwen_pipeline import schema
    holder={};work=Path(work)
    def load(clean,folder,key,activity):
        holder['source']=raw=source(clean,work/'gemini-source',cfg,lambda p,d:activity(p*.45,d))
        return raw,raw['signature']
    def regions(raw,duration):
        from moss_pipeline import regions as source_regions
        return source_regions(raw,duration)
    prompt="""Translate each numbered dialogue into natural spoken LANGUAGE. Source words, speakers and boundaries are already fixed from direct audio analysis. Preserve names, meaning, hesitation and expressive punctuation. Preserve EVERY English square-bracket cue verbatim at the corresponding position, including mid-sentence cues. Never invent, omit, translate or paraphrase cues. Return only a JSON object mapping every supplied region ID to its translation, or null for purely nonverbal/background sounds needing no dubbing. Preserve meaningful short replies. Do not output source text, speakers, timestamps or explanations.""".replace('LANGUAGE',LANGUAGES[cfg['language']])
    result=translate(clean,work,output,cfg,progress,source_loader=load,region_builder=regions,
        source_model=MODEL,authority=AUTHORITY,prompt=prompt,boundary_builder=correct_edges,
        cache_directory='gemini-source',source_input_policy=POLICY,prefix='gemini',
        response_parser=fixed_translations,format_builder=lambda:schema('translations'))
    raw=holder['source']
    translation_requests=result['request_count']
    source_requests=int(not raw.get('_cache_reused',False))
    result.update(model=MODEL,speaker_names=raw['speaker_names'],preserve_source_speaker_ids=True,
        speaker_authority=MODEL,source_transcription_model=MODEL,original_text_source=MODEL,
        source_asr_model=None,asr_used=False,asr_sent_to_llm=False,llm_timestamps=True,
        combined_transcription_and_speakers=True,additional_asr_requests=0,fish_asr_requests=0,
        transcription_mode='gemini_native_audio_transcription_cues_speakers_nicknames_then_text_translation',
        translation_output='region_id_to_translation_or_null_only',
        models={'source_analysis':MODEL,'speakers':MODEL,'delivery_cues':MODEL,'translation':MODEL},
        source_analysis_usage=raw['usage'],llm_audio_input=raw['audio_input'],
        request_count=source_requests+translation_requests,
        request_counts={'source_analysis':source_requests,'translation':translation_requests,'source_asr_api':0,'speakers_api':0})
    atomic_json(output,result)
    return result
