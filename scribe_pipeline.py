"""Scribe v2 owns transcription, speakers and word times; Gemini translates and adds conservative audio-grounded tags."""
from pathlib import Path
import json,re,hashlib
import soundfile as sf
from core import GEMINI,atomic_json,LANGUAGE_CATALOG
from scribe import transcribe_words,MODEL
from fish_pipeline import regions,correct_fish_edges
from asr_word_edges import repair as repair_word_edges,POLICY as WORD_EDGE_POLICY

AUTHORITY='scribe_v2'
OPTIONS={'elevenlabs':{'diarize':True,'tag_audio_events':False}}
POLICY={'version':21,'model':MODEL,'input':'whole_separated_dialogue_original_clock',
    'timestamp_granularities':['word'],'provider_options':OPTIONS,'speakers':'scribe_native_ids',
    'grouping_gap_seconds':.8,'boundaries':'word_times_then_local_waveform_edges',
    'delivery_cues':'gemini_audio_grounded_sentence_and_inline_delivery_descriptions','translation':'gemini_audio_and_fixed_text',
    'qwen':False,'forced_alignment':False,'word_edges':WORD_EDGE_POLICY}
PROMPT="""Translate each fixed dialogue into natural spoken TARGET_LANGUAGE using the attached original-clock audio. Preserve meaning, names and expressive punctuation. Keep Scribe source words, speakers, IDs and times unchanged.
Express the audible performance as fully as useful in English [] cues, sentence by sentence and at meaningful changes inside a sentence. Describe HOW the words physically sound: breath held or released, groans mixed with speech, tight/pressed or soft onset, trembling, broken voice, articulation, pauses, rhythm, pace, intensity and pitch movement, only when actually heard. Do not compress distinct audible details into generic emotion labels or force a fixed descriptor count. Concrete performance descriptions are welcome, but do not invent unseen bodily situations, plot or causes from dialogue meaning. Describe the sound neutrally; plain or uncertain speech needs no invented emotion.
Represent audible cut-offs, swallowed endings, interrupted syllables, stutters, restarts and unfinished words in the TARGET dialogue itself, not just in cues or with polished complete words. For example, Korean 잠깐 that is actually cut off before completion may be 잠ㄲ…; this is an example, never a default replacement. Do not invent truncation when speech is complete. Preserve audible fragments as naturally as the target writing system permits.
Return only {"region_id":{"t":"translation with inline [English cues]","c":[{"i":0,"c":"the identical English cue"}]}}. Each i is the supplied source unit index BEFORE which that cue belongs; the server inserts it into the original. Put the identical cues in t at the corresponding translated positions, in the same order. Use [] only, no XML. c=[] for speech needing no cue. Do not repeat the original transcript, timestamps, speakers or explanations. t may be null only for nonverbal/background-only sound, never a meaningful short reply."""


def source(clean,folder,cfg,progress):
    folder=Path(folder)
    cached=folder/'scribe.json'
    before=json.loads(cached.read_text('utf8')).get('signature') if cached.exists() else None
    language=cfg.get('source_language')
    language=LANGUAGE_CATALOG.get('source_language_policy',{}).get('hint_aliases',{}).get(language,language)
    raw=transcribe_words(clean,folder,cfg['key'],language=language,
        progress=lambda p,d:progress(p,'전체 음성 전사·화자 분석 중' if p<100 else '전사·화자 분석 완료'),
        model=MODEL,provider_options=OPTIONS)
    mapping={};words=[]
    for original in raw['words']:
        word=dict(original);voice=word.get('speaker_id',word.get('speaker'))
        if voice is None:word['speaker']='unknown'
        else:
            key=json.dumps(voice,ensure_ascii=False)
            if key not in mapping:mapping[key]=str(len(mapping))
            word['speaker']=mapping[key]
        words.append(word)
    words,word_edge_repairs=repair_word_edges(clean,words)
    grouped=[{**row,'id':i} for i,row in enumerate(regions({**raw,'words':words},sf.info(clean).duration,clean=clean))]
    from source_word_boundaries import repair_japanese_boundaries
    grouped,word_boundary_repairs=repair_japanese_boundaries(grouped,raw.get('text',''))
    from mai_native_pipeline import record_auto_language
    record_auto_language(raw,cfg)
    return {**raw,'model':MODEL,'segments':grouped,
        'source_word_boundary_repairs':word_boundary_repairs,'source_word_edge_repairs':word_edge_repairs,
        'signature':{'policy':POLICY,'asr':raw['signature']},
        'native_speaker_mapping':[{'native_id':json.loads(k),'speaker_id':v} for k,v in mapping.items()],
        '_api_requests':int(before!=raw['signature'])}


def cue_units(text):
    return [(m.start(),m.group()) for m in re.finditer(r"[A-Za-z0-9]+(?:['’.,][A-Za-z0-9]+)*|[^\s]",text)]


def clean_cue(value):
    if not isinstance(value,str):return None
    value=' '.join(value.strip().strip('[]').split()).lower()
    if not value or not value.isascii() or not any(c.isalpha() for c in value) or any(c in value for c in '[]<>'):return None
    return value


def translations(content,rows):
    value=json.loads(content)
    ids=[str(row['id']) for row in rows];by_id={str(r['id']):r for r in rows}
    if not isinstance(value,dict) or set(value)!=set(ids):raise ValueError('Translation response must contain every supplied dialogue ID')
    values=[];cues={};entries={};rejected=[]
    for identifier in ids:
        entry=value[identifier]
        if isinstance(entry,dict):
            if 't' not in entry:raise ValueError('Missing translated dialogue')
            text=entry['t'];raw_tags=entry.get('c',[])
        else:text=entry;raw_tags=[]
        if text is not None and (not isinstance(text,str) or not text.strip()):raise ValueError('Expected a translated string or null for each dialogue')
        units=cue_units(by_id[identifier]['text']);tags=[]
        for raw in raw_tags if isinstance(raw_tags,list) else []:
            index=raw.get('i',0) if isinstance(raw,dict) else 0
            if isinstance(index,str) and index.isdecimal():index=int(index)
            tag=clean_cue(raw.get('c') if isinstance(raw,dict) else raw)
            if tag is None or type(index) is not int or not 0<=index<len(units):
                rejected.append({'id':identifier,'reason':'invalid_english_cue_or_source_position'});continue
            item={'i':index,'c':tag}
            if item not in tags:tags.append(item)
        tags.sort(key=lambda item:item['i'])
        if text is None:tags=[]
        else:
            untagged=re.sub(r'\[[^\]\n]*\][ \t]*','',text).strip()
            if not untagged:raise ValueError('Translation contains no dialogue')
            normalized=re.sub(r'\[([^\]\n]*)\]',lambda m:'['+(clean_cue(m[1]) or '')+']',text).strip()
            actual=re.findall(r'\[([^\]\n]*)\]',normalized)
            expected=[item['c'] for item in tags]
            if actual==expected:text=normalized
            else:
                # Recover an omitted/invalid translated cue layout without
                # discarding valid dialogue or launching a separate repair model.
                text=(' '.join('['+tag+']' for tag in expected)+' '+untagged).strip()
                if actual or expected:rejected.append({'id':identifier,'reason':'translated_cue_layout_recovered_at_start'})
        cues[identifier]=tags;values.append(text)
        entries[identifier]={'t':text,'c':tags}
    return {'values':values,'speaker_corrections':{},'delivery_cues':cues,
        'response_entries':entries,'rejected_delivery_tags':rejected}


def tagged_source_rows(rows,parsed):
    result=[]
    for row in rows:
        tags=parsed.get('delivery_cues',{}).get(str(row['id']),[])
        if not tags:result.append(dict(row));continue
        units=cue_units(row['text']);insertions={}
        for tag in tags:insertions.setdefault(units[tag['i']][0],[]).append('['+tag['c']+']')
        text=row['text']
        for position,labels in sorted(insertions.items(),reverse=True):text=text[:position]+' '.join(labels)+' '+text[position:]
        result.append({**row,'text':text,'uncued_text':row['text']})
    return result,[]


def cue_audio(clean,work,source_identity):
    from scribe import prepare_input
    folder=Path(work)/'gemini-delivery-audio';folder.mkdir(parents=True,exist_ok=True)
    with Path(clean).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    audio,format,duration=prepare_input(Path(clean),folder,digest)
    info=sf.info(audio)
    return {'path':str(audio),'format':format,'sample_rate':info.samplerate,'channels':info.channels,
        'duration_seconds':duration,'original_clock':True,'sha256':hashlib.sha256(audio.read_bytes()).hexdigest()}


def transcribe(clean,work,output,cfg,progress):
    from mai_pipeline import translate
    from core import LANGUAGES
    from corrected_source_groups import merge_corrected
    holder={}
    def load(clean,folder,key,activity):
        holder['raw']=raw=source(clean,Path(clean).parent/'scribe-source-v1',cfg,lambda p,d:activity(p*.3,d))
        return raw,raw['signature']
    def boundaries(clean,rows,activity):
        corrected,audit=correct_fish_edges(clean,rows,activity)
        merged,merges=merge_corrected(corrected,POLICY['grouping_gap_seconds'])
        holder['rows']=merged;holder['merges']=merges
        return merged,audit+merges
    def schema():
        ids=[str(row['id']) for row in holder['rows']]
        item={'type':'object','properties':{'t':{'type':['string','null']},
            'c':{'type':'array','items':{'type':'object','properties':{'i':{'type':'integer'},'c':{'type':'string'}},'required':['i','c'],'additionalProperties':False}}},'required':['t','c'],'additionalProperties':False}
        return {'type':'json_schema','json_schema':{'name':'tagged_dialogue_translations','strict':True,
            'schema':{'type':'object','properties':{i:item for i in ids},
                'required':ids,'additionalProperties':False}}}
    result=translate(clean,work,output,cfg,progress,source_loader=load,
        region_builder=lambda raw,duration:raw['segments'],source_model=MODEL,authority=AUTHORITY,
        prompt=PROMPT.replace('TARGET_LANGUAGE',LANGUAGES[cfg['language']]),boundary_builder=boundaries,
        cache_directory='scribe-source-v1',source_input_policy=POLICY,prefix='scribe',
        response_parser=translations,format_builder=schema,region_reviser=tagged_source_rows,audio_builder=cue_audio,
        input_builder=lambda rows:[{**{k:r[k] for k in ('id','start','end','speaker','text')},
            'units':[[i,text] for i,(_,text) in enumerate(cue_units(r['text']))]} for r in rows])
    raw=holder['raw']
    result.update(model=GEMINI,preserve_source_speaker_ids=True,speaker_authority=MODEL,
        source_transcription_model=MODEL,transcription_mode='scribe_native_words_speakers_gemini_translation_audio_delivery_and_cutoffs',
        native_speaker_mapping=raw['native_speaker_mapping'],source_word_boundary_repairs=raw['source_word_boundary_repairs'],source_word_edge_repairs=raw['source_word_edge_repairs'],
        corrected_source_gap_merges=holder['merges'],source_asr_model=MODEL,
        asr_used=True,asr_sent_to_llm=True,llm_timestamps=False,
        translation_output='region_id_to_translation_or_null_only',
        models={'source_asr':MODEL,'speakers':MODEL,'translation':GEMINI,'delivery_cues':GEMINI},
        request_counts={'source_asr_api':raw['_api_requests'],'translation':result['request_count'],'performance_review':0,'delivery_cues':0},
        reference_tail_policy={'enabled':False,'method':'scribe_local_acoustic_edges'})
    result['delivery_cue_receipt']={'requests':0,'shared_with':'translation','policy':'audio_grounded_sentence_inline_descriptions_and_target_speech_fragments',
        'annotated_regions':sum(bool(re.search(r'\[[^\]]+\]',r['original_text'])) for r in result['segments'])}
    atomic_json(output,result)
    return result
