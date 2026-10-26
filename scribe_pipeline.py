"""Scribe v2 owns transcription, speakers and word times; Gemini translates text only."""
from pathlib import Path
import json
import soundfile as sf
from core import GEMINI,atomic_json,LANGUAGE_CATALOG
from scribe import transcribe_words,MODEL
from fish_pipeline import regions,correct_fish_edges
from asr_word_edges import repair as repair_word_edges,POLICY as WORD_EDGE_POLICY

AUTHORITY='scribe_v2'
OPTIONS={'elevenlabs':{'diarize':True,'tag_audio_events':False}}
POLICY={'version':19,'model':MODEL,'input':'whole_separated_dialogue_original_clock',
    'timestamp_granularities':['word'],'provider_options':OPTIONS,'speakers':'scribe_native_ids',
    'grouping_gap_seconds':.8,'boundaries':'word_times_then_local_waveform_edges',
    'delivery_cues':'none_no_cue_model_enabled','translation':'gemini_text_only',
    'qwen':False,'forced_alignment':False,'word_edges':WORD_EDGE_POLICY}
PROMPT="""Translate each numbered original dialogue faithfully into natural spoken TARGET_LANGUAGE. Preserve names, meaning, hesitations, unfinished speech and expressive punctuation, including ?, !, ?!, ~ and .... Scribe owns the source words, speakers, IDs and timestamps; never change them, split or merge entries, or move words between IDs. This is TEXT TRANSLATION ONLY. No audio is supplied. Do not infer emotion, add acting cues, invent speaker identities or nicknames, or rewrite the original transcript. Return only one JSON object mapping every supplied dialogue ID to its translated string, or null for a purely nonverbal/background sound that needs no dubbing. Preserve meaningful short replies. No source text, speaker IDs, timestamps or explanations in the output."""


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


def translations(content,rows):
    value=json.loads(content)
    ids=[str(row['id']) for row in rows]
    if not isinstance(value,dict) or set(value)!=set(ids):raise ValueError('Translation response must contain every supplied dialogue ID')
    if any(text is not None and (not isinstance(text,str) or not text.strip()) for text in value.values()):raise ValueError('Expected a translated string or null for each dialogue')
    return {'values':[value[i] for i in ids],'speaker_corrections':{}}


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
        return {'type':'json_schema','json_schema':{'name':'dialogue_translations','strict':True,
            'schema':{'type':'object','properties':{i:{'type':['string','null']} for i in ids},
                'required':ids,'additionalProperties':False}}}
    result=translate(clean,work,output,cfg,progress,source_loader=load,
        region_builder=lambda raw,duration:raw['segments'],source_model=MODEL,authority=AUTHORITY,
        prompt=PROMPT.replace('TARGET_LANGUAGE',LANGUAGES[cfg['language']]),boundary_builder=boundaries,
        cache_directory='scribe-source-v1',source_input_policy=POLICY,prefix='scribe',
        response_parser=translations,format_builder=schema)
    raw=holder['raw']
    result.update(model=GEMINI,preserve_source_speaker_ids=True,speaker_authority=MODEL,
        source_transcription_model=MODEL,transcription_mode='scribe_v2_native_words_speakers_gemini_text_translation_only',
        native_speaker_mapping=raw['native_speaker_mapping'],source_word_boundary_repairs=raw['source_word_boundary_repairs'],source_word_edge_repairs=raw['source_word_edge_repairs'],
        corrected_source_gap_merges=holder['merges'],source_asr_model=MODEL,
        asr_used=True,asr_sent_to_llm=True,llm_timestamps=False,llm_audio_input=None,
        translation_output='region_id_to_translation_or_null_only',
        models={'source_asr':MODEL,'speakers':MODEL,'translation':GEMINI},
        request_counts={'source_asr_api':raw['_api_requests'],'translation':result['request_count'],'performance_review':0,'delivery_cues':0},
        reference_tail_policy={'enabled':False,'method':'scribe_local_acoustic_edges'})
    atomic_json(output,result)
    return result
