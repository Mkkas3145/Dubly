"""Scribe words/clocks, Qwen emotion turns and voice correction, Gemini translation."""
from pathlib import Path
import json
import soundfile as sf
from core import GEMINI,QWEN,atomic_json,LANGUAGE_CATALOG
from qwen_performance_turns import review as review_performance,POLICY as PERFORMANCE_POLICY
from scribe import transcribe_words,MODEL
from fish_pipeline import regions,correct_fish_edges,TEXT_PROMPT

AUTHORITY='scribe_v2'
OPTIONS={'elevenlabs':{'diarize':True,'tag_audio_events':False}}
POLICY={'version':13,'model':MODEL,'input':'whole_separated_dialogue_original_clock',
    'timestamp_granularities':['word'],'provider_options':OPTIONS,
    'speakers':'qwen_audio_review_existing_or_new_ids','grouping_gap_seconds':.8,'punctuation_timestamps':'text_only_never_extend_spoken_boundaries','post_merge':'same_speaker_final_corrected_boundaries_before_qwen_emotion_splits',
    'performance_review':PERFORMANCE_POLICY,
    'boundaries':'local_acoustic_edges_unique_point_reply_recovery_and_verified_collapsed_onsets_no_llm_retiming',
    'delivery_cues':'inline_english_in_same_performance_request','punctuation':'scribe_native_unmodified','translation':'gemini_text_only'}
PROMPT=TEXT_PROMPT.replace('Fish ASR and local acoustic correction','Scribe ASR, local acoustic correction and audio-grounded emotion/speaker review')+''' Preserve the source's ASR punctuation and audible communicative intent in translation: statements, exclamations, questions, surprised questions (?!), repetitions (!!), ellipses (...), tildes (~ / 〜 / ～), long dashes, quotation marks and unfinished speech. Use equivalent target-language punctuation. Do not flatten them all into periods. Preserve every supplied English [] acting/delivery cue verbatim at its corresponding semantic position in the translation, including sentence-internal cues. Do not translate, omit, paraphrase, reorder or invent cues. Emotion-change boundaries have already been reviewed against the source audio: translate each supplied part independently in context, without merging parts or changing speaker IDs.'''


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
    grouped=[{**row,'id':i} for i,row in enumerate(regions({**raw,'words':words},sf.info(clean).duration,clean=clean))]
    from source_word_boundaries import repair_japanese_boundaries
    grouped,word_boundary_repairs=repair_japanese_boundaries(grouped,raw.get('text',''))
    from mai_native_pipeline import record_auto_language
    record_auto_language(raw,cfg)
    return {**raw,'model':MODEL,'segments':grouped,
        'source_word_boundary_repairs':word_boundary_repairs,
        'signature':{'policy':POLICY,'asr':raw['signature']},
        'native_speaker_mapping':[{'native_id':json.loads(k),'speaker_id':v} for k,v in mapping.items()],
        '_api_requests':int(before!=raw['signature'])}


def transcribe(clean,work,output,cfg,progress):
    from moss_pipeline import transcribe as translate
    holder={}
    def activity(p,detail):
        for name in ('MAI ','Qwen ','Gemini '):detail=detail.replace(name,'')
        progress(p,detail)
    def load(clean,cfg,progress):
        holder['source']=source(clean,Path(work)/'scribe-source-v1',cfg,progress)
        return holder['source']
    def boundaries(clean,rows,progress):
        corrected,audit=correct_fish_edges(clean,rows,progress)
        from collapsed_source_onsets import repair
        repaired,checks=repair(clean,corrected,holder['source']['words'],Path(work)/'collapsed-source-onsets',cfg,
            language=cfg.get('source_language'))
        holder['collapsed_onset_checks']=checks
        from corrected_source_groups import merge_corrected
        merged,merges=merge_corrected(repaired,POLICY['grouping_gap_seconds'])
        holder['corrected_gap_merges']=merges
        reviewed,receipt=review_performance(clean,merged,work,cfg,
            lambda p,d:progress(50+p*.15,d))
        holder['performance_review']=receipt
        return reviewed,audit+checks+merges+receipt.get('changes',[])
    result=translate(clean,work,output,cfg,activity,source_loader=load,
        boundary_builder=boundaries,
        authority=AUTHORITY,translation_prompt=PROMPT,region_builder=lambda raw,duration:raw['segments'])
    result.update(preserve_source_speaker_ids=True,source_transcription_model=MODEL,
        transcription_mode='scribe_native_words_local_edges_qwen_emotion_turns_speakers_gemini_translation',
        native_speaker_mapping=holder['source']['native_speaker_mapping'],
        source_word_boundary_repairs=holder['source']['source_word_boundary_repairs'],
        reference_tail_policy={'enabled':False,'method':'scribe_local_acoustic_edges'})
    result['models']['punctuation']=QWEN
    result.pop('punctuation_receipt',None)
    result['models'].update(source_transcription=MODEL,translation=GEMINI,speakers=QWEN,performance_review=QWEN)
    result['speaker_authority']=QWEN
    result['combined_transcription_and_speakers']=False
    result['performance_review_receipt']=holder['performance_review']
    result['models']['delivery_cues']=QWEN
    result['request_counts']['delivery_cues']=0
    result['delivery_cue_receipt']={'requests':0,'shared_with':'performance_review','annotated_regions':sum('[' in r['original_text'] for r in result['segments'])}
    result['speaker_names']={**holder['performance_review'].get('speaker_names',{}),**holder['source'].get('speaker_names',{})}
    result['request_counts']['performance_review']=holder['performance_review'].get('requests',0)
    result['request_counts']['speakers_api']=holder['performance_review'].get('requests',0)
    result['corrected_source_gap_merges']=holder.get('corrected_gap_merges',[])
    result['collapsed_source_onset_checks']=holder.get('collapsed_onset_checks',[])
    atomic_json(output,result)
    return result
