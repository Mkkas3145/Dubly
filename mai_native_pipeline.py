"""Single MAI ASR, bounded waveform correction, fixed speakers, ID-only translation."""
from pathlib import Path
import soundfile as sf
from mai_pipeline import MODEL,AUTHORITY,OPTIONS,transcribe_source,word_regions,correct_edges
from moss_pipeline import transcribe

POLICY={'version':5,'model':MODEL,'input':'whole_separated_dialogue_original_clock',
        'requests':1,'same_speaker_gap_seconds':.8,'speaker_change':'split_immediately',
        'speaker_correction':'redimnet2_plus_before_pause_merging','boundary_correction':'neighbor_bounded_waveform_quiet_edges',
        'translation_output':'region_id_to_translation_or_null_only','gemini_audio':False,
        'delivery_cues':'qwen_source_cues_before_gemini_translation_no_lexical_changes'}

TEXT_PROMPT='''Translate each numbered source dialogue faithfully into natural spoken {language}, preserving meaning, expressive punctuation, meaningful hesitations and unfinished speech. Use surrounding text as context. Speaker IDs and region assignments are fixed by MAI; do not change speakers, move words between IDs or merge regions. No audio is provided. Preserve every supplied English [] performance cue verbatim at the corresponding position in the translation. Do not translate, omit, invent, intensify or add cues.
Return only a JSON object mapping supplied IDs to translated strings, or null for nonverbal/background-only sounds and reactions needing no dubbing. Preserve meaningful short dialogue. Do not output original text, speakers, times or explanations.'''

AUDIO_PROMPT='''Translate each numbered source dialogue faithfully into natural spoken {language}. The full separated dialogue audio accompanies the transcript. Use the audio to retain actual emotion, vocal delivery, hesitation, breathing and expressive punctuation. Add only sparse, short [] performance cues clearly supported by audible delivery and useful for TTS. Write all [] performance cues in concise English for TTS, translating any existing non-English cues faithfully without changing their meaning or intensity. Do not add sound-effect/background labels, identity descriptions, imagined acting, or exaggerated intensity.
Speaker IDs, region boundaries and dialogue assignments are fixed by MAI. Input start/end values only locate each region in the audio; never estimate, change or return times, move words between regions, merge regions, or change speakers. Do not output a new source transcription.
Return only a JSON object mapping region IDs to translated strings with cues, or null for nonverbal/background-only sounds or reactions needing no dubbing. Preserve meaningful short dialogue. No original text, speaker IDs, timestamps or explanations.'''


def detected_languages(raw):
    from core import LANGUAGE_CATALOG
    aliases={'tl':'fil','tagalog':'fil','no':'nb','norwegian':'nb','cmn':'zh','mandarin':'zh','chinese':'zh'}
    for code,row in LANGUAGE_CATALOG['languages'].items():
        for name in [row['name'],row.get('nativeName',''),*row['labels'].values()]:aliases[name.lower()]=code
    def code(value):
        if not isinstance(value,str):return None
        normalized=value.strip().lower().replace('_','-');normalized=aliases.get(normalized,normalized.split('-')[0])
        return normalized if normalized in LANGUAGE_CATALOG['languages'] else None
    main=code(raw.get('language_code') or raw.get('language') or raw.get('detected_language'))
    found={v for r in [*raw.get('segments',[]),*raw.get('words',[])] if isinstance(r,dict) and (v:=code(r.get('language')))}
    if main:found.add(main)
    return main or (next(iter(found)) if len(found)==1 else None),found


def record_auto_language(raw,cfg):
    if not cfg.get('source_language_auto'):return
    import os
    from core import SOURCE_LANGUAGES
    detected,found=detected_languages(raw)
    video_id=os.environ.get('DUBLY_VIDEO_ID')
    if video_id:
        from storage import Store
        Store().update(video_id,detected_source_language=detected)
    if detected and detected not in SOURCE_LANGUAGES:raise RuntimeError('The detected language is not supported by this pipeline. Check the source language.')
    if detected==cfg['language'] and len(found)==1:raise RuntimeError('The detected source language matches the target language. Choose a different target language.')


def load_source(clean,cfg,progress):
    folder=Path(clean).parent/'source-asr-mai-v1'
    # transcribe_source validates its own model/audio/options cache before reuse.
    raw,identity=transcribe_source(clean,folder,cfg['key'],progress,language=cfg.get('source_language'))
    record_auto_language(raw,cfg)
    from native_speaker_repair import prepare as repair_speakers
    repaired,breaks,repair=repair_speakers(clean,raw,Path(clean).parent/'source-speaker-repair-v1',
        lambda p,d:progress(30+p*.65,d))
    rows=word_regions(repaired,sf.info(clean).duration,merge_breaks=breaks)
    return {**repaired,'segments':rows,'model':MODEL,
        'signature':{'asr':identity,'policy':POLICY,'speaker_repair':{'policy':repair['policy'],'signature':repair.get('signature')}},
        '_speaker_repair':repair,'_api_requests':int(not raw.get('_cache_reused',False))}


def translate(clean,work,output,cfg,progress):
    from qwen_delivery_cues import annotate_source
    return transcribe(clean,work,output,cfg,progress,source_loader=load_source,
        boundary_builder=correct_edges,authority=AUTHORITY,translation_prompt=TEXT_PROMPT,source_cue_builder=annotate_source)
