"""Brief dubbing directions using the selected source and output languages."""
from core import LANGUAGE_CATALOG


def language_code(value):
    if not isinstance(value,str):return None
    value=value.strip().replace('_','-').lower()
    if value in ('','auto','unknown','und'):return None
    aliases={'ger':'de','fre':'fr','chi':'zh','cmn':'zh','jap':'ja','dut':'nl','rum':'ro','cze':'cs','gre':'el','per':'fa'}
    value=aliases.get(value,value)
    for code,row in LANGUAGE_CATALOG['languages'].items():
        if value in (code.lower(),str(row.get('iso639_3','')).lower()):return code
    if '-' in value:return language_code(value.split('-')[0])
    return None


def with_transcript_languages(cfg,transcript):
    result=dict(cfg)
    source=language_code(cfg.get('source_language'))
    if source is None:
        detected={language_code(v) for v in transcript.get('source_languages',[]) or []}
        detected.discard(None)
        if len(detected)==1:source=next(iter(detected))
    result['source_language']=source
    result['language']=transcript.get('target_language') or cfg['language']
    return result


def korean_language(value,fallback):
    code=language_code(value)
    return LANGUAGE_CATALOG['languages'][code]['labels']['ko'] if code else fallback


def instructions(target,source=None):
    code=language_code(target);source_code=language_code(source)
    name=LANGUAGE_CATALOG['languages'][code]['labels']['en'] if code else 'the target language'
    source_name=LANGUAGE_CATALOG['languages'][source_code]['labels']['en'] if source_code else None
    source_style=f'{source_name} pronunciation or intonation' if source_name else "the source language's pronunciation or intonation"
    exclude=f' Do not carry over {source_style}.' if source_code!=code or source_code is None else ''
    return f'Preserve the reference voice and acting. Sound like a native {name} speaker, with natural pronunciation, intonation, stress and rhythm.'+exclude+f' Natural adaptation into {name} is allowed.'


def structured_instructions(target,source=None):
    name=korean_language(target,'목표 언어')
    if len(name)>8:name=name.split('(')[0].strip()
    source_code=language_code(source);target_code=language_code(target)
    source_name=korean_language(source,'원본 언어의')
    exclude=f'{source_name} 발음·억양은 배제하고, ' if source_code is None or source_code!=target_code else ''
    return (f'참조의 목소리와 연기를 유지하며, {name} 원어민이 말하는 것처럼 자연스러운 발음·억양·강세·리듬으로 더빙하세요. '
        +exclude+f'자연스러운 {name} 각색은 허용합니다.')
