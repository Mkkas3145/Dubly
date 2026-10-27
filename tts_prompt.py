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


DUBBING_INSTRUCTIONS='Match the voice, acting, and all other qualities of the reference audio (speaker 0) when dubbing. Speaker 1 serves as a reference for voice identity.'


def instructions(target,source=None):
    return DUBBING_INSTRUCTIONS


def structured_instructions(target,source=None):
    return DUBBING_INSTRUCTIONS
