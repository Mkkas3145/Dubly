"""Text units for MMS-FA. Originals remain immutable; romanization is alignment-only."""
import re,unicodedata
from functools import lru_cache

@lru_cache(maxsize=1)
def romanizer():
    from uroman import Uroman
    return Uroman()

@lru_cache(maxsize=1)
def japanese():
    from sudachipy import dictionary
    return dictionary.Dictionary().create()

@lru_cache(maxsize=1)
def iso_languages():
    import pycountry
    from core import SOURCE_LANGUAGES
    result={}
    for code in SOURCE_LANGUAGES:
        value=pycountry.languages.get(alpha_2=code) or pycountry.languages.get(alpha_3=code)
        if value:result[code]=value.alpha_3
    result.update(zh='zho',yue='yue',ja='jpn',ko='kor')
    return result


def text_units(text,language):
    from rich_transcript import spoken
    text=spoken(text)
    language=language.lower().split('-')[0]
    code=iso_languages().get(language,language)
    if code=='jpn':
        from sudachipy import tokenizer
        # Kanji needs Japanese readings, not a generic Chinese transliteration.
        raw=[(m.surface(),m.reading_form()) for m in japanese().tokenize(text,tokenizer.Tokenizer.SplitMode.A)]
    else:
        raw=[(m.group(),m.group()) for m in re.finditer(r"[^\W_]+(?:['’][^\W_]+)*",text,re.UNICODE)]
    result=[]
    for original,reading in raw:
        if not any(c.isalnum() for c in original):continue
        if reading.isnumeric():
            from num2words import num2words
            try:reading=num2words(int(reading),lang=language)
            except (NotImplementedError,ValueError,OverflowError):
                raise ValueError(f'Cannot normalize numeric source token for MMS-FA ({language}): {original}')
        latin=romanizer().romanize_string(reading,lcode=code).lower()
        roman=''.join(c for c in unicodedata.normalize('NFKD',latin) if c in "abcdefghijklmnopqrstuvwxyz'")
        if not roman and code=='jpn' and set(unicodedata.normalize('NFKC',reading))<={'ー'}:
            # Sudachi may emit a prolonged-sound mark as its own token.
            # Keep that original token for transcript ownership, and extend the
            # preceding vowel. Without a local vowel, use MMS's native wildcard
            # instead of inventing a syllable or deleting the source marker.
            previous=result[-1]['roman'] if result else ''
            vowel=next((c for c in reversed(previous) if c in 'aeiou'),None)
            roman=vowel*len(reading) if vowel else '*'
        if not roman:raise ValueError(f'MMS-FA cannot romanize source token: {original}')
        result.append({'word':original,'roman':roman})
    return result
