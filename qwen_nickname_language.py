"""Display-name language rules; never change voice identities or dialogue."""
from tts_prompt import language_code
from core import LANGUAGES


def instruction(language):
    code=language_code(language) or language
    name=LANGUAGES.get(code,'the selected output language')
    return (f'All speaker nicknames MUST be written in {name}, the dubbing OUTPUT language, '
        'not the recording or transcript language. Translate descriptive labels and use '
        'established localized names or transliterate personal names into the normal writing '
        'system of the output language. Do not copy a source-language name unchanged when '
        'its writing system differs, or append its source spelling in parentheses. '
        'Preserve the identified person and speaker ID; this is display-label localization only.')
