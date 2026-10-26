from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
SR = 44100
GEMINI = 'google/gemini-3.8-flash'
QWEN = 'qwen/qwen3.8-omni-flash'
FISH = 'fish-audio/drama-3'
TTS_CONCURRENCY = 20
FIT_CONCURRENCY = 40
DEFAULT_SEPARATOR = 'mel-band-roformer-deux'
DEFAULT_BACKGROUND = 'mel-band-roformer-deux'
DEFAULT_DIARIZER = 'scribe-v2'
MODEL_HASH = '87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e'
LANGUAGE_CATALOG=json.loads((ROOT/'assets/languages.json').read_text(encoding='utf8'))
SOURCE_LANGUAGES=frozenset(LANGUAGE_CATALOG['source'])
SOURCE_SELECTIONS=SOURCE_LANGUAGES|{'auto'}
TARGET_LANGUAGES=frozenset(LANGUAGE_CATALOG['target'])
MAI_LANGUAGES=frozenset(LANGUAGE_CATALOG['model_languages']['mai'])
LANGUAGES={code:row['name'] for code,row in LANGUAGE_CATALOG['languages'].items()}


def config():
    values = dotenv_values(ROOT / '.env')
    return {'key':(os.getenv('OPENROUTER_API_KEY') or values.get('OPENROUTER_API_KEY') or '').strip(),
            'moss_api_key':(os.getenv('MOSS_API_KEY') or values.get('MOSS_API_KEY') or '').strip(),
            'moss_keyterms':os.getenv('MOSS_KEYTERMS') or values.get('MOSS_KEYTERMS') or '[]',
            'language':(values.get('TARGET_LANGUAGE') or 'ko').strip(),
            'separator':(values.get('SEPARATION_MODEL') or DEFAULT_SEPARATOR).strip(),
            'background':(values.get('BACKGROUND_MODEL') or DEFAULT_BACKGROUND).strip(),
            'diarizer':DEFAULT_DIARIZER,'tts_mode':'individual','tts_reference_mode':'same_speaker_tagged_context',
            'speaker_embeddings':False,'tts_output_guard':None,'tts_omit_instructions':False,'tts_delivery_cue':None,
            'tts_forced_alignment':False,'tts_alignment_review':True,'keep_original_on_tts_failure':True}


def video_languages(cfg,job):
    """Persisted upload choices override global defaults, without changing .env."""
    from tts_presets import configure
    result=configure(cfg,job)
    source,target=job.get('source_language'),job.get('target_language')
    if source is None and target is None:return result
    if source not in SOURCE_SELECTIONS or target not in TARGET_LANGUAGES or source==target:
        raise ValueError('Invalid source or target language configuration.')
    result.update(source_language=None if source=='auto' else source,source_language_auto=source=='auto',language=target)
    return result


def atomic_json(path, value):
    import tempfile,time
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Each writer owns its temporary file, including across API/worker processes.
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            stream.write(json.dumps(value,ensure_ascii=False,indent=2))
        for attempt in range(8):
            try:os.replace(temporary,path);break
            except PermissionError:
                # Windows readers can briefly hold a destination without delete sharing.
                if attempt==7:raise
                time.sleep(min(.01*2**attempt,.2))
    finally:
        temporary.unlink(missing_ok=True)


def command(args,cwd=None):
    from local_resources import local_resources
    with local_resources():
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError(result.stderr[-2500:] or '외부 도구 실행 실패')
        return result.stdout


def probe(path):
    return json.loads(command(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
