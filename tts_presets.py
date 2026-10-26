"""Dubly starting points, not benchmarked pronunciation/voice guarantees."""
import json
import math

DEFAULT='creative'
PRESETS={
    'creative':{'label':'창의적','temperature':1.0,'top_p':1.0,
        'description':'억양과 리듬의 변화폭을 넓힙니다. 같은 대사도 다른 연기로 나올 수 있습니다.'},
    'balanced':{'label':'균형','temperature':1.0,'top_p':.9,
        'description':'연기를 살리면서 변화폭을 조금 줄입니다.'},
    'consistent':{'label':'일관적','temperature':.7,'top_p':.7,
        'description':'표현의 변화를 줄입니다. 낮은 값에서는 쉼이 길어지거나 목소리가 평평해졌다는 사용자 보고가 있습니다.'},
}

def options(code):
    if code not in PRESETS:raise ValueError('Invalid TTS preset')
    return {key:PRESETS[code][key] for key in ('temperature','top_p')}

def configure(cfg,job):
    # Existing videos used 1.0/1.0; migration preserves their generation policy.
    code=job.get('tts_preset') or 'creative'
    sampling=options(code)
    override=job.get('tts_sampling_options')
    if override is not None:
        override=json.loads(override) if isinstance(override,str) else override
        if not isinstance(override,dict) or set(override)!={'temperature','top_p'}:
            raise ValueError('Invalid per-video TTS sampling options')
        if any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1 for value in override.values()):
            raise ValueError('TTS sampling values must be between zero and one')
        sampling=dict(override)
    return {**cfg,'tts_preset':code,'tts_provider_options':sampling}
