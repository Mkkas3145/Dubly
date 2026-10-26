import io,json
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR
from tts_prompt import instructions,language_code,with_transcript_languages
import render


def test_english_prompt_uses_source_and_target_pronunciation_names():
    assert instructions('ko','jpn')==('Dub in native Korean. Do not carry over Japanese pronunciation or intonation. '
        'Use natural Korean pronunciation and intonation. Natural adaptation into Korean is allowed.')
    assert 'Do not carry over English' in instructions('ja','eng')
    assert 'Use natural Japanese' in instructions('ja','eng')
    assert 'Do not carry over Korean' in instructions('fr','ko-KR')
    assert 'Use natural French' in instructions('fr','ko-KR')
    assert "the source language's pronunciation" in instructions('en',None)
    assert 'Japanese' not in instructions('en',None)
    assert 'Do not carry over' not in instructions('en','eng')
    assert language_code('jpn')=='ja' and language_code('ja-JP')=='ja'


def test_auto_language_uses_transcript_without_guessing_japanese():
    cfg={'language':'ko','source_language':None,'source_language_auto':True}
    assert with_transcript_languages(cfg,{'source_languages':['jpn'],'target_language':'en'})['source_language']=='ja'
    assert with_transcript_languages(cfg,{'source_languages':['fra'],'target_language':'ko'})['source_language']=='fr'
    assert with_transcript_languages(cfg,{'source_languages':['jpn','eng'],'target_language':'ko'})['source_language'] is None
    assert with_transcript_languages(cfg,{'source_languages':['unknown'],'target_language':'ko'})['source_language'] is None
    assert with_transcript_languages({**cfg,'source_language':'en'},{'source_languages':['jpn']})['source_language']=='en'
    assert cfg['source_language'] is None


def test_actual_requests_and_caches_use_resolved_pair(tmp_path):
    clean=tmp_path/'clean.wav';sf.write(clean,np.full((SR,1),.1),SR)
    row={'start':0.,'end':1.,'lo':0,'hi':SR,'original_text':'source','translated_text':'출력','speaker_id':'0'}
    sent=[]
    class Reply(io.BytesIO):headers={}
    def reply(req,**kwargs):sent.append(json.loads(req.data));return Reply(b'audio')
    base={'language':'ko','key':'test','source_language':None,'tts_provider_options':{'temperature':1,'top_p':1}}
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'1'}}):
        a=render.generate_speech(0,row,clean,tmp_path,with_transcript_languages(base,{'source_languages':['jpn']}))
        b=render.generate_speech(0,row,clean,tmp_path,with_transcript_languages(base,{'source_languages':['eng'],'target_language':'en'}))
    assert sent[0]['instructions']==instructions('ko','ja')
    assert sent[1]['instructions']==instructions('en','en')
    assert a['policy']!=b['policy']
    assert a['generation']['tts_policy']==a['policy'] and b['generation']['tts_policy']==b['policy']
    assert sent[0]['input']=='출력' and '<|speaker:' not in sent[0]['input']


def test_structured_prompt_preserves_reference_acting_and_uses_selected_language_pair():
    from core import TARGET_LANGUAGES
    from tts_prompt import structured_instructions
    for target in TARGET_LANGUAGES:
        prompt=structured_instructions(target,'ja')
        assert '모든 조건들을 충족하며' in prompt and '목소리와 연기' in prompt
        assert '각색은 허용합니다.' in prompt
    assert structured_instructions('ko','ja')==('참조의 목소리와 연기를 모든 조건들을 충족하며 자연스러운 한국어로 말하세요. '
        '일본어 발음·억양은 배제하고, 자연스러운 한국어 각색은 허용합니다.')
    assert '영어 발음·억양은 배제하고' in structured_instructions('ja','en')
    assert '자연스러운 일본어 각색은 허용합니다.' in structured_instructions('ja','en')
    assert '원본 언어의 발음·억양은 배제하고' in structured_instructions('ko',None)
    assert '일본어' not in structured_instructions('ko',None)
    assert '배제' not in structured_instructions('ko','kor')
