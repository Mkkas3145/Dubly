import json
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from core import config, SR
import qwen_audio_pipeline as q


def turn(start=0.2,end=0.9,speaker=0,text='[surprised]本当？',translation='[surprised]정말?'):
    return dict(start=start,end=end,speaker=speaker,original_text=text,translated_text=translation)


def test_schema_and_parser_preserve_cues_and_nullable_translation():
    entries=[turn(),turn(1.,1.4,translation=None)]
    for response in [entries,{'segments':entries}]:
        rows=q.parse(json.dumps(response),2.)
        assert rows[0]['text']=='[surprised]本当？'
        assert rows[0]['translation']=='[surprised]정말?'
        assert rows[1]['translation'] is None
    schema=q.response_format()['json_schema']['schema']
    assert set(schema['properties']['segments']['items']['required'])==set(turn())


@pytest.mark.parametrize('change',[{'start':-1},{'end':3.},{'speaker':True},{'end':float('nan')},
    {'original_text':''},{'translated_text':''}])
def test_invalid_model_output_is_rejected(change):
    with pytest.raises(ValueError):q.parse(json.dumps({'segments':[{**turn(),**change}]}),2.)


def test_native_audio_dispatch_one_request_cues_and_cache(tmp_path):
    from speech import transcribe
    from reference_timing import refine_references
    audio=tmp_path/'dialogue.wav'
    signal=np.zeros(3*SR,np.float32)
    signal[int(.3*SR):int(.8*SR)]=.15*np.sin(np.arange(int(.8*SR)-int(.3*SR))*.04)
    sf.write(audio,signal,SR)
    compressed=tmp_path/'input.mp3';compressed.write_bytes(b'fake audio')
    entries=[turn(),turn(1.2,1.5,translation=None),turn(2.,2.4,translation='[laughing]')]
    def native_request(payload,key,folder,duration,progress):
        assert payload['model']==q.MODEL
        assert payload['reasoning']=={'enabled':False}
        assert payload['max_tokens']==131072
        content=payload['messages'][0]['content']
        assert len(content)==2 and content[1]['input_audio']=={'format':'mp3','data':'ZmFrZSBhdWRpbw=='}
        assert 'square-bracketed [] cues' in content[0]['text']
        assert 'Cues MUST appear in original_text itself' in content[0]['text']
        assert 'copy every cue unchanged' in content[0]['text']
        assert 'Identify speakers solely by their audible voices' in content[0]['text']
        assert 'Do not use dialogue meaning, conversational context' in content[0]['text']
        assert 'return speaker:null rather than guessing' in content[0]['text']
        return q.parse(json.dumps(entries),duration),{'usage':{},'elapsed_seconds':1.}
    cfg={**config(),'key':'test','diarizer':'qwen-omni-full'}
    with patch.object(q,'encode_audio',return_value=(compressed,{'format':'mp3'})), \
         patch.object(q,'request',side_effect=native_request) as request, \
         patch('speech.http_json',side_effect=AssertionError('Unexpected extra LLM/ASR call')):
        result=transcribe(audio,tmp_path/'work',tmp_path/'segments.json',cfg,lambda *a:None)
        cached=transcribe(audio,tmp_path/'work',tmp_path/'segments.json',cfg,lambda *a:None)
        measured,_=refine_references(audio,result,tmp_path/'references')
    assert request.call_count==1
    assert result==cached==measured
    assert len(result['segments'])==1 and len(result['excluded_intervals'])==2
    assert result['segments'][0]['original_text']=='[surprised]本当？'
    assert result['segments'][0]['translated_text']=='[surprised]정말?'
    window=result['reference_windows'][0]
    assert window['start']==result['segments'][0]['start']
    assert window['end']==result['segments'][0]['end']
    assert result['additional_asr_requests']==0

def test_lossless_format_conversion_preserves_source_rate_and_channels(tmp_path):
 import scribe
 source=tmp_path/'stereo.wav';folder=tmp_path/'input';folder.mkdir()
 sf.write(source,np.random.default_rng(4).normal(0,.02,(48000,2)),48000,subtype='PCM_24')
 audio,fmt,duration=scribe.prepare_input(source,folder,'native-source')
 info=sf.info(audio)
 assert fmt=='flac' and info.samplerate==48000 and info.channels==2 and duration==1
 original,_=sf.read(source);converted,_=sf.read(audio)
 assert np.array_equal(original,converted)


def test_qwen_tries_high_quality_format_before_lowering_sample_rate(tmp_path):
 source=tmp_path/'stereo.wav';folder=tmp_path/'input';folder.mkdir()
 sf.write(source,np.random.default_rng(7).normal(0,.02,(48000*2,2)),48000)
 audio,manifest=q.encode_audio(source,folder,'native-source',lambda *a:None,base64_limit=150000)
 assert manifest['sample_rate']==48000 and manifest['channels']==2
 assert manifest['bitrate_kbps']==320
 assert manifest['base64_bytes']<=150000 and abs(sf.info(audio).duration-2)<.15
