import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import render

def run(tmp_path,*,fail_encoder=False,invalid_mf=False,all_fail=False):
    source=tmp_path/'source.mp4';mixed=tmp_path/'mixed.wav';destination=tmp_path/'final.mp4'
    destination.write_bytes(b'previous')
    calls=[];probes=[]
    def encode(args):
        calls.append(args);encoder=args[args.index('-c:a')+1]
        if all_fail or (fail_encoder and encoder=='aac_mf'):raise RuntimeError('encoder unavailable')
        # The old published MP4 stays untouched until validation succeeds.
        assert destination.read_bytes()==b'previous'
        (tmp_path/'final.partial.mp4').write_bytes(encoder.encode())
    def probe(path):
        probes.append(path)
        return {'encoder':calls[-1][calls[-1].index('-c:a')+1]} if calls else {}
    def validate(info,*args):
        passed=not (invalid_mf and info['encoder']=='aac_mf')
        return {'passed':passed,'errors':[] if passed else ['invalid output duration']}
    with patch.object(render,'MUX_AUDIO_ENCODERS',('aac_mf','aac')), \
         patch.object(render.sf,'info',return_value=SimpleNamespace(duration=2.)), \
         patch.object(render,'command',side_effect=encode),patch.object(render,'probe',side_effect=probe), \
         patch.object(render,'validate_mux',side_effect=validate):
        if all_fail:
            with pytest.raises(RuntimeError,match='MP4 저장 실패'):render.mux(source,mixed,destination,'ko',1.)
        else:assert render.mux(source,mixed,destination,'ko',1.)==destination
    return calls,destination,json.loads((tmp_path/'mux-validation.json').read_text('utf8'))

def test_media_foundation_is_first_choice_with_video_copy_and_320k(tmp_path):
    calls,destination,report=run(tmp_path)
    assert len(calls)==1 and destination.read_bytes()==b'aac_mf'
    assert report['audio_encoder']=='aac_mf' and not report['fallback_used']
    args=calls[0]
    assert args[args.index('-c:v')+1]=='copy'
    assert args[args.index('-b:a')+1]=='320k'
    assert args[args.index('-movflags')+1]=='+faststart'
    assert args[args.index('-t')+1]=='2.0'

@pytest.mark.parametrize('failure',['unsupported','invalid_output'])
def test_failed_mf_falls_back_to_native_aac_before_publishing(tmp_path,failure):
    calls,destination,report=run(tmp_path,fail_encoder=failure=='unsupported',invalid_mf=failure=='invalid_output')
    assert [args[args.index('-c:a')+1] for args in calls]==['aac_mf','aac']
    assert destination.read_bytes()==b'aac'
    assert report['passed'] and report['fallback_used'] and report['audio_encoder']=='aac'
    assert not report['attempts'][0]['passed'] and report['attempts'][1]['passed']

def test_failed_both_encoders_preserves_existing_video(tmp_path):
    calls,destination,report=run(tmp_path,all_fail=True)
    assert len(calls)==2 and destination.read_bytes()==b'previous'
    assert not report['passed'] and len(report['attempts'])==2
