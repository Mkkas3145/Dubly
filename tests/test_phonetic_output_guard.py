from types import SimpleNamespace
from unittest.mock import patch
import math
import numpy as np
import soundfile as sf
from core import SR
from phonetic_output_guard import eligible,inspect,POLICY
from phonetic_replay import _candidate_scores
from waveform_output_guard import _path


def bank():
    return [{'own':True,'original_text':'スパイ','translated_text':'스파이','path':'own.wav'},
            {'own':False,'original_text':'ミッション','translated_text':'스파이','path':'donor.wav'}]


def test_incomplete_cognate_inside_longer_requested_text_never_loads_model(tmp_path):
    refs=bank();refs[0]['translated_text']='스파이, 미션, 두근두근'
    with patch('phonetic_output_guard._get_runtime',side_effect=AssertionError('No target anchor')):
        result=inspect('missing.wav',refs,tmp_path/'out.wav',{'action':'keep'})
    assert result['action']=='keep' and result['phonetic_check']['action']=='skip'
    assert not eligible(refs)


def test_positive_proposal_crops_only_supplied_samples_and_preserves_raw_audio(tmp_path):
    rng=np.random.default_rng(8);values=rng.normal(0,.1,(SR*2,2)).astype('float32')
    source=tmp_path/'source.wav';sf.write(source,values,SR,subtype='FLOAT')
    raw=source.read_bytes();fake=SimpleNamespace(encode=lambda *args:{})
    proposal={'action':'crop_candidate','keep_start':.25,'keep_end':1.5}
    with patch('phonetic_output_guard._get_runtime',return_value=(fake,None)), \
         patch('phonetic_output_guard.propose_crop',return_value=proposal):
        result=inspect(source,bank(),tmp_path/'out.wav',{'action':'keep','content_verified':False})
    assert result['action']=='cropped' and result['phonetic_check']['policy']==POLICY
    kept,rate=sf.read(tmp_path/'out.wav',always_2d=True)
    assert np.max(np.abs(kept-values[round(.25*SR):round(1.5*SR)]))<2**-22
    assert source.read_bytes()==raw and rate==SR


def test_model_failure_preserves_previous_decision_without_aborting_video(tmp_path):
    previous={'action':'keep','reason':'no_verified_edge_replay','source_frames':100}
    with patch('phonetic_output_guard._get_runtime',side_effect=RuntimeError('model unavailable')):
        result=inspect('missing.wav',bank(),tmp_path/'out.wav',previous)
    assert all(result[k]==v for k,v in previous.items())
    assert result['phonetic_check']['action']=='unavailable'
    assert not (tmp_path/'out.wav').exists()


def test_valid_waveform_crop_is_not_cropped_twice(tmp_path):
    previous={'action':'cropped','path':'already.wav'}
    with patch('phonetic_output_guard._get_runtime',side_effect=AssertionError('Already verified')):
        assert inspect('missing.wav',bank(),tmp_path/'out.wav',previous)==previous


def test_optimized_dtw_retains_full_path_scores_for_every_candidate():
    similarity=np.random.default_rng(9).uniform(0,1,(7,19))
    actual=_candidate_scores(similarity)
    m,n=similarity.shape
    expected=[]
    for a in range(n):
        for b in range(a+max(3,math.ceil(m*.65)),min(n,a+math.floor(m*1.55))+1):
            rows,cols=_path(1-similarity[:,a:b])
            score=similarity[:,a:b][rows,cols].mean()-.12*abs(math.log((b-a)/m))
            expected.append([a,b,score])
    assert np.allclose(actual,np.array(expected),atol=1e-10)
