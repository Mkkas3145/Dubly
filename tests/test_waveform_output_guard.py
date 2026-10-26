from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR
from waveform_output_guard import inspect_output,check_batch,features,compare

def utterance(seed,stretch=1.):
    rng=np.random.default_rng(seed);duration=1.2*stretch;t=np.arange(round(SR*duration))/SR
    u=t/stretch;frequencies=rng.uniform(250,3000,7)
    f=np.interp(u,np.linspace(0,1.2,7),frequencies)
    phase=np.cumsum(f)*2*np.pi/SR
    envelope=.02+.12*np.exp(-((u-.22)/.12)**2)+.2*np.exp(-((u-.65)/.2)**2)+.08*np.exp(-((u-1.05)/.1)**2)
    return (envelope*(np.sin(phase)+.3*np.sin(phase*1.7))).astype('float32')

def fixture(tmp_path,prefix=True,suffix=False,stretch=1.):
    reference=utterance(1);target=utterance(77)
    sf.write(tmp_path/'ref.wav',reference,SR,subtype='FLOAT')
    copied=utterance(1,stretch)*.7
    gap=np.zeros(round(.3*SR),dtype='float32')
    pieces=([copied,gap] if prefix else [])+[target]+([gap,copied] if suffix else [])
    values=np.concatenate(pieces);sf.write(tmp_path/'tts.wav',values,SR,subtype='FLOAT')
    return values,[{'path':str(tmp_path/'ref.wav'),'index':9}]

def test_gain_changed_reference_prefix_removed_without_touching_remaining_samples(tmp_path):
    values,bank=fixture(tmp_path)
    report=inspect_output(tmp_path/'tts.wav',bank,tmp_path/'guarded.wav')
    assert report['action']=='cropped' and 1.2<report['keep_start']<1.5
    kept,rate=sf.read(tmp_path/'guarded.wav');lo=round(report['keep_start']*rate)
    assert np.max(np.abs(kept-values[lo:]))<2**-22
    assert report['asr_requests']==0 and not report['content_verified']

def test_suffix_and_both_edges_are_removed_only_at_real_quiet_boundaries(tmp_path):
    for prefix in (False,True):
        values,bank=fixture(tmp_path,prefix=prefix,suffix=True)
        report=inspect_output(tmp_path/'tts.wav',bank,tmp_path/'guarded.wav')
        assert report['action']=='cropped'
        kept,_=sf.read(tmp_path/'guarded.wav');lo=round(report['keep_start']*SR);hi=round(report['keep_end']*SR)
        assert np.max(np.abs(kept-values[lo:hi]))<2**-22
        assert 1.2<len(kept)/SR<1.8

def test_time_warp_is_measured_without_changing_output_voice(tmp_path):
    _,bank=fixture(tmp_path,stretch=1.15)
    report=inspect_output(tmp_path/'tts.wav',bank,tmp_path/'guarded.wav')
    assert report['action']=='cropped'

def test_single_or_only_reference_like_utterances_remain_intact(tmp_path):
    ref=utterance(1);sf.write(tmp_path/'ref.wav',ref,SR)
    for audio in (ref,np.concatenate([ref,np.zeros(round(.3*SR)),ref])):
        sf.write(tmp_path/'tts.wav',audio,SR)
        report=inspect_output(tmp_path/'tts.wav',[{'path':str(tmp_path/'ref.wav')}],tmp_path/'out.wav')
        assert report['action']=='keep' and not (tmp_path/'out.wav').exists()

def test_different_content_same_energy_envelope_is_not_a_replay(tmp_path):
    values,bank=fixture(tmp_path)
    # Both synthetic speakers use precisely the same energy rhythm.
    for seed in (17,24,50):
        sf.write(tmp_path/'tts.wav',np.concatenate([utterance(seed),np.zeros(round(.3*SR)),utterance(77)]),SR)
        report=inspect_output(tmp_path/'tts.wav',bank,tmp_path/'out.wav')
        assert report['action']=='keep'

def test_no_quiet_separation_or_protected_cognate_never_authorizes_cut(tmp_path):
    values,bank=fixture(tmp_path)
    sf.write(tmp_path/'tts.wav',np.concatenate([utterance(1),utterance(77)]),SR)
    assert inspect_output(tmp_path/'tts.wav',bank,tmp_path/'out.wav')['action']=='keep'
    fixture(tmp_path);bank[0]['protected']=True
    assert inspect_output(tmp_path/'tts.wav',bank,tmp_path/'out.wav')['action']=='keep'

def test_production_checker_uses_no_network_and_falls_back_to_generated_audio(tmp_path):
    _,bank=fixture(tmp_path)
    p={'index':0,'segment':{'original_text':'a','translated_text':'b'},'generated':tmp_path/'tts.wav',
        'reference':tmp_path/'ref.wav','folder':tmp_path,'generation':{}}
    with patch('waveform_output_guard.protected_text',return_value=False), \
         patch('urllib.request.urlopen',side_effect=AssertionError('No cloud inspection')):
        checked=check_batch([p],{},tmp_path/'checks')[0]
    assert checked['generation']['tts_output_check']['action']=='cropped'
    p['reference']=tmp_path/'missing.wav';p['generated']=tmp_path/'tts.wav'
    with patch('waveform_output_guard.protected_text',return_value=False):
        checked=check_batch([p],{},tmp_path/'checks')[0]
    assert checked['generated']==tmp_path/'tts.wav' and '_guard_error' not in checked
    assert checked['generation']['tts_output_check']['action']=='keep_uncertain'

def test_default_production_path_crops_then_fits_and_reuses_cache_without_asr(tmp_path):
    import render
    from core import config
    from waveform_output_guard import POLICY
    values,_=fixture(tmp_path)
    folder=tmp_path/'pieces';folder.mkdir();clean=tmp_path/'ref.wav'
    segments=[{'start':0.,'end':1.2,'lo':0,'hi':round(1.2*SR),
        'original_text':'こんにちは','translated_text':'안녕하세요'}]
    calls=[]
    def generate(reference,segment,language,key,destination):
        sf.write(destination,values,SR,format='WAV',subtype='FLOAT');calls.append('tts')
        return {'generated_seconds':len(values)/SR,'tts_policy':render.tts_policy(language)}
    def fit(source,destination,frames,reference,text,language,**kwargs):
        data,rate=sf.read(source,always_2d=True)
        assert 1.2<len(data)/rate<1.5
        sf.write(destination,np.repeat(data,2,axis=1),rate,subtype='FLOAT');calls.append('fit')
        return {'fit_version':render.FIT_VERSION,'timing_method':render.TIMING_POLICY,
            'output_frames':len(data),'speech_window_frames':[0,frames],'collision_budget_frames':None}
    cfg={**config(),'key':'test','tts_output_guard':'waveform'}
    with patch('render.request_audio',side_effect=generate),patch('render.bounded_timing_audio',side_effect=fit), \
         patch('urllib.request.urlopen',side_effect=AssertionError('No ASR/LLM inspection')):
        result=render.synthesize_all(segments,clean,folder,cfg,lambda *v:None)
    assert calls==['tts','fit']
    assert result[0]['tts_output_guard_policy']==POLICY
    assert result[0]['tts_output_check']['action']=='cropped'
    with patch('render.request_audio',side_effect=AssertionError('cached TTS')), \
         patch('waveform_output_guard.inspect_output',side_effect=AssertionError('cached check')), \
         patch('render.bounded_timing_audio',side_effect=AssertionError('cached fit')):
        assert render.synthesize_all(segments,clean,folder,cfg,lambda *v:None)==result

def test_requested_cross_language_cognates_inside_longer_line_are_protected():
    from waveform_output_guard import protected_text
    assert protected_text('スパイ','스파이, 미션, 두근두근!')
    assert protected_text('ミッション','스파이, 미션, 두근두근!')
    assert protected_text('スパイ、ミッション、わくわく！','스파이, 미션, 두근두근!')


def test_reference_replay_split_into_shorter_generated_groups_is_still_removed(tmp_path):
    values,bank=fixture(tmp_path)
    # Model-independent energy segmentation can split a replay differently.
    # Neither half meets the reference duration ratio; their full span does.
    def groups(audio,rate):
        if len(audio)==round(1.2*SR):return [(0,len(audio))]
        return [(0,round(.6*SR)),(round(.6*SR),round(1.2*SR)),
                (round(1.5*SR),len(audio))]
    with patch('waveform_output_guard.voice_groups',side_effect=groups):
        report=inspect_output(tmp_path/'tts.wav',bank,tmp_path/'guarded.wav')
    assert report['action']=='cropped'
    assert report['utterances'][0]['matched_span']==[0,1]
    kept,rate=sf.read(tmp_path/'guarded.wav')
    assert np.max(np.abs(kept-values[round(report['keep_start']*rate):]))<2**-22


def test_complete_reference_matches_when_generated_pause_segmentation_is_coarser(tmp_path):
    gap=np.zeros(round(.3*SR),dtype='float32')
    ref=np.concatenate([utterance(1),gap,utterance(2)])
    values=np.concatenate([ref,gap,utterance(77)])
    sf.write(tmp_path/'ref.wav',ref,SR,subtype='FLOAT')
    sf.write(tmp_path/'tts.wav',values,SR,subtype='FLOAT')
    def groups(audio,rate):
        if len(audio)==len(ref):return [(0,round(1.2*SR)),(round(1.5*SR),len(ref))]
        return [(0,len(ref)),(len(ref)+len(gap),len(audio))]
    with patch('waveform_output_guard.voice_groups',side_effect=groups):
        report=inspect_output(tmp_path/'tts.wav',[{'path':str(tmp_path/'ref.wav')}],tmp_path/'out.wav')
    assert report['action']=='cropped' and report['reference_parts']==3
    assert report['keep_start']>=len(ref)/SR


def test_regrouping_does_not_remove_a_whole_reference_like_output(tmp_path):
    ref=utterance(1);sf.write(tmp_path/'ref.wav',ref,SR,subtype='FLOAT')
    sf.write(tmp_path/'tts.wav',ref,SR,subtype='FLOAT')
    calls=[[(0,len(ref)//2),(len(ref)//2,len(ref))],[(0,len(ref))]]
    with patch('waveform_output_guard.voice_groups',side_effect=calls):
        report=inspect_output(tmp_path/'tts.wav',[{'path':str(tmp_path/'ref.wav')}],tmp_path/'out.wav')
    assert report['reason']=='no_distinct_remaining_utterance'
    assert not (tmp_path/'out.wav').exists()


def test_small_energy_boundary_shift_matches_but_lost_speech_energy_does_not():
    ref=utterance(1)
    shifted=ref[round(.04*SR):]*.8
    result=compare(features(shifted,SR),features(ref,SR))
    assert result['matched'] and result['aligned_energy_coverage']>=.9
    # Tiny common fragments cannot authorize removing a larger utterance.
    altered=ref.copy();altered[:round(.4*SR)]=utterance(77)[:round(.4*SR)]
    result=compare(features(altered,SR),features(ref,SR))
    assert not result['matched']
