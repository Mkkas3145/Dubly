from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import timing,render
from core import SR

def tone(seconds,amplitude=.15):
    t=np.arange(round(seconds*SR))/SR
    return np.repeat((amplitude*np.sin(2*np.pi*220*t))[:,None],2,axis=1).astype('float32')

def silence(seconds):return np.zeros((round(seconds*SR),2),dtype='float32')

def fake_stretch(source,destination,frames):
    values=tone(frames/SR);sf.write(destination,values,SR,subtype='FLOAT');return values

class TimingTests(unittest.TestCase):
    def setUp(self):
        self.backend=patch.object(timing,'stretch',side_effect=fake_stretch)
        self.r3=self.backend.start();self.addCleanup(self.backend.stop)
        self.collision_backend=patch.object(timing,'collision_stretch',side_effect=fake_stretch)
        self.collision_backend.start();self.addCleanup(self.collision_backend.stop)
        self.aligner=patch.object(timing,'alignment_hint',return_value={'status':'unavailable'})
        self.mfa=self.aligner.start();self.addCleanup(self.aligner.stop)

    def aligned_pause(self):
        self.mfa.return_value={'status':'aligned','phones':[],
            'words':[{'start':0.,'end':.4,'label':'하나'},{'start':1.,'end':1.4,'label':'둘'}]}

    def test_longer_than_reference_is_not_accelerated_when_next_dialogue_is_later(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);source=folder/'tts.wav';sf.write(source,tone(1.5),SR,subtype='FLOAT')
            result=timing.bounded_timing_audio(source,folder/'fit.wav',SR,collision_budget_frames=2*SR)
            self.assertEqual(result['tempo'],1.)
            self.assertEqual(result['speed_correction_status'],'not_needed')
            self.assertGreater(sf.info(folder/'fit.wav').frames,SR)
            timing.collision_stretch.assert_not_called()

    def test_collision_speedup_uses_next_start_budget_instead_of_reference_end(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);source=folder/'tts.wav';sf.write(source,tone(2.4),SR,subtype='FLOAT')
            result=timing.bounded_timing_audio(source,folder/'fit.wav',SR,collision_budget_frames=2*SR)
            self.assertEqual(result['collision_budget_frames'],2*SR)
            self.assertEqual(sf.info(folder/'fit.wav').frames,2*SR)
            self.assertEqual(result['overlap_after_speedup_frames'],0)
            self.assertLess(result['tempo'],1.3)

    def test_last_dialogue_has_no_speedup_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);source=folder/'tts.wav';sf.write(source,tone(2.4),SR,subtype='FLOAT')
            result=timing.bounded_timing_audio(source,folder/'fit.wav',SR,collision_budget_frames=None)
            self.assertEqual(result['tempo'],1.)
            self.assertGreater(sf.info(folder/'fit.wav').frames,2*SR)
            timing.collision_stretch.assert_not_called()

    def test_measured_long_pause_is_shortened_before_any_speedup(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);values=np.concatenate([tone(.6),silence(1.2),tone(.6)])
            sf.write(folder/'tts.wav',values,SR,subtype='FLOAT')
            result=timing.bounded_timing_audio(folder/'tts.wav',folder/'fit.wav',SR,collision_budget_frames=round(1.8*SR))
            audio=sf.read(folder/'fit.wav',dtype='float32',always_2d=True)[0]
            self.assertEqual(result['tempo'],1.);timing.collision_stretch.assert_not_called()
            self.assertGreater(result['collision_silence_removed_seconds'],.5)
            np.testing.assert_array_equal(audio[:round(.6*SR)],values[:round(.6*SR)])
            np.testing.assert_array_equal(audio[-round(.6*SR):],values[-round(.6*SR):])

    def test_speedup_preserves_remaining_interior_quiet_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);quiet=np.full((round(.2*SR),2),.00001,np.float32)
            values=np.concatenate([tone(.8),quiet,tone(.8)])
            sf.write(folder/'tts.wav',values,SR,subtype='FLOAT')
            result=timing.bounded_timing_audio(folder/'tts.wav',folder/'fit.wav',SR,collision_budget_frames=round(1.3*SR))
            self.assertFalse(result['quiet_pauses_stretched'])
            self.assertGreaterEqual(result['speedup_speech_chunks'],2)
            self.assertGreater(result['speedup_preserved_quiet_seconds'],.07)
            self.assertEqual(result['overlap_after_speedup_frames'],0)

    def test_measured_edge_and_chunk_statistics_are_json_serializable(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);values=np.concatenate([silence(.5),tone(.8),silence(.3),tone(.8),silence(.4)])
            sf.write(folder/'tts.wav',values,SR,subtype='FLOAT')
            result=timing.bounded_timing_audio(folder/'tts.wav',folder/'fit.wav',SR,collision_budget_frames=round(1.5*SR))
            saved=json.loads(json.dumps(result,ensure_ascii=False))
            self.assertIsInstance(saved['tts_samples_trimmed'],int)
            self.assertGreater(saved['tts_leading_silence_removed_seconds'],.3)
            self.assertGreater(saved['tts_trailing_silence_removed_seconds'],.2)

    def test_quiet_synthesis_noise_after_short_line_does_not_force_speedup(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);rng=np.random.default_rng(42)
            noise=rng.normal(0,.0005,(round(.7*SR),2)).astype(np.float32)
            values=np.concatenate([silence(.07),tone(.37),noise])
            sf.write(folder/'tts.wav',values,SR,subtype='FLOAT')
            result=timing.bounded_timing_audio(folder/'tts.wav',folder/'fit.wav',round(.47*SR),collision_budget_frames=round(.89*SR))
            self.assertEqual(result['tempo'],1.);timing.collision_stretch.assert_not_called()
            self.assertGreater(result['tts_trailing_silence_removed_seconds'],.5)
            self.assertLess(sf.info(folder/'fit.wav').duration,.7)

    def test_persistent_noise_floor_does_not_accelerate_short_speech_at_different_volumes(self):
        for scale in (.1,1.,3.):
            with self.subTest(scale=scale),tempfile.TemporaryDirectory() as temp:
                folder=Path(temp);rng=np.random.default_rng(123)
                voice=np.concatenate([tone(.1),silence(.12),tone(.24)])
                noise=rng.normal(0,.0012,(round(.8*SR),2)).astype(np.float32)
                values=np.concatenate([silence(.08),voice,noise])*scale
                sf.write(folder/'tts.wav',values,SR,subtype='FLOAT')
                result=timing.bounded_timing_audio(folder/'tts.wav',folder/'fit.wav',round(.45*SR),collision_budget_frames=round(.76*SR))
                self.assertEqual(result['tempo'],1.)
                self.assertTrue(result['outer_silence_measurement']['persistent_noise_detected'])
                self.assertGreater(result['tts_trailing_silence_removed_seconds'],.6)
                json.dumps(result)
        timing.collision_stretch.assert_not_called()

    def test_clean_quiet_speech_is_not_mistaken_for_a_noise_floor(self):
        for amplitude in (.001,.01,.1):
            values=tone(.6,amplitude)
            trimmed,head,tail=timing.trim_outer_silence(values)
            self.assertEqual(len(trimmed),len(values));self.assertEqual(head+tail,0)

    def test_quiet_harmonic_opening_is_preserved_beside_loud_speech(self):
        values=np.concatenate([tone(.3,.001),tone(.4,.15),silence(.2)])
        trimmed,head,tail=timing.trim_outer_silence(values)
        self.assertEqual(head,0)
        _,gate=timing.measured_energy_gate(timing.envelope(values),.02,values)
        self.assertTrue(gate['quiet_harmonic_speech_protected'])
        np.testing.assert_array_equal(trimmed[:round(.3*SR)],values[:round(.3*SR)])

    def test_collision_failure_keeps_complete_pause_audio_and_reports_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            source=folder/'tts.wav';sf.write(source,np.concatenate([silence(.2),tone(1.4),silence(.2)]),SR,subtype='FLOAT')
            timing.pause_only_audio(source,folder/'expected.wav',SR)
            expected=sf.read(folder/'expected.wav',dtype='float32',always_2d=True)[0]
            with patch.object(timing,'collision_stretch',side_effect=RuntimeError('No non-overlapping Speedy result')):
                result=timing.bounded_timing_audio(source,folder/'fitted.wav',SR,collision_budget_frames=SR)
            actual=sf.read(folder/'fitted.wav',dtype='float32',always_2d=True)[0]
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(result['speed_correction_status'],'skipped_keep_pause_audio')
            self.assertEqual(result['tempo_engine'],'none')
            self.assertTrue(result['overlap_allowed'])
            self.assertEqual(result['overlap_after_speedup_frames'],len(actual)-SR)
            self.assertIn('No non-overlapping',result['speed_correction_fallback_reason'])

    def test_invalid_speedy_output_is_not_used_as_fallback(self):
        for output in [np.empty((0,2)),np.full((20,2),np.nan),np.zeros((10,1))]:
            with self.subTest(shape=output.shape),tempfile.TemporaryDirectory() as temp:
                folder=Path(temp);sf.write(folder/'tts.wav',tone(1.4),SR,subtype='FLOAT')
                with patch.object(timing,'collision_stretch',return_value=output):
                    result=timing.bounded_timing_audio(folder/'tts.wav',folder/'fit.wav',SR,collision_budget_frames=SR)
                actual=sf.read(folder/'fit.wav',dtype='float32',always_2d=True)[0]
                self.assertGreater(len(actual),SR)
                self.assertTrue(np.isfinite(actual).all())
                self.assertEqual(result['speed_correction_status'],'skipped_keep_pause_audio')

    def test_bounded_invalid_original_tts_still_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);sf.write(folder/'tts.wav',silence(1),SR,subtype='FLOAT')
            with self.assertRaisesRegex(RuntimeError,'무음'):
                timing.bounded_timing_audio(folder/'tts.wav',folder/'fit.wav',SR,collision_budget_frames=SR//2)

    def test_valid_speedy_output_with_residual_overlap_is_not_truncated(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);sf.write(folder/'tts.wav',tone(1.4),SR,subtype='FLOAT')
            def overrun(source,dest,frames):return tone((frames+100)/SR)
            with patch.object(timing,'collision_stretch',side_effect=overrun):
                result=timing.bounded_timing_audio(folder/'tts.wav',folder/'fit.wav',SR,collision_budget_frames=SR)
            self.assertEqual(result['overlap_after_speedup_frames'],100)
            self.assertTrue(result['overlap_allowed'])
            self.assertEqual(result['speed_correction_status'],'applied')

    def test_both_directions_adjust_pauses_before_r3_and_preserve_voiced_samples(self):
        self.aligned_pause()
        for target in [SR,round(1.6*SR)]:
            with self.subTest(target=target),tempfile.TemporaryDirectory() as temp:
                folder=Path(temp);values=np.concatenate([tone(.4),silence(.6),tone(.4)])
                sf.write(folder/'tts.wav',values,SR,subtype='FLOAT')
                report=timing.fit_audio(folder/'tts.wav',folder/'fitted.wav',target,text='하나 둘')
                result=sf.read(folder/'fitted.wav',dtype='float32',always_2d=True)[0]
                np.testing.assert_array_equal(result[:round(.4*SR)],values[:round(.4*SR)])
                np.testing.assert_array_equal(result[-round(.4*SR):],values[-round(.4*SR):])
                self.assertEqual(report['tempo_engine'],'none');self.r3.assert_not_called()
                self.assertEqual(report['mfa_status'],'aligned')
                self.assertEqual(report['pause_adjustment_share'],1)
                self.assertEqual(sum(c['added_frames'] for c in report['pause_adjustments']),target-len(values))
                self.assertEqual(sf.info(folder/'fitted.wav').frames,target)

    def test_only_the_difference_remaining_after_pause_compression_uses_r3(self):
        self.aligned_pause()
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);body=np.concatenate([tone(.4),silence(.6),tone(.4)])
            sf.write(folder/'tts.wav',body,SR,subtype='FLOAT');sent=[]
            def backend(source,dest,frames):
                sent.append(sf.read(source,dtype='float32',always_2d=True)[0]);return fake_stretch(source,dest,frames)
            self.r3.side_effect=backend
            target=round(.7*SR)
            report=timing.fit_audio(folder/'tts.wav',folder/'fit.wav',target,text='하나 둘')
            self.assertGreater(report['pause_removed_seconds'],.4)
            self.assertEqual(len(sent[0]),len(body)+sum(c['added_frames'] for c in report['pause_adjustments']))
            self.assertAlmostEqual(report['tempo'],len(sent[0])/target,places=5)
            self.assertGreater(report['pause_adjustment_share'],.5)
            self.assertEqual(sf.info(folder/'fit.wav').frames,target)

    def test_failed_alignment_or_in_word_pause_keeps_pauses_intact(self):
        for hint in [{'status':'failed','reason':'alignment failed'}, {'status':'unavailable'},
                     {'status':'aligned','words':[{'start':0.,'end':1.4,'label':'기뻐하자'}],'phones':[]}]:
            with self.subTest(status=hint['status']),tempfile.TemporaryDirectory() as temp:
                self.mfa.return_value=hint;folder=Path(temp)
                body=np.concatenate([tone(.4),silence(.6),tone(.4)])
                sf.write(folder/'tts.wav',body,SR,subtype='FLOAT');sent=[]
                def backend(source,dest,frames):
                    sent.append(sf.read(source,dtype='float32',always_2d=True)[0]);return fake_stretch(source,dest,frames)
                self.r3.side_effect=backend
                report=timing.fit_audio(folder/'tts.wav',folder/'fit.wav',3*SR,text='기뻐하자')
                np.testing.assert_array_equal(sent[0],body)
                self.assertEqual(report['pause_adjustments'],[])
                self.assertEqual(report['tempo_engine'],'google_speedy')

    def test_r3_slowdown_has_no_point85_floor(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);sf.write(folder/'tts.wav',tone(.5),SR,subtype='FLOAT')
            result=timing.fit_audio(folder/'tts.wav',folder/'fit.wav',2*SR)
            self.assertAlmostEqual(result['tempo'],.25,places=5)
            self.assertEqual(result['unfilled_speech_seconds'],0)

    def test_quiet_stop_closure_never_splits_the_onset_even_at_a_word_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);values=tone(.9);values[round(.42*SR):round(.44*SR)]=0
            sf.write(folder/'tts.wav',values,SR,subtype='FLOAT')
            sf.write(folder/'reference.wav',tone(2),SR,subtype='FLOAT')
            self.mfa.return_value={'status':'aligned','words':[{'start':0.,'end':.4,'label':'앞말'},
                {'start':.4,'end':.9,'label':'뒷말'}], 'phones':[{'start':0.,'end':.4,'label':'a'},
                {'start':.4,'end':.46,'label':'tʷ'},{'start':.46,'end':.9,'label':'a'}]}
            sent=[]
            def backend(source,dest,frames):
                sent.append(sf.read(source,dtype='float32',always_2d=True)[0])
                return fake_stretch(source,dest,frames)
            self.r3.side_effect=backend
            with patch('urllib.request.urlopen',side_effect=AssertionError('No pause planning API')):
                report=timing.fit_audio(folder/'tts.wav',folder/'fit.wav',2*SR,folder/'reference.wav',
                    '앞말 뒷말',original_text='原文',planner_key='unused')
            self.r3.assert_called_once();self.assertEqual(report['new_pause_count'],0)
            np.testing.assert_array_equal(sent[0],values)
            self.assertEqual(sf.info(folder/'fit.wav').frames,2*SR)
            self.assertEqual(report['pause_planning_api_requests'],0)

    def test_stale_gemini_plan_cannot_override_local_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);values=tone(1)
            for name in ['tts.wav','reference.wav']:sf.write(folder/name,values,SR,subtype='FLOAT')
            with patch('urllib.request.urlopen',side_effect=AssertionError('No Gemini request')):
                report=timing.fit_audio(folder/'tts.wav',folder/'fit.wav',SR,folder/'reference.wav',
                    '번역문',original_text='원문',planner_key='unused',planned_direction={'plan':'obsolete'})
            self.assertEqual(report['tempo'],1)
            self.assertEqual(report['pause_planning'],'algorithmic_mfa_and_waveform')
            audio=sf.read(folder/'fit.wav',dtype='float32',always_2d=True)[0]
            np.testing.assert_array_equal(audio,values)

    def test_unity_does_not_resynthesize_or_start_a_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);sf.write(folder/'tts.wav',tone(1),SR,subtype='FLOAT')
            result=timing.fit_audio(folder/'tts.wav',folder/'fit.wav',SR)
            self.assertEqual(result['tempo_engine'],'none');self.r3.assert_not_called()
            self.mfa.assert_not_called()

    def test_outer_margin_is_used_without_extending_the_segment(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            sf.write(folder/'reference.wav',np.concatenate([tone(.8),silence(.2)]),SR,subtype='FLOAT')
            sf.write(folder/'tts.wav',tone(.9),SR,subtype='FLOAT')
            result=timing.fit_audio(folder/'tts.wav',folder/'fit.wav',SR,folder/'reference.wav')
            self.assertEqual(result['tempo'],1);self.assertGreater(result['source_margin_used_seconds'],0)
            self.r3.assert_not_called()

    def test_old_seed_cache_refits_without_a_new_fish_request(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            sf.write(folder/'clean.wav',tone(2),SR,subtype='FLOAT')
            sf.write(folder/'0000-tts.mp3',tone(.8),SR,format='WAV',subtype='FLOAT')
            sf.write(folder/'0000-fitted.wav',silence(2),SR,subtype='FLOAT')
            policy=render.tts_policy('ko')
            (folder/'0000-generation.json').write_text(json.dumps({'generation_id':'cached','tts_policy':policy}))
            (folder/'0000.json').write_text(json.dumps({'fit_version':34,'timing_method':'verified_pauses_then_seed_vc_speedup_only_on_next_dialogue_overlap','tempo_engine':'seed_vc_44k_f0','tts_policy':policy,'speech_window_frames':[0,2*SR]}))
            segment=dict(lo=0,hi=2*SR,start=0.,end=2.,original_text='original',translated_text='translated')
            with patch.object(render,'request_audio',return_value={'generation_id':'cached','tts_policy':policy}):
                render.generate_speech(0,segment,folder/'clean.wav',folder,dict(language='ko',key='unused'),force_prepared=True)
            with patch.object(render,'request_audio',side_effect=AssertionError('Reuse Fish cache')):
                report=render.synthesize(0,segment,folder/'clean.wav',folder,dict(language='ko',key='unused'))
            self.assertEqual(report['fit_version'],timing.FIT_VERSION);self.assertEqual(report['generation_id'],'cached')
            self.assertEqual(report['tempo_engine'],'none')
            self.assertEqual(sf.info(folder/'0000-fitted.wav').frames,round(.8*SR))
            with patch.object(render,'bounded_timing_audio',side_effect=AssertionError('Reuse pause/overlap cache')):
                self.assertEqual(render.synthesize(0,segment,folder/'clean.wav',folder,dict(language='ko')),report)

    def test_engine_change_reuses_previously_selected_tts_retry_even_above_two_x(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            sf.write(folder/'clean.wav',tone(1),SR,subtype='FLOAT')
            sf.write(folder/'0000-tts.mp3',tone(2.4),SR,format='WAV',subtype='FLOAT')
            generation={'generation_id':'selected-retry','tts_policy':render.tts_policy('ko')}
            for name in ['0000-generation.json','0000-alternate.json']:
                (folder/name).write_text(json.dumps(generation))
            segment=dict(lo=0,hi=SR,start=0.,end=1.,original_text='original',translated_text='translated',next_dialogue_start_frame=SR)
            with patch.object(render,'request_audio',return_value=generation):
                render.generate_speech(0,segment,folder/'clean.wav',folder,dict(language='ko',key='unused'),force_prepared=True)
            with patch.object(render,'request_audio',side_effect=AssertionError('Do not regenerate an already selected retry')):
                result=render.synthesize(0,segment,folder/'clean.wav',folder,dict(language='ko',key='unused'))
            self.assertAlmostEqual(result['tempo'],2.4,places=5)
            self.assertEqual(result['generation_id'],'selected-retry')
            self.assertEqual(sf.info(folder/'0000-fitted.wav').frames,SR)

    def test_failed_r3_does_not_overwrite_previous_result(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);sf.write(folder/'tts.wav',tone(.7),SR,subtype='FLOAT')
            sf.write(folder/'fit.wav',tone(1),SR,subtype='FLOAT');previous=(folder/'fit.wav').read_bytes()
            self.r3.side_effect=RuntimeError('R3 failed')
            with self.assertRaisesRegex(RuntimeError,'R3 failed'):timing.fit_audio(folder/'tts.wav',folder/'fit.wav',SR)
            self.assertEqual((folder/'fit.wav').read_bytes(),previous)

    def test_invalid_tts_does_not_call_r3(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);sf.write(folder/'tts.wav',silence(1),SR,subtype='FLOAT')
            with self.assertRaises(RuntimeError):timing.fit_audio(folder/'tts.wav',folder/'fit.wav',SR)
            self.r3.assert_not_called()

if __name__=='__main__':unittest.main()
