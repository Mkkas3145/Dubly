from pathlib import Path
import tempfile,unittest,json
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR
import timing,render

class OuterSilenceTests(unittest.TestCase):
    def test_leading_gap_is_kept_unless_collision_requires_speedup(self):
        pad=np.zeros((SR//10,2),np.float32)
        quiet=np.zeros((SR,2),np.float32)
        breath=np.full((SR//5,2),.001,np.float32)
        voice=np.full((SR,2),.1,np.float32)
        values=np.concatenate([pad,quiet,breath,voice])
        kept,removed=timing.reduce_quiet_head_for_collision(values,len(values),len(pad))
        np.testing.assert_array_equal(kept,values);self.assertEqual(removed,0)
        reduced,removed=timing.reduce_quiet_head_for_collision(values,len(values)-SR//2,len(pad))
        self.assertEqual(removed,SR//2)
        np.testing.assert_array_equal(reduced[:len(pad)],pad)
        np.testing.assert_array_equal(reduced[-len(breath)-len(voice):],np.concatenate([breath,voice]))
        reduced,removed=timing.reduce_quiet_head_for_collision(values,SR,len(pad))
        self.assertLessEqual(removed,SR-round(.08*SR))
        np.testing.assert_array_equal(reduced[-len(breath)-len(voice):],np.concatenate([breath,voice]))

    def test_individual_generation_keeps_long_tail_when_no_acceleration_is_needed(self):
        rng=np.random.default_rng(10)
        voice=np.full((SR,2),.1,np.float32);breath=rng.normal(0,.001,(SR,2)).astype(np.float32)
        values=np.concatenate([np.zeros((SR,2),np.float32),voice,breath])
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);sf.write(p/'source.wav',values,SR,subtype='FLOAT')
            report=timing.bounded_timing_audio(p/'source.wav',p/'out.wav',3*SR,collision_budget_frames=3*SR,preserve_tail=True)
            result,_=sf.read(p/'out.wav',dtype='float32',always_2d=True)
            self.assertGreater(report['tts_leading_silence_removed_seconds'],.8)
            self.assertEqual(report['tts_trailing_silence_removed_seconds'],0)
            self.assertEqual(report['unspoken_tail_removed_seconds'],0)
            np.testing.assert_array_equal(result[-SR:],breath)

    def test_tail_reduction_is_conditional_protects_words_and_fades_breath(self):
        voice=np.full((SR,2),.1,np.float32);breath=np.full((SR,2),.001,np.float32)
        values=np.concatenate([voice,breath])
        unchanged,report=timing.reduce_unspoken_tail(values,round(2.5*SR),SR)
        np.testing.assert_array_equal(unchanged,values);self.assertEqual(report['removed_frames'],0)
        reduced,report=timing.reduce_unspoken_tail(values,round(1.2*SR),SR)
        self.assertEqual(len(reduced),round(1.2*SR))
        np.testing.assert_array_equal(reduced[:SR],voice)
        self.assertGreater(report['fade_frames'],0);self.assertTrue(np.all(reduced[-1]==0))
        # A budget shorter than the words cannot authorize truncating them.
        shortened,report=timing.reduce_unspoken_tail(values,round(.7*SR),SR)
        self.assertEqual(len(shortened),SR+round(.08*SR))
        np.testing.assert_array_equal(shortened[:SR],voice)
        unverified,report=timing.reduce_unspoken_tail(values,SR,None)
        np.testing.assert_array_equal(unverified,values)
        self.assertEqual(report['status'],'no_verified_spoken_endpoint')

    def test_verified_tail_is_reduced_before_requesting_collision_acceleration(self):
        values=np.concatenate([np.full((SR,2),.1,np.float32),np.full((SR,2),.001,np.float32)])
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);sf.write(p/'source.wav',values,SR,subtype='FLOAT')
            hint={'status':'aligned','words':[{'start':0.,'end':1.,'label':'말'}]}
            with patch('timing.alignment_hint',return_value=hint),patch('timing.source_guided_plan',return_value={'boundaries':[],'matches':[],'mfa_reviews':[]}),patch('timing.collision_stretch',side_effect=AssertionError('Reduce breath first')):
                report=timing.bounded_timing_audio(p/'source.wav',p/'out.wav',SR,text='말',collision_budget_frames=round(1.2*SR),preserve_outer_edges=True)
            self.assertEqual(report['tempo'],1.)
            self.assertAlmostEqual(report['unspoken_tail_removed_seconds'],.8)
            result,_=sf.read(p/'out.wav',dtype='float32',always_2d=True)
            np.testing.assert_array_equal(result[:SR],values[:SR]);self.assertTrue(np.all(result[-1]==0))

    def test_verified_batch_slice_keeps_long_quiet_breath_at_both_edges(self):
        rng=np.random.default_rng(7)
        inhale=rng.normal(0,.001,(round(.6*SR),2)).astype(np.float32)
        voice=np.full((SR,2),.1,np.float32)
        exhale=rng.normal(0,.001,(round(1.1*SR),2)).astype(np.float32)
        values=np.concatenate([inhale,voice,exhale])
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);sf.write(p/'source.wav',values,SR,subtype='FLOAT')
            with patch('timing.trim_outer_silence',side_effect=AssertionError('Batch edges must stay intact')):
                report=timing.bounded_timing_audio(p/'source.wav',p/'out.wav',3*SR,preserve_outer_edges=True)
            result,_=sf.read(p/'out.wav',dtype='float32',always_2d=True)
            np.testing.assert_array_equal(result,values)
            self.assertEqual(report['tts_leading_silence_removed_seconds'],0)
            self.assertEqual(report['tts_trailing_silence_removed_seconds'],0)
            self.assertTrue(report['verified_batch_edges_preserved'])

    def test_batch_collision_speedup_receives_complete_breath_instead_of_trimmed_body(self):
        voice=np.full((SR,2),.1,np.float32)
        breath=np.full((SR,2),.001,np.float32);values=np.concatenate([voice,breath])
        seen=[]
        def stretch(body,destination,frames):
            seen.append(body.copy());return body[::2],0,1
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);sf.write(p/'source.wav',values,SR,subtype='FLOAT')
            with patch('timing.shorten_collision_pauses',side_effect=lambda audio,budget:(audio,[])),patch('timing.stretch_speech_keep_pauses',side_effect=stretch):
                timing.bounded_timing_audio(p/'source.wav',p/'out.wav',SR,collision_budget_frames=SR,preserve_outer_edges=True)
        np.testing.assert_array_equal(seen[0],values)

    def test_low_noisy_breath_and_short_release_survive_tail_trimming(self):
        rng=np.random.default_rng(42)
        voice=np.full((SR,2),.1,np.float32)
        breath=rng.normal(0,.0015,(round(.2*SR),2)).astype(np.float32)
        release=np.full((round(.02*SR),2),.003,np.float32)
        spoken=np.concatenate([voice,breath,release])
        values=np.concatenate([spoken,np.zeros((round(.08*SR),2),np.float32)])
        with patch('timing.measured_energy_gate',return_value=(.015,{})):
            result,head,tail=timing.trim_outer_silence(values)
        self.assertEqual(head,0)
        np.testing.assert_array_equal(result[:len(spoken)],spoken)
        self.assertEqual(tail,0)
        np.testing.assert_array_equal(result,values)

    def test_long_external_silence_does_not_trigger_acceleration(self):
        voice=np.full((SR,2),.1,dtype=np.float32)
        values=np.concatenate([np.zeros((5*SR,2),dtype=np.float32),voice,np.zeros((2*SR,2),dtype=np.float32)])
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);sf.write(p/'tts.wav',values,SR,subtype='FLOAT')
            sf.write(p/'ref.wav',voice,SR,subtype='FLOAT')
            with patch('timing.alignment_hint',return_value={'status':'unavailable'}),patch('timing.collision_stretch',side_effect=AssertionError('Silence must not cause speedup')):
                report=timing.bounded_timing_audio(p/'tts.wav',p/'out.wav',SR,p/'ref.wav','text',collision_budget_frames=round(1.2*SR))
            result,_=sf.read(p/'out.wav',dtype='float32',always_2d=True)
            self.assertEqual(report['tempo_engine'],'none')
            self.assertGreater(report['tts_leading_silence_removed_seconds'],4.9)
            self.assertGreaterEqual(report['tts_trailing_silence_removed_seconds'],1.89)
            onset=int(np.flatnonzero(np.max(np.abs(result),axis=1)>.05)[0])
            np.testing.assert_array_equal(result[onset:onset+SR],voice)
            json.dumps(report)

    def test_quiet_edges_and_internal_pause_are_retained(self):
        x=np.full((2*SR,2),.1,dtype=np.float32)
        x[:SR//10]=.0003;x[-SR//10:]=.0003;x[SR//2:SR]=0
        result,head,tail=timing.trim_outer_silence(x)
        np.testing.assert_array_equal(result,x)
        self.assertEqual((head,tail),(0,0))

    def test_default_sampling_options_and_expression_max_cache_is_invalidated(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);sf.write(p/'ref.wav',np.ones(SR,dtype=np.float32)*.1,SR)
            with patch('render.urllib.request.urlopen',side_effect=RuntimeError('capture')) as request:
                with self.assertRaisesRegex(RuntimeError,'capture'):
                    render.request_audio(p/'ref.wav',{'original_text':'source','translated_text':'target'},'ko','unused',p/'tts.mp3')
            payload=json.loads(request.call_args.args[0].data)
            self.assertNotIn('temperature',payload)
            self.assertNotIn('top_p',payload)
            self.assertEqual(payload['provider']['options']['fish-audio'],
                {'latency':'normal'})
            expected='native Korean, The listener should perceive the voice as the same speaker as in the reference.'
            self.assertEqual(payload['instructions'],expected)
            self.assertEqual(render.tts_instructions('ko'),expected)
            self.assertEqual(render.tts_instructions('ko',brief=True),expected)
            import hashlib
            old=[render.FISH,'ko',render.tts_instructions('ko'),render.tts_instructions('ko',brief=True),render.REFERENCE_MARGIN_POLICY,{'temperature':1.0,'top_p':1.0,'latency':'normal'}]
            old_policy=hashlib.sha256(json.dumps(old,ensure_ascii=False).encode()).hexdigest()
            self.assertNotEqual(old_policy,render.tts_policy('ko'))

if __name__=='__main__':unittest.main()
