from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import SR
from reference_clean import reference_windows,edge_guard
from render import reference_audio,prepare_segments,synthesize
from render import mix,transition_weights
from reference_timing import refine_references,exclusions_outside_references

def sample(sr,offset=0.,gain=1.,burst=True,inside_word=False):
    t=np.arange(round(1.5*sr))/sr
    x=np.zeros(len(t),dtype=np.float32)
    voice=(t>=.3)&(t<.6)
    x[voice]=.2*np.sin(2*np.pi*220*t[voice])
    tail=(t>=.6)&(t<1.1)
    x[tail]=.2*np.exp(-(t[tail]-.6)*12)*np.sin(2*np.pi*220*t[tail])
    if burst:
        a,b=round(.83*sr),round(.845*sr)
        rng=np.random.default_rng(7)
        x[a:b]+=.12*rng.normal(size=b-a).astype(np.float32)*np.hanning(b-a)
    data=np.r_[np.zeros(round(offset*sr),dtype=np.float32),x*gain]
    measure={'start':offset+.25,'end':offset+1.1,'aligned_start':offset+.3,'aligned_end':offset+(.9 if inside_word else .6)}
    return data,measure

class ReferenceCleanTests(unittest.TestCase):
    def test_post_word_attack_sets_shared_end_and_protected_original(self):
        x,m=sample(SR);window=reference_windows(x,SR,[m])[0]
        self.assertIsNotNone(window['tail_transient'])
        self.assertEqual(window['start_frame'],round(m['start']*SR))
        self.assertLess(window['end'],.84);self.assertGreater(window['end'],.7)
        self.assertEqual(window['replacement_end'],window['end'])
        self.assertEqual(window['boundary_policy'],'shared_clean_interval')
        protected=window['preserved_intervals'][0]
        self.assertGreater(protected['start'],window['end'])
        self.assertEqual(protected['end'],1.1)

    def test_smooth_voice_decay_is_not_trimmed(self):
        x,m=sample(SR,burst=False);window=reference_windows(x,SR,[m])[0]
        self.assertEqual(window['end'],1.1)

    def test_phoneme_inside_last_word_is_never_removed(self):
        x,m=sample(SR,inside_word=True);window=reference_windows(x,SR,[m])[0]
        self.assertEqual(window['end'],1.1)

    def test_early_asr_end_does_not_turn_a_final_consonant_into_an_effect(self):
        for sr in [16000,44100,48000]:
            for gain in [.1,1.,2.]:
                with self.subTest(sr=sr,gain=gain):
                    x,m=sample(sr,gain=gain)
                    t=np.arange(len(x))/sr
                    # The final word is still audible when its stop release
                    # changes spectrum; ASR incorrectly ended it at .6.
                    active=(t>=.6)&(t<.86)
                    x[active]+=.08*gain*np.sin(2*np.pi*340*t[active])
                    x=np.stack([x,-x],axis=1)
                    window=reference_windows(x,sr,[m])[0]
                    self.assertIsNone(window['tail_transient'])
                    self.assertEqual(window['end'],m['end'])

    def test_sustained_new_sound_after_voice_decay_is_not_part_of_reference(self):
        for sr in [16000,44100,48000]:
            for offset,gain in [(0.,.1),(3.173,1.),(10.291,2.)]:
                with self.subTest(sr=sr,offset=offset,gain=gain):
                    t=np.arange(round(3*sr))/sr
                    x=np.full(len(t),.002,dtype=np.float32)
                    voice=(t>=.3)&(t<.8)
                    x[voice]+=.2*np.sin(2*np.pi*220*t[voice])
                    tail=(t>=.8)&(t<1.2)
                    x[tail]+=.2*np.exp(-(t[tail]-.8)*20)*np.sin(2*np.pi*220*t[tail])
                    event=(t>=1.4)&(t<2.8)
                    x[event]+=.2*np.sin(2*np.pi*1700*t[event])
                    x=np.r_[np.zeros(round(offset*sr),dtype=np.float32),x*gain]
                    measure={'start':offset+.25,'end':offset+2.9,'aligned_start':offset+.3,'aligned_end':offset+.8}
                    window=reference_windows(x,sr,[measure])[0]
                    self.assertLess(window['end'],offset+1.4)
                    self.assertGreater(window['end'],offset+1.)
                    self.assertEqual(window['tail_transient']['kind'],'sustained_event_after_voice_decay')
                    self.assertTrue(window['preserved_intervals'])

    def test_same_voice_echo_after_quiet_is_kept(self):
        sr=44100;t=np.arange(3*sr)/sr
        x=np.zeros(len(t),dtype=np.float32)
        for start,end,amp in [(.3,.8,.2),(1.4,1.8,.03)]:
            mask=(t>=start)&(t<end)
            x[mask]=amp*np.sin(2*np.pi*220*t[mask])
        measure={'start':.25,'end':2.,'aligned_start':.3,'aligned_end':.8}
        self.assertEqual(reference_windows(x,sr,[measure])[0]['end'],2.)

    def test_shift_gain_rate_and_stereo_do_not_require_special_cases(self):
        for sr in [16000,44100,48000]:
            for shift,gain in [(0.,.1),(3.173,1.),(10.291,2.)]:
                with self.subTest(sr=sr,shift=shift,gain=gain):
                    x,m=sample(sr,shift,gain);x=np.stack([x,-x],axis=1)
                    window=reference_windows(x,sr,[m])[0]
                    self.assertEqual(window['start_frame'],round(m['start']*sr))
                    self.assertIsNotNone(window['tail_transient'])
                    self.assertLess(window['end']-shift,.84)
                    self.assertGreater(window['end']-shift,.7)

    def test_edge_guard_preserves_middle_samples_exactly(self):
        x=np.random.default_rng(2).normal(size=2000).astype(np.float32)
        result=edge_guard(x,88,88)
        np.testing.assert_array_equal(result[88:-88],x[88:-88])
        self.assertEqual(result[0],0);self.assertEqual(result[-1],0)

    def test_clean_reference_length_is_preserved_without_forcing_tts_to_match_it(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);x,m=sample(SR);sf.write(folder/'clean.wav',np.stack([x,x],axis=1),SR,subtype='FLOAT')
            window=reference_windows(x,SR,[m])[0]
            transcript={'segments':[dict(start=window['start'],end=window['end'],original_text='source',translated_text='target')],'reference_windows':[window]}
            segment=prepare_segments(transcript,len(x))[0]
            sf.write(folder/'0000-tts.mp3',np.ones(SR//5,dtype=np.float32)*.1,SR,format='WAV')
            with (patch('render.request_audio',return_value={'generated_seconds':.2}),
                  patch('timing.alignment_hint',return_value={'status':'unavailable'}),
                  patch('timing.collision_stretch',side_effect=AssertionError('No next dialogue; no acceleration'))):
                report=synthesize(0,segment,folder/'clean.wav',folder,{'key':'test','language':'ko'})
            tight_frames=sf.info(folder/'0000-timing-reference.wav').frames
            self.assertEqual(tight_frames,segment['hi']-segment['lo'])
            self.assertGreaterEqual(sf.info(folder/'0000-reference.wav').frames,tight_frames)
            self.assertLessEqual(sf.info(folder/'0000-reference.wav').frames-tight_frames,round(.4*SR))
            self.assertEqual(sf.info(folder/'0000-fitted.wav').frames,SR//5+report['placement_offset_frames'])
            reference,_=sf.read(folder/'0000-reference.wav');self.assertEqual(reference[-1],0)

    def test_finalized_boundaries_restore_original_with_standard_100ms_transition(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);x,m=sample(SR);voice=np.stack([x,x],axis=1)
            sf.write(folder/'clean.wav',voice,SR,subtype='FLOAT')
            original=voice+.1;sf.write(folder/'original.wav',original,SR,subtype='FLOAT')
            evidence={'segments':[{'source_index':0,'start':.3,'end':.6,
                'words':[{'word':'source','start':.3,'end':.6}]}],
                'model':'elevenlabs/scribe-v2','usage':{},'unmatched_segments':[]}
            transcript={'segments':[dict(start=.2,end=1.3,original_text='source',translated_text='target')],
                'source_languages':['ja'],'excluded_intervals':[]}
            with patch('reference_timing.source_evidence',return_value=evidence),patch('reference_edges.acoustic_boundaries',return_value=[dict(m)]):
                result,measurements=refine_references(folder/'clean.wav',transcript,folder/'boundaries')
            window=result['reference_windows'][0];segment=prepare_segments(result,len(x))[0]
            self.assertEqual(segment['lo'],segment['reference_lo'])
            self.assertEqual(segment['hi'],segment['reference_hi'])
            self.assertEqual(result['segments'][0]['end'],measurements[0]['end'])
            protected=result['reference_tail_intervals'][0]
            self.assertGreater(protected['start'],window['end'])
            self.assertEqual(protected['end'],m['end'])
            intervals=[{'lo':round(protected['start']*SR),'hi':round(protected['end']*SR)}]
            weights=transition_weights(0,len(x),[segment],intervals)
            hi=segment['hi'];attack=intervals[0]['lo']
            self.assertEqual(weights[hi],1)
            self.assertTrue(np.any((weights[hi:attack]>0)&(weights[hi:attack]<1)))
            self.assertLess(float(np.max(np.abs(np.diff(weights[hi-1:attack+1])))),.02)
            np.testing.assert_array_equal(weights,transition_weights(0,len(x),[segment]))
            self.assertAlmostEqual(float(weights[hi+round(.15*SR)]),0.,places=5)
            sf.write(folder/'0000-fitted.wav',np.full((segment['hi']-segment['lo'],2),.03,dtype=np.float32),SR,subtype='FLOAT')
            mix(folder/'original.wav',folder/'clean.wav',[segment],folder,folder/'mix.wav',lambda *args:None,result['excluded_intervals'])
            mixed,_=sf.read(folder/'mix.wav',dtype='float32',always_2d=True)
            np.testing.assert_array_equal(mixed[hi+round(.7*SR):],original[hi+round(.7*SR):])

    def test_shared_policy_rejects_a_different_replacement_range(self):
        x,m=sample(SR);window=reference_windows(x,SR,[m])[0]
        transcript={'segments':[dict(start=m['start'],end=m['end'],original_text='source',translated_text='target')],'reference_windows':[window]}
        with self.assertRaisesRegex(RuntimeError,'일치하지 않습니다'):
            prepare_segments(transcript,len(x))

    def test_gemini_exclusion_cannot_trim_or_reject_a_measured_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);x,m=sample(SR,burst=False)
            sf.write(folder/'clean.wav',x,SR,subtype='FLOAT')
            evidence={'segments':[{'source_index':0,'start':.3,'end':.6,
                'words':[{'word':'source','start':.3,'end':.6}]}],
                'model':'elevenlabs/scribe-v2','usage':{},'unmatched_segments':[]}
            exclusions=[{'start':.35,'end':.5,'reason':'nonlexical_vocalization'},
                {'start':.1,'end':.32,'reason':'singing'},
                {'start':1.,'end':1.3,'reason':'mixed_speakers'}]
            transcript={'segments':[dict(start=.2,end=1.3,original_text='source',translated_text='target')],
                'source_languages':['ja'],'excluded_intervals':exclusions}
            with patch('reference_timing.source_evidence',return_value=evidence),patch('reference_edges.acoustic_boundaries',return_value=[dict(m)]):
                result,measurements=refine_references(folder/'clean.wav',transcript,folder/'boundaries')
            segment=prepare_segments(result,len(x))[0]
            self.assertEqual((segment['start'],segment['end']),(m['start'],m['end']))
            self.assertEqual((segment['lo'],segment['hi']),(segment['reference_lo'],segment['reference_hi']))
            self.assertEqual(result['excluded_intervals'],[
                {'start':.1,'end':.25,'reason':'singing'},
                {'start':1.1,'end':1.3,'reason':'mixed_speakers'}])
            self.assertEqual(len(result['excluded_interval_adjustments']),3)
            self.assertEqual(transcript['excluded_intervals'],exclusions)
            protected=[{'lo':round(e['start']*SR),'hi':round(e['end']*SR)} for e in result['excluded_intervals']]
            weights=transition_weights(0,len(x),[segment],protected)
            np.testing.assert_array_equal(weights[segment['lo']:segment['hi']],1)

    def test_exclusion_can_split_around_multiple_references_without_moving_them(self):
        windows=[{'start_frame':10,'end_frame':20},{'start_frame':30,'end_frame':40}]
        intervals=[{'start':.05,'end':.45,'reason':'nonlexical_vocalization'}]
        kept,changes=exclusions_outside_references(intervals,windows,50,100)
        self.assertEqual([(e['start'],e['end']) for e in kept],[(.05,.1),(.2,.3),(.4,.45)])
        self.assertEqual(len(changes),1)
        touching=[{'start':0.,'end':.1,'reason':'singing'},{'start':.4,'end':.5,'reason':'singing'}]
        self.assertEqual(exclusions_outside_references(touching,windows,50,100),(touching,[]))

if __name__=='__main__':unittest.main()
