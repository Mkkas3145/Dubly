import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference_edges import acoustic_boundaries

def word(start,end,probability=1.):return dict(word='word',start=start,end=end,probability=probability)
def row(start,end,confidence=1.):return dict(start=start,end=end,words=[word(start,end)],confidence=confidence)

def recording(sr,duration,voices):
    values=np.zeros(round(sr*duration),dtype=np.float32)
    for start,end,tail,amp in voices:
        lo,hi=round(start*sr),round(end*sr)
        values[lo:hi]+=amp*np.sin(np.arange(hi-lo)*2*np.pi*233/sr)
        stop=min(len(values),round((end+tail)*sr))
        if stop>hi:
            time=np.arange(stop-hi)/max(1,stop-hi)
            values[hi:stop]+=amp*np.exp(-time*7)*np.sin(np.arange(stop-hi)*2*np.pi*233/sr)
    return values

class ReferenceTimingTests(unittest.TestCase):
    def test_soft_final_syllable_is_measured_without_requiring_loud_phrase_level(self):
        for sr in [16000,44100,48000]:
            for gain in [.2,1.,2.]:
                x=recording(sr,2,[(.3,.8,0,.2*gain),(1.1,1.28,.05,.003*gain)])
                measured=acoustic_boundaries(x,sr,[dict(words=[word(.3,.8),word(1.15,1.3)])])
                self.assertEqual(measured[0]['end_anchor_recovery'],'local_final_word_with_acoustic_evidence')
                self.assertGreaterEqual(measured[0]['end'],1.28)

    def test_stationary_noise_at_missing_final_word_is_not_treated_as_voice(self):
        sr=44100;x=recording(sr,3,[(.3,.8,0,.2)])
        rng=np.random.default_rng(42)
        x[sr:2*sr]=rng.normal(0,.0003,sr)
        with self.assertRaisesRegex(RuntimeError,'마지막 발성'):
            acoustic_boundaries(x,sr,[dict(words=[word(.3,.8),word(1.4,1.6)])])

    def test_late_word_anchor_with_search_start_inside_voice_recovers_attack(self):
        for sr in [16000,44100,48000]:
            for gain in [.02,.2,.8]:
                with self.subTest(sr=sr,gain=gain):
                    x=recording(sr,3,[(.5,.85,0,gain),(1.,1.4,.1,gain)])
                    measured=acoustic_boundaries(x,sr,[row(1.,1.4)])
                    self.assertLessEqual(measured[0]['start'],.5)
                    self.assertGreater(measured[0]['start'],.4)
                    self.assertEqual(measured[0]['start_anchor_recovery'],'search_edge_continuation_to_sustained_quiet')

    def test_search_edge_recovery_does_not_cross_previous_voice(self):
        sr=44100;x=recording(sr,3,[(.3,.65,0,.2),(.9,1.35,0,.2)])
        measured=acoustic_boundaries(x,sr,[row(.3,.65),row(1.25,1.35)])
        self.assertGreaterEqual(measured[1]['start'],.8)
        self.assertLessEqual(measured[1]['start'],.91)
        self.assertLessEqual(measured[0]['end_frame'],measured[1]['start_frame'])

    def test_trailing_voice_decay_is_kept_past_word_timestamp(self):
        sr=44100;x=recording(sr,3,[(.8,1.1,.6,.2),(1.9,2.2,.2,.2)])
        result=acoustic_boundaries(x,sr,[row(.8,1.1),row(1.9,2.2)])
        self.assertLessEqual(result[0]['start'],.8)
        self.assertGreater(result[0]['end'],1.4)
        self.assertLess(result[0]['end'],1.9)
        self.assertLessEqual(result[0]['end'],result[1]['start'])

    def test_short_voice_and_neighbor_are_measured_jointly(self):
        sr=44100;x=recording(sr,2,[(.6,.78,.37,.25),(1.1,1.5,.15,.2)])
        result=acoustic_boundaries(x,sr,[row(.6,.78),row(1.1,1.4)])
        self.assertGreater(result[0]['end'],1.)
        self.assertLessEqual(result[0]['end'],1.11)
        self.assertLessEqual(result[0]['end'],result[1]['start'])

    def test_time_translation_gain_and_sample_rate_do_not_change_policy(self):
        for sr in [16000,44100,48000]:
            for gain in [.02,.2,.8]:
                with self.subTest(sr=sr,gain=gain):
                    for offset in [0.,3.17,17.43]:
                        x=recording(sr,offset+3,[(offset+.6,offset+.9,.35,gain),(offset+1.5,offset+1.9,.2,gain)])
                        result=acoustic_boundaries(x,sr,[row(offset+.6,offset+.9),row(offset+1.5,offset+1.9)])
                        self.assertLess(abs(result[0]['start']-offset-.56),.035)
                        self.assertGreater(result[0]['end'],offset+1.08)
                        self.assertLessEqual(result[0]['end_frame'],result[1]['start_frame'])

    def test_first_syllable_before_internal_word_pause_is_preserved(self):
        sr=44100;x=recording(sr,3,[(.5,.7,0,.2),(1.3,1.7,.1,.2)])
        result=acoustic_boundaries(x,sr,[row(.5,1.7)])
        self.assertLessEqual(result[0]['start'],.5)
        self.assertGreater(result[0]['end'],1.7)

    def test_soft_attack_before_loud_vowel_is_preserved(self):
        sr=44100;x=recording(sr,2,[(.5,.65,0,.01),(.65,.9,.15,.2)])
        result=acoustic_boundaries(x,sr,[row(.5,.9)])
        self.assertLessEqual(result[0]['start'],.51)

    def test_silence_and_missing_alignment_error_without_old_time_fallback(self):
        with self.assertRaises(RuntimeError):acoustic_boundaries(np.zeros(44100,dtype=np.float32),44100,[row(.2,.4)])
        with self.assertRaises(RuntimeError):acoustic_boundaries(np.zeros(44100,dtype=np.float32),44100,[dict(words=[])])

    def test_recording_edge_is_clipped_to_real_frames(self):
        sr=44100;x=recording(sr,1,[(.1,.95,.5,.2)])
        result=acoustic_boundaries(x,sr,[row(.1,.95)])
        self.assertEqual(result[-1]['end_frame'],len(x))

    def test_provided_word_timeline_is_used_without_another_aligner(self):
        sr=44100;x=recording(sr,4,[(.5,.8,.2,.2)])
        primary=[row(.5,.8,.1)]
        result=acoustic_boundaries(x,sr,primary)
        self.assertLess(result[0]['end'],1.3)
        self.assertEqual(result[0]['alignment_backend'],'provided_word_timestamps')

    def test_global_shift_missing_final_word_is_recovered_from_provider_and_waveform(self):
        from reference_clean import reference_windows
        for sr in [16000,44100,48000]:
            for offset in [0.,3.17]:
                with self.subTest(sr=sr,offset=offset):
                    x=recording(sr,offset+3,[(offset+.5,offset+1.3,.05,.2),
                        (offset+1.8,offset+2.05,.1,.2)])
                    words=[word(offset+.5,offset+1.3),
                        {**word(offset+1.57,offset+1.59),'provider_start':offset+1.8,'provider_end':offset+1.82}]
                    measured=acoustic_boundaries(x,sr,[dict(words=words)])
                    cleaned=reference_windows(x,sr,measured)
                    self.assertEqual(measured[0]['end_anchor_recovery'],'provider_anchor_with_acoustic_evidence')
                    self.assertAlmostEqual(measured[0]['aligned_end'],offset+1.82)
                    self.assertGreater(cleaned[0]['end'],offset+2.05)

    def test_provider_anchor_in_silence_does_not_bypass_acoustic_measurement(self):
        sr=16000;x=recording(sr,3,[(.5,.9,0.,.2)])
        words=[word(.5,.9),{**word(1.4,1.42),'provider_start':1.8,'provider_end':1.82}]
        with self.assertRaisesRegex(RuntimeError,'마지막 발성'):
            acoustic_boundaries(x,sr,[dict(words=words)])

    def test_global_shift_missing_first_word_is_remeasured_near_provider_anchor(self):
        for sr in [16000,44100,48000]:
            for offset in [0.,3.17,17.43]:
                with self.subTest(sr=sr,offset=offset):
                    x=recording(sr,offset+3,[(offset+.9,offset+1.05,0,.2),
                        (offset+1.35,offset+1.7,0,.2),(offset+2.1,offset+2.4,0,.2)])
                    first={**word(offset+.6,offset+.65),'provider_start':offset+.9,'provider_end':offset+.95}
                    measured=acoustic_boundaries(x,sr,[dict(words=[first,word(offset+1.35,offset+1.7)]),row(offset+2.1,offset+2.4)])
                    self.assertEqual(measured[0]['start_anchor_recovery'],'provider_anchor_with_acoustic_evidence')
                    self.assertAlmostEqual(measured[0]['aligned_start'],offset+.9)
                    self.assertLessEqual(measured[0]['start'],offset+.91)
                    self.assertGreater(measured[0]['start'],offset+.8)
                    self.assertLessEqual(measured[0]['end_frame'],measured[1]['start_frame'])

    def test_onset_recovery_rejects_silent_anchor_and_does_not_take_next_voice(self):
        sr=16000;x=recording(sr,3,[(1.35,1.7,0,.2),(2.1,2.4,0,.2)])
        first={**word(.6,.65),'provider_start':.9,'provider_end':.95}
        with self.assertRaisesRegex(RuntimeError,'발성 시작'):
            acoustic_boundaries(x,sr,[dict(words=[first,word(1.35,1.7)]),row(2.1,2.4)])

    def test_quiet_gap_after_short_first_syllable_does_not_erase_that_syllable(self):
        for sr in [16000,44100,48000]:
            with self.subTest(sr=sr):
                x=recording(sr,3,[(.5,.9,.8,.2),(1.35,1.43,0,.2),(1.8,2.2,0,.2)])
                first={**word(1.272,1.352),'provider_start':1.394,'provider_end':1.474}
                measured=acoustic_boundaries(x,sr,[row(.5,.9),dict(words=[first,word(1.8,2.2)])])
                self.assertEqual(measured[1]['start_anchor_recovery'],'provider_anchor_with_acoustic_evidence')
                self.assertLess(measured[1]['start'],1.43)
                self.assertLessEqual(measured[0]['end_frame'],measured[1]['start_frame'])

    def test_quiet_first_word_recovery_requires_local_contrast(self):
        for gain in [.02,.2,.8]:
            with self.subTest(gain=gain):
                sr=16000;x=recording(sr,3,[(.9,1.0,0,gain*.04),(1.35,1.7,0,gain)])
                first={**word(.6,.65),'provider_start':.9,'provider_end':.95}
                measured=acoustic_boundaries(x,sr,[dict(words=[first,word(1.35,1.7)])])
                self.assertEqual(measured[0]['start_anchor_recovery'],'local_first_word_with_acoustic_evidence')
                self.assertLessEqual(measured[0]['start'],.91)

    def test_onset_recovery_rejects_steady_low_background(self):
        sr=16000;x=recording(sr,3,[(.75,1.2,0,.005),(1.35,1.7,0,.2)])
        first={**word(.3,.35),'provider_start':.9,'provider_end':.95}
        with self.assertRaisesRegex(RuntimeError,'발성 시작'):
            acoustic_boundaries(x,sr,[dict(words=[first,word(1.35,1.7)])])

    def test_quiet_final_syllable_uses_local_contrast_not_phrase_peak(self):
        for sr in [16000,44100,48000]:
            for gain in [.02,.2,.8]:
                with self.subTest(sr=sr,gain=gain):
                    x=recording(sr,3,[(.5,1.2,0.,gain),(1.8,1.95,.05,gain*.04)])
                    words=[word(.5,1.2),word(1.8,1.95)]
                    result=acoustic_boundaries(x,sr,[dict(words=words)])
                    self.assertEqual(result[0]['end_anchor_recovery'],'local_final_word_with_acoustic_evidence')
                    self.assertGreaterEqual(result[0]['end'],1.95)

    def test_steady_background_at_final_word_is_not_accepted_as_a_syllable(self):
        sr=16000;x=recording(sr,3,[(.5,.9,0.,.2),(1.5,2.2,0.,.005)])
        with self.assertRaisesRegex(RuntimeError,'마지막 발성'):
            acoustic_boundaries(x,sr,[dict(words=[word(.5,.9),word(1.8,1.9)])])

if __name__=='__main__':unittest.main()
