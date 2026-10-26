from pathlib import Path
from unittest.mock import patch
import json,sys,tempfile,unittest
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import SR
import timing,render

def tone(seconds):
    t=np.arange(round(seconds*SR))/SR
    return np.repeat((.1*np.sin(2*np.pi*220*t))[:,None],2,axis=1).astype('float32')

class CollisionTimingTests(unittest.TestCase):
    def setUp(self):
        def backend(source,dest,frames):
            values=tone(frames/SR);sf.write(dest,values,SR,subtype='FLOAT');return values
        self.backend=patch('timing.collision_stretch',side_effect=backend)
        self.backend.start();self.addCleanup(self.backend.stop)
        self.r3=patch('timing.stretch',side_effect=AssertionError('No R3 fallback'))
        self.r3.start();self.addCleanup(self.r3.stop)

    def test_no_slowdown_and_no_speedup_when_gap_or_last_line_allows_overrun(self):
        for source_seconds,reference_seconds,budget in [(1.,3.,3*SR),(2.,1.,3*SR),(3.,1.,None),(2.,1.,2*SR)]:
            with self.subTest(source=source_seconds,reference=reference_seconds,budget=budget),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);values=tone(source_seconds)
                sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
                sf.write(root/'ref.wav',tone(reference_seconds),SR,subtype='FLOAT')
                with patch('timing.collision_stretch',side_effect=AssertionError('No collision: do not change speed')):
                    report=timing.bounded_timing_audio(root/'tts.wav',root/'out.wav',round(reference_seconds*SR),root/'ref.wav',collision_budget_frames=budget)
                np.testing.assert_array_equal(sf.read(root/'out.wav',dtype='float32',always_2d=True)[0],values)
                self.assertEqual(report['tempo'],1.)
                self.assertEqual(report['speed_correction_trigger'],'none')

    def test_speedup_only_removes_the_overlap_and_keeps_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);values=np.concatenate([np.zeros((SR//10,2)),tone(2),np.zeros((SR//5,2))]).astype('float32')
            sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
            report=timing.bounded_timing_audio(root/'tts.wav',root/'out.wav',SR,collision_budget_frames=round(1.5*SR))
            actual=sf.read(root/'out.wav',dtype='float32',always_2d=True)[0]
            self.assertEqual(len(actual),round(1.5*SR))
            self.assertGreater(report['tempo'],1.)
            self.assertEqual(report['tempo_engine'],'google_speedy')
            self.assertEqual(report['overlap_after_speedup_frames'],0)
            np.testing.assert_array_equal(actual[:SR//20],values[:SR//20])
            np.testing.assert_array_equal(actual[-SR//10:],values[-SR//10:])
            json.dumps(report)

    def test_pause_compression_resolves_collision_before_r3(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);values=np.concatenate([tone(.4),np.zeros((round(.6*SR),2)),tone(.4)]).astype('float32')
            sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
            hint={'status':'aligned','phones':[],'words':[{'start':0.,'end':.4,'label':'앞말'},{'start':1.,'end':1.4,'label':'뒷말'}]}
            with patch('timing.alignment_hint',return_value=hint),patch('timing.collision_stretch',side_effect=AssertionError('Pauses suffice')):
                report=timing.bounded_timing_audio(root/'tts.wav',root/'out.wav',SR,text='앞말 뒷말',collision_budget_frames=SR)
            self.assertEqual(sf.info(root/'out.wav').frames,SR)
            self.assertGreater(report['pause_removed_seconds'],.39)
            self.assertEqual(report['pause_adjustment_share'],1.)
            self.assertEqual(report['tempo'],1.)

    def test_failure_preserves_previous_output_and_extreme_edges_are_not_cut(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);values=np.concatenate([np.zeros((SR,2)),tone(.5),np.zeros((SR,2))]).astype('float32')
            sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
            destination=root/'out.wav';destination.write_bytes(b'previous')
            with patch('timing.collision_stretch',side_effect=RuntimeError('R3 failed')):
                with self.assertRaises(RuntimeError):timing.bounded_timing_audio(root/'tts.wav',destination,SR,collision_budget_frames=SR)
            self.assertEqual(destination.read_bytes(),b'previous')
            report=timing.bounded_timing_audio(root/'tts.wav',destination,SR,collision_budget_frames=SR)
            self.assertEqual(sf.info(destination).frames,SR)
            self.assertEqual(report['speed_correction_region'],'full_audio')

    def test_next_start_change_invalidates_only_fitting_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sf.write(root/'clean.wav',tone(3),SR,subtype='FLOAT')
            sf.write(root/'0000-tts.mp3',tone(2),SR,format='WAV',subtype='FLOAT')
            policy=render.tts_policy('ko')
            (root/'0000-generation.json').write_text(json.dumps({'generation_id':'cached','tts_policy':policy}))
            segment={'lo':0,'hi':SR,'start':0.,'end':1.,'original_text':'original','translated_text':'translated','next_dialogue_start_frame':3*SR}
            cfg={'language':'ko','key':'unused'}
            with patch('render.request_audio',side_effect=AssertionError('Reuse TTS')),patch('timing.alignment_hint',return_value={'status':'unavailable'}):
                before=render.synthesize(0,segment,root/'clean.wav',root,cfg)
                self.assertEqual(before['tempo'],1.)
                segment['next_dialogue_start_frame']=SR
                after=render.synthesize(0,segment,root/'clean.wav',root,cfg)
                self.assertEqual(after['generation_id'],'cached')
                self.assertEqual(after['collision_budget_frames'],SR)
                self.assertEqual(after['output_frames'],SR)

if __name__=='__main__':unittest.main()
