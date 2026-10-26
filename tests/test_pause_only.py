from pathlib import Path
from unittest.mock import patch
import sys,tempfile,unittest
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import SR
import timing

def tone(seconds):
    t=np.arange(round(seconds*SR))/SR
    return np.repeat((.15*np.sin(2*np.pi*220*t))[:,None],2,axis=1).astype('float32')

class PauseOnlyTests(unittest.TestCase):
    def setUp(self):
        self.hint=patch('timing.alignment_hint',return_value={'status':'aligned','phones':[],
            'words':[{'start':0.,'end':.4,'label':'앞말'},{'start':1.,'end':1.4,'label':'뒷말'}]})
        self.mfa=self.hint.start();self.addCleanup(self.hint.stop)
        self.speed=patch('timing.stretch',side_effect=AssertionError('Never change speech speed'))
        self.r3=self.speed.start();self.addCleanup(self.speed.stop)

    def body(self):return np.concatenate([tone(.4),np.zeros((round(.6*SR),2),dtype='float32'),tone(.4)])

    def test_safe_pause_expands_and_compresses_without_changing_voiced_samples(self):
        for seconds in [1.,1.6]:
            with self.subTest(target=seconds),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);values=self.body()
                sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
                sf.write(root/'reference.wav',tone(seconds),SR,subtype='FLOAT')
                result=timing.pause_only_audio(root/'tts.wav',root/'out.wav',round(seconds*SR),root/'reference.wav','앞말 뒷말')
                actual=sf.read(root/'out.wav',dtype='float32',always_2d=True)[0]
                self.assertEqual(len(actual),round(seconds*SR))
                np.testing.assert_array_equal(actual[:round(.4*SR)],values[:round(.4*SR)])
                np.testing.assert_array_equal(actual[-round(.4*SR):],values[-round(.4*SR):])
                self.assertTrue(result['pause_adjustments']);self.assertEqual(result['tempo'],1)
                self.assertEqual(result['tempo_engine'],'none');self.r3.assert_not_called()

    def test_unresolved_long_and_short_audio_is_neither_stretched_padded_nor_truncated(self):
        for seconds,expected in [(.5,.88),(3.,1.7)]:
            with self.subTest(target=seconds),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);values=self.body()
                sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
                sf.write(root/'reference.wav',tone(seconds),SR,subtype='FLOAT')
                result=timing.pause_only_audio(root/'tts.wav',root/'out.wav',round(seconds*SR),root/'reference.wav','앞말 뒷말')
                actual=sf.read(root/'out.wav',dtype='float32',always_2d=True)[0]
                self.assertEqual(len(actual),round(expected*SR))
                self.assertNotEqual(len(actual),round(seconds*SR))
                self.assertAlmostEqual(result['remaining_duration_error_seconds'],expected-seconds,places=5)
                self.assertEqual(result['tempo'],1);self.r3.assert_not_called()
                np.testing.assert_array_equal(actual[:round(.4*SR)],values[:round(.4*SR)])
                np.testing.assert_array_equal(actual[-round(.4*SR):],values[-round(.4*SR):])

    def test_failed_alignment_still_trims_only_external_silence(self):
        self.mfa.return_value={'status':'failed','reason':'no reliable boundary'}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);values=np.concatenate([np.zeros((SR//5,2),dtype='float32'),self.body(),np.zeros((SR//3,2),dtype='float32')])
            sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
            result=timing.pause_only_audio(root/'tts.wav',root/'out.wav',SR,text='앞말 뒷말')
            actual=sf.read(root/'out.wav',dtype='float32',always_2d=True)[0]
            expected,head,tail=timing.trim_outer_silence(values)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(result['tts_samples_trimmed'],head+tail)
            self.assertGreater(head,0);self.assertGreater(tail,0)
            self.assertEqual(result['pause_adjustments'],[]);self.assertEqual(result['mfa_status'],'failed')
            self.r3.assert_not_called()

    def test_quiet_consonant_closure_is_not_split_to_fill_a_short_clip(self):
        self.mfa.return_value={'status':'aligned','words':[{'start':0.,'end':.4,'label':'앞말'},
            {'start':.4,'end':.9,'label':'뒷말'}],'phones':[{'start':0.,'end':.4,'label':'a'},
            {'start':.4,'end':.46,'label':'tʷ'},{'start':.46,'end':.9,'label':'a'}]}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);values=tone(.9);values[round(.42*SR):round(.44*SR)]=0
            sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
            result=timing.pause_only_audio(root/'tts.wav',root/'out.wav',2*SR,text='앞말 뒷말')
            actual=sf.read(root/'out.wav',dtype='float32',always_2d=True)[0]
            np.testing.assert_array_equal(actual,values)
            self.assertEqual(result['new_pause_count'],0);self.assertEqual(result['tempo_engine'],'none')

    def test_existing_pause_capacity_precedes_additional_verified_word_junctions(self):
        # A normal pause and a separate, shorter but corroborated word gap.
        values=np.concatenate([tone(.4),np.zeros((round(.2*SR),2)),tone(.4),np.zeros((round(.06*SR),2)),tone(.4)]).astype('float32')
        self.mfa.return_value={'status':'aligned','words':[
            {'start':0.,'end':.4,'label':'앞말'}, {'start':.6,'end':1.,'label':'중간말'},
            {'start':1.06,'end':1.46,'label':'뒷말'}], 'phones':[
            {'start':0.,'end':.4,'label':'a'},{'start':.4,'end':.6,'label':'sil'},
            {'start':.6,'end':1.,'label':'a'},{'start':1.,'end':1.06,'label':'sil'},
            {'start':1.06,'end':1.46,'label':'a'}]}
        for target in [1.51,3.]:
            with self.subTest(target=target),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
                report=timing.pause_only_audio(root/'tts.wav',root/'out.wav',round(target*SR),text='앞말 중간말 뒷말')
                actual=sf.read(root/'out.wav',dtype='float32',always_2d=True)[0]
                np.testing.assert_array_equal(actual[:round(.4*SR)],values[:round(.4*SR)])
                np.testing.assert_array_equal(actual[-round(.4*SR):],values[-round(.4*SR):])
                if target==1.51:
                    self.assertEqual(report['new_pause_count'],0)
                    self.assertEqual(len(actual),round(target*SR))
                else:
                    self.assertGreater(report['new_pause_count'],0)
                    self.assertGreater(report['pause_added_seconds'],.1)
                    self.assertLess(len(actual),round(target*SR))
                self.r3.assert_not_called()

if __name__=='__main__':unittest.main()
