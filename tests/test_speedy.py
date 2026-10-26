from pathlib import Path
import tempfile,unittest,sys
from unittest.mock import patch
import numpy as np,soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import SR
import speedy,timing

class SpeedyTests(unittest.TestCase):
    def test_real_speedy_never_overruns_and_preserves_stereo_pitch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);t=np.arange(2*SR)/SR
            values=np.column_stack([.12*np.sin(2*np.pi*220*t),.09*np.sin(2*np.pi*330*t)]).astype('float32')
            source=root/'input.wav';sf.write(source,values,SR,subtype='FLOAT')
            for speed in [1.25,1.5]:
                target=round(len(values)/speed);out=root/'out.wav'
                actual=speedy.stretch(source,out,target)
                self.assertLessEqual(len(actual),target)
                self.assertLessEqual(target-len(actual),speedy.MAX_LENGTH_ERROR_FRAMES)
                self.assertEqual(actual.shape[1],2)
                self.assertEqual(sf.info(out).subtype,'FLOAT')
                for channel,hz in [(0,220),(1,330)]:
                    x=actual[SR//5:-SR//5,channel]
                    spectrum=np.abs(np.fft.rfft(x*np.hanning(len(x))))
                    observed=np.fft.rfftfreq(len(x),1/SR)[spectrum.argmax()]
                    self.assertAlmostEqual(observed,hz,delta=3)

    def test_real_collision_retains_speech_pitch_and_fits_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);t=np.arange(2*SR)/SR
            body=np.repeat((.1*np.sin(2*np.pi*220*t))[:,None],2,axis=1).astype('float32')
            values=np.concatenate([np.zeros((SR//10,2)),body,np.zeros((SR//5,2))]).astype('float32')
            sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
            timing.pause_only_audio(root/'tts.wav',root/'pause.wav',SR,preserve_outer_edges=True)
            paused,_=sf.read(root/'pause.wav',dtype='float32',always_2d=True)
            lo,hi=timing.speech_bounds(paused,relative=.02)
            report=timing.bounded_timing_audio(root/'tts.wav',root/'output.wav',SR,collision_budget_frames=round(1.5*SR),preserve_outer_edges=True)
            result,_=sf.read(root/'output.wav',dtype='float32',always_2d=True)
            self.assertLessEqual(len(result),round(1.5*SR))
            self.assertEqual(report['tempo_engine'],'google_speedy')
            self.assertEqual(report['overlap_after_speedup_frames'],0)
            # Current policy may shorten quiet outer padding to avoid collisions.
            # The voiced signal must remain finite and retain its original pitch.
            self.assertTrue(np.isfinite(result).all())
            x=result[:,0];spectrum=np.abs(np.fft.rfft(x*np.hanning(len(x))))
            pitch=np.fft.rfftfreq(len(x),1/SR)[spectrum.argmax()]
            self.assertAlmostEqual(pitch,220,delta=3)

    def test_failed_worker_preserves_previous_file_without_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.wav';out=root/'output.wav'
            sf.write(source,np.ones((SR,2),dtype='float32')*.1,SR,subtype='FLOAT')
            out.write_bytes(b'previous')
            with patch('speedy.subprocess.run',side_effect=RuntimeError('worker failed')):
                with self.assertRaises(RuntimeError):speedy.stretch(source,out,SR//2)
            self.assertEqual(out.read_bytes(),b'previous')

if __name__=='__main__':unittest.main()
