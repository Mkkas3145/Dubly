from pathlib import Path
from unittest.mock import patch
import sys,tempfile,unittest
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import SR,command,probe
import render,timing


class NaturalAudioTests(unittest.TestCase):
    def test_all_decoded_samples_and_silences_survive_shorter_and_longer_source_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            values=np.zeros((round(1.8*SR),2),dtype=np.float32)
            t=np.arange(round(.7*SR))/SR
            values[round(.2*SR):round(.9*SR)]=(.1*np.sin(2*np.pi*220*t))[:,None]
            values[SR:round(1.5*SR)]=.03
            sf.write(root/'tts.wav',values,SR,subtype='FLOAT')
            for frames in [SR//2,3*SR]:
                with (patch('timing.stretch',side_effect=AssertionError('No speed changes')),
                      patch('timing.adjust_pauses',side_effect=AssertionError('No pause changes')),
                      patch('timing.alignment_hint',side_effect=AssertionError('No MFA'))):
                    result=timing.natural_audio(root/'tts.wav',root/'natural.wav',frames)
                actual=sf.read(root/'natural.wav',dtype='float32',always_2d=True)[0]
                np.testing.assert_array_equal(actual,values)
                self.assertEqual(result['tempo'],1)
                self.assertFalse(result['pause_adjustment_enabled'])
                self.assertEqual(result['tts_samples_trimmed'],0)

    def test_overlap_survives_mix_block_boundaries_and_the_video_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);frames=12*SR
            original=np.full((frames,2),.3,dtype=np.float32)
            clean=np.full_like(original,.1)
            sf.write(root/'original.wav',original,SR,subtype='FLOAT')
            sf.write(root/'clean.wav',clean,SR,subtype='FLOAT')
            segments=[{'lo':round(9.8*SR),'hi':10*SR},{'lo':round(10.2*SR),'hi':round(10.4*SR)},
                      {'lo':round(11.8*SR),'hi':frames}]
            audio=[np.full((2*SR,2),-.05,dtype=np.float32),np.full((SR,2),.025,dtype=np.float32),
                   np.full((SR,2),-.05,dtype=np.float32)]
            end=round(12.8*SR);expected_tts=np.zeros((end,2),dtype=np.float32)
            for i,(s,v) in enumerate(zip(segments,audio)):
                sf.write(root/f'{i:04d}-fitted.wav',v,SR,subtype='FLOAT')
                gain=render.voice_gain(clean[s['lo']:s['hi']],v)
                expected_tts[s['lo']:s['lo']+len(v)]+=v*gain
            render.mix(root/'original.wav',root/'clean.wav',segments,root,root/'mixed.wav',lambda *args:None)
            actual=sf.read(root/'mixed.wav',dtype='float32',always_2d=True)[0]
            original=np.pad(original,((0,end-frames),(0,0)))
            clean=np.pad(clean,((0,end-frames),(0,0)))
            weights=render.transition_weights(0,end,render.replacement_intervals(segments,frames,render.speech_placements(segments,root)))
            expected=np.clip(original-clean*weights[:,None]+expected_tts,-1,1)
            np.testing.assert_allclose(actual,expected,atol=1e-7)
            self.assertEqual(len(actual),end)
            self.assertTrue(np.any(actual[frames:]))
            np.testing.assert_array_equal(actual[:9*SR],original[:9*SR])

    def test_mux_keeps_the_complete_audio_tail_without_reencoding_video(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            command(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','color=black:s=32x32:r=10:d=1',
                '-an','-c:v','libx264',str(root/'source.mp4')])
            sf.write(root/'audio.wav',np.full((2*SR,2),.02,dtype=np.float32),SR,subtype='FLOAT')
            render.mux(root/'source.mp4',root/'audio.wav',root/'output.mp4','ko',1.)
            result=probe(root/'output.mp4')
            # AAC packet padding can add at most one 1024-sample frame.
            self.assertAlmostEqual(float(result['format']['duration']),2.,delta=1024/SR+.001)
            def video_hash(path):return command(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:v:0',
                '-c','copy','-f','hash','-hash','sha256','-'])
            self.assertEqual(video_hash(root/'source.mp4'),video_hash(root/'output.mp4'))

    def test_decode_failure_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'output.wav';out.write_bytes(b'previous')
            with patch('timing.command',side_effect=RuntimeError('decode failed')):
                with self.assertRaises(RuntimeError):timing.natural_audio(root/'missing',out,SR)
            self.assertEqual(out.read_bytes(),b'previous')


if __name__=='__main__':unittest.main()
