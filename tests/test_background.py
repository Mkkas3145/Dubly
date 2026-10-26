from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import SR
from background import prepare_background
import render


class BackgroundTests(unittest.TestCase):
    def test_source_voice_stays_removed_until_extended_tts_ends_then_fades(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);frames=4*SR
            original=np.full((frames,2),.3,dtype='float32')
            voice=np.full_like(original,.1);bed=np.full_like(original,[.02,-.01])
            for name,data in [('original',original),('clean',voice),('background',bed)]:
                sf.write(root/f'{name}.wav',data,SR,subtype='FLOAT')
            segment={'lo':SR,'hi':round(1.2*SR)}
            tts=np.full((SR,2),[.04,-.02],dtype='float32')
            sf.write(root/'0000-fitted.wav',tts,SR,subtype='FLOAT')
            with patch.object(render,'voice_gain',return_value=1):
                render.mix(root/'original.wav',root/'clean.wav',[segment],root,root/'mixed.wav',lambda *a:None,background=root/'background.wav')
            actual=sf.read(root/'mixed.wav',dtype='float32',always_2d=True)[0]
            np.testing.assert_allclose(actual[SR:2*SR],bed[SR:2*SR]+tts,atol=1e-7)
            np.testing.assert_allclose(actual[2*SR:round(2.2*SR)],bed[2*SR:round(2.2*SR)],atol=1e-7)
            midpoint=round(2.3*SR)
            np.testing.assert_allclose(actual[midpoint],(original[midpoint]+bed[midpoint])*.5,atol=1e-6)
            np.testing.assert_array_equal(actual[round(2.5*SR):],original[round(2.5*SR):])

    def test_coverage_survives_overlapping_extended_tails_and_leaves_inputs_unchanged(self):
        segments=[{'lo':SR,'hi':2*SR},{'lo':3*SR,'hi':4*SR},{'lo':5*SR,'hi':6*SR}]
        placements=[{'lo':SR,'hi':round(5.5*SR)},{'lo':3*SR,'hi':round(6.5*SR)},{'lo':5*SR,'hi':8*SR}]
        before=[dict(s) for s in segments]
        intervals=render.replacement_intervals(segments,7*SR,placements)
        weights=render.transition_weights(0,8*SR,intervals)
        for p in placements:np.testing.assert_array_equal(weights[p['lo']:p['hi']],1)
        self.assertLessEqual(float(weights.max()),1.)
        self.assertEqual(segments,before)

    def test_direct_background_preserves_stereo_tts_and_original_outside_overlapping_fades(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            wave=np.random.default_rng(42).uniform(-.04,.04,(12*SR,2)).astype('float32')
            vocals=np.full_like(wave,.02)
            bed=np.full_like(wave,[.003,-.007])
            for name,data in [('original',wave),('clean',vocals),('background',bed)]:
                sf.write(root/f'{name}.wav',data,SR,subtype='FLOAT')
            segments=[{'lo':round(a*SR),'hi':round(b*SR)} for a,b in [(9.8,10),(10.2,10.4)]]
            translated=np.zeros_like(wave)
            for n,s in enumerate(segments):
                generated=np.full((s['hi']-s['lo'],2),[.005,-.002],dtype='float32')
                sf.write(root/f'{n:04d}-fitted.wav',generated,SR,subtype='FLOAT')
                translated[s['lo']:s['hi']]=generated
            with patch.object(render,'voice_gain',return_value=1):
                render.mix(root/'original.wav',root/'clean.wav',segments,root,root/'mixed.wav',lambda *args:None,background=root/'background.wav')
            actual,_=sf.read(root/'mixed.wav',always_2d=True,dtype='float32')
            weights=render.transition_weights(0,len(wave),render.replacement_intervals(segments,len(wave)))
            expected=wave*(1-weights[:,None])+bed*weights[:,None]+translated
            np.testing.assert_allclose(actual,expected,atol=1e-7)
            np.testing.assert_array_equal(actual[weights==0],wave[weights==0])
            for s in segments:np.testing.assert_allclose(actual[s['lo']:s['hi']]-bed[s['lo']:s['hi']],translated[s['lo']:s['hi']],atol=1e-8)

    def test_invalid_background_does_not_replace_existing_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,rate,channels in [('original',SR,2),('clean',SR,2),('background',48000,1)]:
                sf.write(root/f'{name}.wav',np.zeros((100,channels)),rate,subtype='FLOAT')
            (root/'mixed.wav').write_bytes(b'previous')
            with self.assertRaises(RuntimeError):
                render.mix(root/'original.wav',root/'clean.wav',[],root,root/'mixed.wav',lambda *args:None,background=root/'background.wav')
            self.assertEqual((root/'mixed.wav').read_bytes(),b'previous')

    def test_background_cache_is_invalidated_by_source_and_wrong_frame_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'original.wav'
            sf.write(source,np.zeros((100,2)),SR,subtype='FLOAT')
            def infer(source,destination,progress,**kwargs):
                self.assertEqual(kwargs,{'instrumental':True})
                destination.parent.mkdir(parents=True,exist_ok=True)
                sf.write(destination,np.zeros((sf.info(source).frames,2)),SR,subtype='FLOAT')
            with patch('hyperace.separate',side_effect=infer) as model:
                output=prepare_background(source,root,lambda *args:None,model_name='hyperace-v2-instrumental')
                prepare_background(source,root,lambda *args:None,model_name='hyperace-v2-instrumental')
                self.assertEqual(model.call_count,1)
                sf.write(output,np.zeros((99,2)),SR,subtype='FLOAT')
                prepare_background(source,root,lambda *args:None,model_name='hyperace-v2-instrumental')
                self.assertEqual(model.call_count,2)
                sf.write(source,np.ones((100,2))*.01,SR,subtype='FLOAT')
                prepare_background(source,root,lambda *args:None,model_name='hyperace-v2-instrumental')
                self.assertEqual(model.call_count,3)

    def test_deux_uses_the_second_instrumental_stem_and_preserves_stereo(self):
        from deux import select_instrumental
        prediction=np.zeros((1,2,2,123),dtype='float32')
        prediction[0,0]=.5
        prediction[0,1,0]=.03;prediction[0,1,1]=-.02
        np.testing.assert_array_equal(select_instrumental(prediction,123),prediction[0,1])
        with self.assertRaises(RuntimeError):select_instrumental(prediction,124)
        with self.assertRaises(RuntimeError):select_instrumental(prediction[:,0],123)

    def test_deux_background_does_not_reuse_hyperace_cache(self):
        from core import DEFAULT_BACKGROUND
        self.assertEqual(DEFAULT_BACKGROUND,'mel-band-roformer-deux')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'original.wav'
            sf.write(source,np.zeros((100,2)),SR,subtype='FLOAT')
            old=root/'runs/hyperace-v2-instrumental-v1/background.wav'
            old.parent.mkdir(parents=True);sf.write(old,np.ones((100,2))*.2,SR,subtype='FLOAT')
            def infer(source,destination,progress):
                destination.parent.mkdir(parents=True,exist_ok=True)
                sf.write(destination,np.zeros((100,2)),SR,subtype='FLOAT')
            with patch('deux.separate',side_effect=infer) as model:
                output=prepare_background(source,root,lambda *args:None,model_name='mel-band-roformer-deux')
                self.assertNotEqual(output,old);self.assertEqual(model.call_count,1)
                prepare_background(source,root,lambda *args:None,model_name='mel-band-roformer-deux')
                self.assertEqual(model.call_count,1)
                metadata=json.loads(output.with_suffix('.json').read_text())
                self.assertEqual(metadata['overlap'],2)
