from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from separation_audio import overlap_add,overlap_add_stems
from separation import cache_directory,separate
from core import DEFAULT_SEPARATOR,config
import hyperace
from speech import Transcript,normalize


class SeparationTests(unittest.TestCase):
    def test_overlap_add_preserves_exact_samples_length_and_stereo_at_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            for length in [1,7,64,69,1001]:
                with self.subTest(length=length):
                    wave=np.random.default_rng(length).uniform(-.5,.5,(length,2)).astype(np.float32)
                    wave[:,1]*=-.25
                    sf.write(folder/'source.wav',wave,48000,subtype='FLOAT')
                    progress=[]
                    overlap_add(folder/'source.wav',folder/'result.wav',lambda samples:samples,lambda p,d:progress.append(p),chunk=64,hop=8)
                    result,rate=sf.read(folder/'result.wav',always_2d=True,dtype='float32')
                    self.assertEqual(rate,48000);self.assertEqual(result.shape,wave.shape)
                    np.testing.assert_allclose(result,wave,atol=1e-7)
                    self.assertEqual(progress[-1],100)

    def test_nonfinite_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);sf.write(folder/'source.wav',np.zeros((80,2)),48000,subtype='FLOAT')
            with self.assertRaises(RuntimeError):
                overlap_add(folder/'source.wav',folder/'result.wav',lambda samples:np.full_like(samples,np.nan),lambda *args:None,chunk=64,hop=8)

    def test_model_caches_cannot_mix_and_model_name_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            kim=cache_directory(root,'melband-kim');bandit=cache_directory(root,'bandit-v2-multi')
            selected=cache_directory(root,'hyperace-v2-vocals')
            (kim/'segments.json').write_text('old transcript')
            self.assertFalse((bandit/'segments.json').exists())
            self.assertFalse((selected/'segments.json').exists())
            with self.assertRaises(RuntimeError):cache_directory(root,'../../outside')

    def test_default_dispatch_uses_selected_vocals_model(self):
        self.assertEqual(DEFAULT_SEPARATOR,'mel-band-roformer-deux')
        with patch('core.dotenv_values',return_value={}):self.assertEqual(config()['separator'],DEFAULT_SEPARATOR)
        with patch('deux.separate') as selected:
            separate('source','destination','progress')
            selected.assert_called_once_with('source','destination','progress',stem='Vocals')

    def test_two_stems_share_each_inference_and_keep_exact_stereo_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            for length in [1,7,69,1001]:
                wave=np.random.default_rng(length).uniform(-.1,.1,(length,2)).astype('float32')
                sf.write(folder/'source.wav',wave,44100,subtype='FLOAT')
                calls=[]
                def infer(samples):
                    calls.append(1)
                    return np.stack([samples*.25,samples*.75])
                overlap_add_stems(folder/'source.wav',[folder/'voice.wav',folder/'bg.wav'],infer,lambda *a:None,chunk=64,hop=32)
                self.assertEqual(len(calls),len(range(-32,length,32)))
                for path,scale in [('voice',.25),('bg',.75)]:
                    result,rate=sf.read(folder/f'{path}.wav',always_2d=True,dtype='float32')
                    self.assertEqual(rate,44100);self.assertEqual(result.shape,wave.shape)
                    np.testing.assert_allclose(result,wave*scale,atol=1e-7)

    def test_pair_cache_rejects_missing_or_modified_stem(self):
        from deux import prepare_pair
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.wav';folder=root/'deux'
            sf.write(source,np.zeros((100,2)),44100,subtype='FLOAT')
            def infer(source,destinations,progress):
                self.assertEqual(list(destinations),['Vocals','Instrumental'])
                folder.mkdir(exist_ok=True)
                for path in destinations.values():sf.write(path,np.zeros((100,2)),44100,subtype='FLOAT')
            with patch('deux.separate_stems',side_effect=infer) as model:
                voice,bg=prepare_pair(source,folder,lambda *a:None)
                prepare_pair(source,folder,lambda *a:None);self.assertEqual(model.call_count,1)
                sf.write(voice,np.ones((100,2))*.1,44100,subtype='FLOAT')
                prepare_pair(source,folder,lambda *a:None);self.assertEqual(model.call_count,2)
                bg.unlink()
                prepare_pair(source,folder,lambda *a:None);self.assertEqual(model.call_count,3)

    def test_selected_model_failure_keeps_previous_output_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.wav';destination=root/'dialogue.wav'
            sf.write(source,np.zeros((100,2)),44100,subtype='FLOAT');destination.write_bytes(b'previous result')
            def fail(src,dst,*args,**kwargs):dst.write_bytes(b'partial output');raise RuntimeError('Invalid prediction')
            with patch.object(hyperace,'load_model',return_value=object()),patch.object(hyperace,'overlap_add',side_effect=fail):
                with self.assertRaises(RuntimeError):hyperace.separate(source,destination,lambda *args:None)
            self.assertEqual(destination.read_bytes(),b'previous result')
            self.assertFalse((root/'dialogue.partial.wav').exists())

    def test_selected_model_rejects_mono_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'mono.wav';sf.write(source,np.zeros(100),44100)
            with patch.object(hyperace,'load_model') as load:
                with self.assertRaises(RuntimeError):hyperace.separate(source,root/'dialogue.wav',lambda *args:None)
                load.assert_not_called()

    def test_singing_is_excluded_without_merging_across_the_song(self):
        def segment(a,b,text):return {'start':a,'end':b,'original_text':text,'translated_text':text,'speaker_id':'speaker_1','emotion':'neutral'}
        parsed=Transcript.model_validate({'source_language':'ja','speakers':[{'id':'speaker_1','description':'voice'}],
            'segments':[segment(0.,1.,'대사'),segment(1.1,1.4,'가사'),segment(1.5,2.,'네')],
            'excluded_intervals':[{'start':1.,'end':1.5,'reason':'singing'}]})
        result=normalize([s.model_dump() for s in parsed.segments],[s.model_dump() for s in parsed.excluded_intervals])
        self.assertEqual([s['original_text'] for s in result],['대사','네'])


if __name__=='__main__':unittest.main()
