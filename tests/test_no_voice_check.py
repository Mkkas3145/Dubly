import json,tempfile,unittest,sys,types
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
import soundfile as sf
from core import SR,config
import render

class NoVoiceCheckTests(unittest.TestCase):
    def test_legacy_enabled_setting_does_not_load_or_call_voice_model(self):
        fake=types.ModuleType('voice_similarity')
        fake.select_candidate=Mock(side_effect=AssertionError('Voice checks removed'))
        fake.policy=Mock(side_effect=AssertionError('Voice checks removed'))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sf.write(root/'source.wav',np.full((SR,2),.1,dtype=np.float32),SR)
            def request(reference,segment,language,key,destination):
                sf.write(destination,np.full((SR,2),.1,dtype=np.float32),SR,format='WAV')
                return {'generation_id':'single','generated_seconds':1.}
            segment={'lo':0,'hi':SR,'start':0.,'end':1.,'original_text':'source','translated_text':'target'}
            with patch.dict(sys.modules,{'voice_similarity':fake}),patch('render.request_audio',side_effect=request) as api:
                prepared=render.generate_speech(0,segment,root/'source.wav',root,{'language':'ko','key':'unused','voice_similarity':True,'tts_quality_enabled':True})
            api.assert_called_once();fake.select_candidate.assert_not_called();fake.policy.assert_not_called()
            self.assertNotIn('voice_similarity',prepared['generation'])

    def test_status_is_only_tts_even_with_legacy_config(self):
        seen=[]
        def queue(items,generator,fitter,progress):progress(0,1,0,0,0);return []
        with patch('render.pipelined_queue',side_effect=queue):
            render.synthesize_individual_all([{'lo':0}],None,None,{'voice_similarity':True},lambda *a:None,status=lambda p,s:seen.append(s))
        self.assertTrue(any(f'TTS 1/{render.TTS_CONCURRENCY}' in s for s in seen))
        self.assertFalse(any('음색' in s for s in seen))

    def test_configuration_no_longer_exposes_voice_check(self):
        with patch('core.dotenv_values',return_value={'VOICE_SIMILARITY_ENABLED':'true','TTS_QUALITY_ENABLED':'true'}):
            self.assertNotIn('voice_similarity',config())
            self.assertNotIn('tts_quality_enabled',config())

if __name__=='__main__':unittest.main()
