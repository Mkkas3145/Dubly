import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from source_recovery import recover,previous_island


class SourceRecoveryTests(unittest.TestCase):
    def cascade(self,checks):
        row=dict(source_index=0,start=.5,end=.8,words=[],translated_text='dialogue')
        job=dict(id=0,kind='prefix',row_index=0,source_index=0,translated_text='dialogue',
                 context_start=0.,context_end=1.,asr_start=.5)
        with tempfile.TemporaryDirectory() as temp:
            with patch('source_recovery.jobs_for',return_value=[job]),patch('source_recovery.config',return_value={'key':'test'}),\
                 patch('source_recovery.transcribe',return_value={'text':'recognized source'}) as api,\
                 patch('source_recovery.align_stage',return_value={0:{}}),\
                 patch('source_recovery.accept_prefix',side_effect=checks):
                result=recover(Path(temp)/'unused.wav',np.zeros((16000,2),np.float32),16000,{},
                    [row],[],[],[],Path(temp)/'recovery','ja')
                models=[c.kwargs['model'] for c in api.call_args_list]
        return result,models,row

    def test_successful_mai_does_not_call_whisper(self):
        updated=dict(source_index=0,start=.2,end=.8)
        result,models,_=self.cascade([(updated,{'reason':'verified'})])
        self.assertEqual(models,['microsoft/mai-transcribe-2'])
        self.assertEqual(result[0][0]['start'],.2)
        self.assertEqual(result[2][0]['attempts'],['elevenlabs/scribe-v2','microsoft/mai-transcribe-2'])

    def test_whisper_is_last_model_and_is_called_only_after_mai_failure(self):
        updated=dict(source_index=0,start=.2,end=.8)
        result,models,_=self.cascade([(None,{'reason':'missing'}),(updated,{'reason':'verified'})])
        self.assertEqual(models,['microsoft/mai-transcribe-2','openai/whisper-large-v3'])
        self.assertEqual(result[2][0]['status'],'recovered')
        self.assertEqual(len(result[2][0]['attempts']),3)

    def test_three_failed_models_do_not_assign_an_unverified_boundary(self):
        result,models,original=self.cascade([(None,{'reason':'missing'}),(None,{'reason':'missing'})])
        self.assertEqual(result[0],[original]);self.assertEqual(len(models),2)
        self.assertEqual(result[2][0]['status'],'unresolved')

    def test_nearest_island_does_not_reach_back_to_a_more_distant_voice(self):
        for sr in [16000,44100,48000]:
            for offset in [0.,4.37]:
                with self.subTest(sr=sr,offset=offset):
                    x=np.zeros((round((offset+3)*sr),2),np.float32)
                    for start,end in [(1.,1.5),(2.,2.2)]:
                        a,b=round((offset+start)*sr),round((offset+end)*sr)
                        x[a:b]=.2*np.sin(np.arange(b-a)[:,None]*2*np.pi*230/sr)
                    found=previous_island(x,sr,offset+.8,offset+2.45,.1)
                    self.assertGreater(found[0],offset+1.9);self.assertLess(found[0],offset+2.05)


if __name__=='__main__':unittest.main()
