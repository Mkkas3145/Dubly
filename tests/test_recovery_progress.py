import tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from source_progress import Progress
from source_neighborhood import repair
from reference_edges import validated_boundaries


class RecoveryProgressTests(unittest.TestCase):
    def test_provider_wait_emits_updates_without_inventing_percentage(self):
        seen=[]
        with Progress(lambda p,d:seen.append((p,d)),interval=.01).running() as status:
            status.update(25,'ASR 응답 0/2');time.sleep(.055)
        self.assertGreaterEqual(len(seen),3)
        self.assertTrue(all(p==25 and '0/2' in d for p,d in seen))

    def test_independent_neighborhoods_recognize_in_parallel(self):
        rows=[dict(source_index=i,start=i+.1,end=i+.4) for i in [0,2,4,6]]
        transcript={'segments':[{'translated_text':f'line {i}.'} for i in range(7)]}
        for i in [1,5]:transcript['segments'][i]['translated_text']='First. Second!'
        missing=[dict(source_index=i,translated_text=transcript['segments'][i]['translated_text']) for i in [1,5]]
        barrier=threading.Barrier(2);calls=[];progress=[]
        def transcribe(*args,**kwargs):
            calls.append(kwargs['model']);barrier.wait(3)
            return {'text':'source','words':[]}
        def match(targets,words,duration):
            if not words:return None
            return [dict(source_index=i,start=i+.1,end=i+.4,words=[dict(word='v',start=i+.1,end=i+.4)]) for i,text in targets]
        with tempfile.TemporaryDirectory() as temp,patch('source_neighborhood.match_neighborhood',side_effect=match),\
             patch('source_neighborhood.verified_native_words',return_value=[dict(word='v',start=0.,end=.1)]),\
             patch('source_neighborhood.assign_speakers'):
            result=repair(Path(temp)/'source.wav',np.zeros((8*16000,2),np.float32),16000,transcript,rows,missing,[],
                Path(temp)/'repair','Japanese','unused',transcribe,
                lambda *a,**k:self.fail('Successful native times must skip local alignment'),
                ('microsoft/mai-transcribe-2','openai/whisper-large-v3'),progress=lambda p,d:progress.append(d))
        self.assertEqual(calls,['microsoft/mai-transcribe-2']*2)
        self.assertEqual(result[1],[]);self.assertTrue(all(r['status']=='recovered' for r in result[2]))
        self.assertTrue(any('응답 2/2' in d for d in progress))

    def test_one_unmeasured_line_preserves_other_lines(self):
        for sr in [16000,44100,48000]:
            x=np.zeros(sr*4,np.float32)
            for a,b in [(.3,.6),(2.5,2.8)]:
                lo,hi=round(a*sr),round(b*sr);x[lo:hi]=.2*np.sin(np.arange(hi-lo)*2*np.pi*200/sr)
            rows=[dict(source_index=i,words=[dict(word='x',start=a,end=b)])
                  for i,a,b in [(4,.3,.6),(17,1.4,1.6),(29,2.5,2.8)]]
            kept,measured,errors=validated_boundaries(x,sr,rows)
            self.assertEqual([r['source_index'] for r in kept],[4,29]);self.assertEqual(len(measured),2)
            self.assertEqual(errors[0]['source_index'],17)
            self.assertEqual(errors[0]['action'],'preserve_original_audio')

if __name__=='__main__':unittest.main()
