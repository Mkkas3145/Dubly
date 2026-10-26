from pathlib import Path
import sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import SR
from mfa_client import MfaClient,WorkerError
from mfa_worker import ResidentAligner
from mfa_hint import parse_alignment,alignment_hint
from pause_matching import source_guided_plan,match_events
from timing import adjust_pauses


STUB='''import json,os,sys,time
for line in sys.stdin:
 r=json.loads(line)
 if r['text']=='hang':time.sleep(20)
 if r['text']=='exit':sys.exit(1)
 if r['text']=='bad':print('MFA_RESULT {}',flush=True);continue
 print('MFA_RESULT '+json.dumps({'ok':r['text']!='fail','id':r['id'],'pid':os.getpid(),'fatal':False,'error':'test'}),flush=True)
'''


def tone(seconds):
    t=np.arange(round(seconds*SR))/SR
    return np.repeat((.2*np.sin(2*np.pi*220*t))[:,None],2,axis=1).astype('float32')


class FastMfaTests(unittest.TestCase):
    def client(self,root):
        script=root/'stub.py';script.write_text(STUB,encoding='utf-8')
        return MfaClient([sys.executable,'-u',str(script)],log=root/'stderr.log')

    def test_ten_requests_reuse_one_process_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            client=self.client(Path(temp))
            try:
                replies=[client.request('audio','ok','output',timeout=5) for _ in range(10)]
                self.assertEqual(len({r['pid'] for r in replies}),1)
                self.assertEqual([r['id'] for r in replies],list(range(1,11)))
            finally:client.close()
            self.assertIsNotNone(client.process.poll());self.assertFalse(client.reader.is_alive())

    def test_timeout_disconnection_and_bad_reply_are_bounded(self):
        for text in ['hang','exit','bad']:
            with self.subTest(text=text),tempfile.TemporaryDirectory() as temp:
                client=self.client(Path(temp))
                try:
                    with self.assertRaises(WorkerError) as error:client.request('a',text,'b',timeout=.3 if text=='hang' else 5)
                    self.assertTrue(error.exception.fatal)
                finally:client.close()
                self.assertFalse(client.reader.is_alive())

    def test_bad_utterance_does_not_restart_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            client=self.client(Path(temp))
            try:
                first=client.request('a','ok','b',timeout=5)
                with self.assertRaises(WorkerError) as error:client.request('a','fail','b',timeout=5)
                self.assertFalse(error.exception.fatal)
                self.assertEqual(first['pid'],client.request('a','ok','b',timeout=5)['pid'])
            finally:client.close()

    def engine(self,known):
        engine=object.__new__(ResidentAligner);engine.g2p_cache={}
        engine.tokenizer=lambda text:SimpleNamespace(normalized_text=text,pronunciation_text=text)
        engine.g2p=SimpleNamespace(rewriter=Mock(return_value=[SimpleNamespace(pronunciation='a')]))
        engine.pronunciation=lambda word,phones:(word,phones)
        def add(pronunciation):known.add(pronunciation[0])
        engine.lexicon=SimpleNamespace(word_table=SimpleNamespace(member=lambda word:word in known),add_pronunciation=add)
        return engine

    def test_known_words_skip_g2p_and_new_words_are_added_only_once(self):
        engine=self.engine({'known'})
        self.assertEqual(engine.tokenize('known known'),'known known');engine.g2p.rewriter.assert_not_called()
        engine.tokenize('known new new');engine.tokenize('new known')
        engine.g2p.rewriter.assert_called_once_with('new')

    def test_failed_g2p_is_cached_instead_of_retried_for_each_segment(self):
        engine=self.engine(set());engine.g2p.rewriter.return_value=[]
        engine.tokenize('unknown');engine.tokenize('unknown');engine.g2p.rewriter.assert_called_once_with('unknown')

    def test_alignment_must_match_text_and_have_valid_intervals(self):
        data={'tiers':{'words':{'entries':[[0,.3,'one'],[.3,.5,'<eps>'],[.5,.8,'two']]},
                       'phones':{'entries':[[0,.3,'a'],[.3,.5,'sil'],[.5,.8,'b']]}}}
        self.assertEqual(len(parse_alignment(data,'one two',.8)['words']),2)
        with self.assertRaises(ValueError):parse_alignment(data,'wrong text',.8)
        data['tiers']['phones']['entries'][1]=[.2,.5,'sil']
        with self.assertRaises(ValueError):parse_alignment(data,'one two',.8)

    def test_mfa_cannot_authorize_edits_inside_a_word_or_without_quiet(self):
        hint={'status':'aligned','words':[{'start':0.,'end':1.,'label':'기뻐하자'}],'phones':[]}
        self.assertEqual(source_guided_plan(tone(1),hint=hint)['boundaries'],[])
        body=np.concatenate([tone(.3),np.zeros((round(.3*SR),2)),tone(.3)]).astype('float32')
        self.assertEqual(source_guided_plan(body,hint=hint)['boundaries'],[])

    def test_safe_pauses_expand_without_source_matches_and_share_the_deficit(self):
        body=np.concatenate([tone(.3),np.zeros((round(.3*SR),2)),tone(.3),np.zeros((round(.3*SR),2)),tone(.3)]).astype('float32')
        plan=source_guided_plan(body)
        edited,changes=adjust_pauses(body,len(body)+round(.2*SR),plan['boundaries'])
        self.assertEqual(len(changes),2);self.assertEqual(len(edited),len(body)+round(.2*SR))
        self.assertEqual([r['added_frames'] for r in changes],[round(.1*SR),round(.1*SR)])
        np.testing.assert_array_equal(edited[:round(.3*SR)],body[:round(.3*SR)])
        np.testing.assert_array_equal(edited[-round(.3*SR):],body[-round(.3*SR):])

    def test_short_pauses_need_word_boundary_evidence(self):
        body=np.concatenate([tone(.3),np.zeros((round(.1*SR),2)),tone(.3)]).astype('float32')
        self.assertEqual(source_guided_plan(body)['boundaries'],[])
        hint={'status':'aligned','phones':[],'words':[{'start':0.,'end':.3,'label':'one'},{'start':.4,'end':.7,'label':'two'}]}
        self.assertEqual(len(source_guided_plan(body,hint=hint)['boundaries']),1)

    def test_source_match_can_add_evidence_but_is_not_required_for_expansion(self):
        import soundfile as sf
        body=np.concatenate([tone(.4),np.zeros((round(.3*SR),2)),tone(.4)]).astype('float32')
        with tempfile.TemporaryDirectory() as temp:
            ref=Path(temp)/'reference.wav'
            for pause in [.5,.8]:
                reference=np.concatenate([tone(.4),np.zeros((round(pause*SR),2)),tone(.4)]).astype('float32')
                sf.write(ref,reference,SR,subtype='FLOAT')
                plan=source_guided_plan(body,ref,len(reference))
                edited,changes=adjust_pauses(body,len(body)+3*SR,plan['boundaries'])
                self.assertEqual(sum(c['added_frames'] for c in changes),round((pause-.3)*SR))
                self.assertEqual(len(edited),len(body)+round((pause-.3)*SR))
                np.testing.assert_array_equal(edited[:round(.4*SR)],body[:round(.4*SR)])
                np.testing.assert_array_equal(edited[-round(.4*SR):],body[-round(.4*SR):])
            sf.write(ref,tone(2),SR,subtype='FLOAT')
            plan=source_guided_plan(body,ref,2*SR)
            edited,changes=adjust_pauses(body,4*SR,plan['boundaries'])
            self.assertEqual(sum(c['added_frames'] for c in changes),round(.15*SR))
            self.assertLess(len(edited),4*SR)

    def test_single_pause_never_absorbs_a_large_deficit_and_limits_scale_with_audio(self):
        for scale in [1.,2.,3.]:
            body=np.concatenate([tone(.4*scale),np.zeros((round(.2*scale*SR),2)),tone(.4*scale)]).astype('float32')
            plan=source_guided_plan(body)
            edited,changes=adjust_pauses(body,len(body)+3*SR,plan['boundaries'])
            # A relative duration budget, not a fixed seconds or sample-specific exception.
            self.assertAlmostEqual(sum(c['added_frames'] for c in changes)/SR,.1*scale,places=3)
            self.assertLess(len(edited),len(body)+3*SR)
            edit=changes[0];cut=edit['at_frame'];added=edit['added_frames']
            np.testing.assert_array_equal(np.concatenate([edited[:cut],edited[cut+added:]]),body)

    def test_adaptive_expansion_uses_existing_pause_lengths_without_changing_speech(self):
        body=np.concatenate([tone(.8),np.zeros((round(.2*SR),2)),tone(.8),
            np.zeros((round(.4*SR),2)),tone(.8)]).astype('float32')
        plan=source_guided_plan(body)
        edited,changes=adjust_pauses(body,len(body)+round(.2*SR),plan['boundaries'])
        self.assertEqual(len(changes),2)
        self.assertAlmostEqual(changes[1]['added_frames']/changes[0]['added_frames'],2.,places=2)
        self.assertEqual(len(edited),len(body)+round(.2*SR))

    def test_missing_runtime_does_not_trigger_network_requests(self):
        with tempfile.TemporaryDirectory() as temp,patch('mfa_hint.runtime_ready',return_value=False),\
             patch('urllib.request.urlopen',side_effect=AssertionError('No API for pause fitting')):
            self.assertEqual(alignment_hint(tone(1),'one two','ko',Path(temp)/'cache.json')['status'],'unavailable')


if __name__=='__main__':unittest.main()
