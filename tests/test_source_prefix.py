import unittest,json
import numpy as np
from source_prefix import prefix_jobs,apply_proposal,select_proposal
from reference_edges import acoustic_boundaries


def row(start=2.,end=2.6):
    return {'source_index':0,'start':start,'end':end,'words':[
        {'word':'know','start':start,'end':end}],'backend':'scribe'}


def job():
    return {'source_index':0,'text':'I... know!','missing_prefix_chars':1,
        'context_start':0.,'context_end':3.,'neighbor_floor':0.,'neighbor_ceiling':3.}


def proposal():
    return {'words':[{'word':'I','start':.6,'end':.9},{'word':'know','start':2.,'end':2.6}],
        'vad_start':0.,'vad_step':.01,'vad':np.ones(300).tolist()}


def audio(sr=16000,offset=0.,gain=.2):
    x=np.zeros((round((3+offset)*sr),2),dtype=np.float32)
    for a,b in [(.6,.9),(2.,2.6)]:
        lo,hi=round((a+offset)*sr),round((b+offset)*sr)
        x[lo:hi]=gain*np.sin(np.arange(hi-lo)[:,None]*2*np.pi*230/sr)
    return x


class SourcePrefixTests(unittest.TestCase):
    def test_only_missing_lexical_opening_triggers_alignment_in_multiple_languages(self):
        for full,recognized in [('I... I know!','I know'),('ぼ、僕が？','僕が'),('나…나는요!','나는요')]:
            with self.subTest(text=full):
                r={**row(),'words':[{'word':recognized,'start':2.,'end':2.6}]}
                jobs=prefix_jobs([{'original_text':full}], [r],3.)
                self.assertEqual(len(jobs),1);self.assertGreater(jobs[0]['missing_prefix_chars'],0)
        r={**row(),'words':[{'word':'I know.','start':2.,'end':2.6}]}
        self.assertEqual(prefix_jobs([{'original_text':'I know!'}],[r],3.),[])

    def test_missing_opening_and_internal_pause_survive_acoustic_measurement(self):
        for sr in [16000,44100,48000]:
            for offset,gain in [(0.,.2),(4.73,.02),(17.1,.8)]:
                with self.subTest(sr=sr,offset=offset,gain=gain):
                    r=row(2+offset,2.6+offset);j=job();p=proposal()
                    for key in ('context_start','context_end','neighbor_floor','neighbor_ceiling'):j[key]+=offset
                    for w in p['words']:w['start']+=offset;w['end']+=offset
                    p['vad_start']+=offset
                    values=audio(sr,offset,gain)
                    fixed,report=apply_proposal(values,sr,r,j,p)
                    self.assertEqual(report['status'],'recovered')
                    measured=acoustic_boundaries(values,sr,[fixed])[0]
                    self.assertLessEqual(measured['start'],.6+offset)
                    self.assertGreaterEqual(measured['end'],2.6+offset)
                    self.assertGreater(measured['duration'],2.)

    def test_silence_or_isolated_non_speech_cannot_become_a_missing_prefix(self):
        noise=audio();noise[:16000]=np.random.default_rng(7).normal(0,.1,(16000,2))
        for values,probability in [(np.zeros((48000,2),dtype=np.float32),1.),(noise,0.)]:
            p=proposal();p['vad']=(np.ones(300)*probability).tolist()
            updated,report=apply_proposal(values,16000,row(),job(),p)
            self.assertEqual(report['status'],'unchanged');self.assertEqual(updated,row())

    def test_zero_duration_prefix_is_not_given_an_invented_duration(self):
        p=proposal();p['words'][0]['end']=p['words'][0]['start']
        updated,report=apply_proposal(audio(),16000,row(),job(),p)
        self.assertEqual(report['reason'],'missing_prefix_has_no_positive_alignment')
        self.assertEqual(updated,row())

    def test_wrong_transcript_and_crossing_previous_dialogue_are_rejected(self):
        p=proposal();p['words'][0]['word']='you'
        self.assertEqual(apply_proposal(audio(),16000,row(),job(),p)[1]['reason'],'aligned_text_mismatch')
        j=job();j['neighbor_floor']=.8
        self.assertEqual(apply_proposal(audio(),16000,row(),j,proposal())[1]['reason'],'neighbor_boundary_conflict')

    def test_multiple_rejected_proposals_are_serializable_and_keep_original(self):
        p=proposal();p['words'][0]['end']=p['words'][0]['start']
        p['alternatives']=[p['words']]
        updated,report=select_proposal(audio(),16000,row(),job(),p)
        self.assertEqual(updated,row());self.assertEqual(len(report['rejected_proposals']),2)
        self.assertEqual(json.loads(json.dumps(report))['status'],'unchanged')

    def test_collapsed_body_cannot_validate_an_earlier_different_voice(self):
        p=proposal();p['words'][1].update(start=1.1,end=1.1)
        fixed,report=apply_proposal(audio(),16000,row(),job(),p)
        self.assertEqual(report['status'],'unchanged')
        self.assertEqual(report['reason'],'recognized_body_collapsed_in_alignment')
        self.assertEqual(fixed,row())

    def test_onset_recovery_does_not_move_the_known_asr_body_to_another_voice(self):
        p=proposal();p['words'][1].update(start=.9,end=1.4)
        p['vad']=(np.arange(300)<150).astype(float).tolist()
        values=audio();values[round(.9*16000):round(1.4*16000)]=.1
        _,report=apply_proposal(values,16000,row(),job(),p)
        self.assertEqual(report['reason'],'body_alignment_disagrees_with_asr')
        p['vad']=np.ones(300).tolist()
        _,report=apply_proposal(values,16000,row(),job(),p)
        self.assertEqual(report['reason'],'body_alignment_disagrees_with_asr')

    def test_nearest_complete_alignment_wins_over_an_unrelated_earlier_voice(self):
        values=audio();values[25600:28800]=.2*np.sin(np.arange(3200)[:,None]*2*np.pi*310/16000)
        p=proposal();p['words'][0].update(start=.6,end=1.8)
        p['alternatives']=[[{'word':'I','start':1.6,'end':1.8},{'word':'know','start':2.,'end':2.6}]]
        fixed,report=select_proposal(values,16000,row(),job(),p)
        self.assertEqual(report['status'],'recovered');self.assertEqual(fixed['start'],1.6)
        self.assertEqual(fixed['words'][-1],row()['words'][-1])
        self.assertTrue(any(r['reason']=='prefix_alignment_spans_unvoiced_gap' for r in report['rejected_proposals']))

    def test_first_word_stuck_on_previous_dialogue_context_edge_is_rejected(self):
        j=job();j['context_start']=.6;j['neighbor_floor']=.6
        _,report=apply_proposal(audio(),16000,row(),j,proposal())
        self.assertEqual(report['reason'],'prefix_stuck_at_previous_dialogue_context_edge')


if __name__=='__main__':unittest.main()
