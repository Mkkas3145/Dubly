import json,tempfile,unittest,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR
import render,batch_tts,speech


def line(i,speaker=None):
    return {'lo':i*SR,'hi':(i+1)*SR,'start':float(i),'end':float(i+1),
        'original_text':f'original {i}','translated_text':f'target {i}',**({'speaker_id':speaker} if speaker else {})}


class BatchTests(unittest.TestCase):
    def test_conflicting_two_lines_do_not_discard_an_independent_third_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);values=np.zeros((3*SR,2),np.float32)
            values[2*SR:round(2.4*SR)]=.1
            sf.write(p/'generated.wav',values,SR,subtype='FLOAT')
            words=[{'word':t,'start':a,'end':b} for t,a,b in [('alpha',.1,.4),('beta',.6,.8),('gamma',1.,1.4),('delta',2.,2.4)]]
            loc=[{'matched':True,'coverage':1.,'word_indices':ids,'words':[words[i] for i in ids],'start':a,'end':b} for ids,a,b in [([0,1],.1,.8),([1,2],.6,1.4),([3],2.,2.4)]]
            rows=[{**line(i),'translated_text':t} for i,t in enumerate(['alpha beta','beta gamma','delta'])]
            with patch('scribe.transcribe_words',return_value={'words':words}),patch('source_alignment.locate_transcript',return_value=loc):
                _,cuts,report=batch_tts.split_generation(p/'unused.mp3',rows,[0,1,2],p,{'key':'test','language':'en','keep_original_on_tts_failure':True})
            self.assertIsNone(cuts[0]);self.assertIsNone(cuts[1]);self.assertIsNotNone(cuts[2])
            self.assertEqual(len(report['kept_original']),2)

    def test_stationary_generated_floor_does_not_reject_word_verified_gap(self):
        values=np.full((2*SR,2),.03,np.float32)
        values[round(.6*SR):round(1.3*SR)]=np.random.default_rng(10).normal(0,.001,(round(1.3*SR)-round(.6*SR),2))
        boundary,review=batch_tts.verified_dialogue_boundary(values,SR,.6,1.3)
        self.assertTrue(review['verified']);self.assertTrue(.6*SR<=boundary<1.3*SR)

    def test_boundary_does_not_cut_inside_low_energy_breath(self):
        values=np.full((2*SR,2),.1,np.float32)
        # Words are loud; a quiet breath occupies the middle of their gap.
        values[round(.5*SR):round(.7*SR)]=.00001
        values[round(.7*SR):round(1.1*SR)]=.0003
        values[round(1.1*SR):round(1.3*SR)]=.00001
        boundary,review=batch_tts.verified_dialogue_boundary(values,SR,.5,1.3)
        self.assertTrue(review['verified'])
        self.assertFalse(.7*SR<=boundary<1.1*SR)

    def test_unsafe_boundary_keeps_affected_pair_without_discarding_other_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);values=np.full((4*SR,2),.1,np.float32)
            values[round(1.7*SR):round(2.*SR)]=0
            values[round(2.7*SR):round(3.*SR)]=0
            sf.write(p/'generated.wav',values,SR,subtype='FLOAT')
            words=[{'word':text,'start':start,'end':end} for text,start,end in
                   [('Alpha',.1,.8),('Bravo',1.,1.7),('Charlie',2.,2.7),('Delta',3.,3.7)]]
            rows=[{'translated_text':w['word']} for w in words]
            with patch('scribe.transcribe_words',return_value={'words':words}):
                actual,cuts,report=batch_tts.split_generation(p/'unused.mp3',rows,list(range(4)),p,
                    {'language':'en','key':'unused','keep_original_on_tts_failure':True})
            self.assertEqual(cuts[:2],[None,None])
            self.assertTrue(all(c is not None for c in cuts[2:]))
            self.assertEqual([r['source_index'] for r in report['kept_original']],[0,1])
            self.assertTrue(all(r['reason']=='unverified_generated_dialogue_boundary' for r in report['kept_original']))
            self.assertLessEqual(cuts[2][0],round(2.*SR));self.assertGreaterEqual(cuts[2][1],round(2.7*SR))

    def test_boundary_cannot_borrow_quietness_inside_previous_aligned_word(self):
        values=np.full((SR,2),.1,np.float32)
        values[round(.3*SR):round(.4*SR)]=0
        boundary,review=batch_tts.verified_dialogue_boundary(values,SR,.5,.7)
        self.assertIsNone(boundary);self.assertFalse(review['verified'])
        values[round(.54*SR):round(.65*SR)]=0
        boundary,review=batch_tts.verified_dialogue_boundary(values,SR,.5,.7)
        self.assertTrue(review['verified']);self.assertTrue(.5*SR<=boundary<=.7*SR)

    def test_context_mode_requests_each_line_separately_without_batch_splitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);folder=p/'pieces';folder.mkdir()
            sf.write(p/'clean.wav',np.full((3*SR,2),.2,dtype=np.float32),SR,subtype='FLOAT')
            rows=[line(0,'0'),line(1,'1'),line(2,'0')];calls=[]
            def request(reference,segment,language,key,destination):
                calls.append((segment['translated_text'],sf.info(reference).frames))
                self.assertNotIn('_joint_generation',segment)
                sf.write(destination,np.full((SR,2),.1),SR,format='WAV')
                return {'generation_id':'test'}
            cfg={'language':'ko','key':'unused','tts_mode':'individual','tts_reference_mode':'same_speaker_context'}
            with patch('render.request_audio',side_effect=request),patch('render.fit_speech',side_effect=lambda item,cfg:{'index':item['index']}),patch('batch_tts.split_generation',side_effect=AssertionError('No batch splitting')):
                result=render.synthesize_all(rows,p/'clean.wav',folder,cfg,lambda *a:None)
            self.assertEqual(sorted(calls),[('target 0',2*SR),('target 1',SR),('target 2',2*SR)])
            self.assertEqual([r['index'] for r in result],[0,1,2])
            self.assertNotEqual(rows[0]['_shared_reference_path'],rows[2]['_shared_reference_path'])
            self.assertEqual(rows[0]['_shared_reference_indices'],[0,2])
            self.assertEqual(rows[2]['_shared_reference_text'],'original 2\noriginal 0')
            receipt=json.loads((folder/'0002-generation.json').read_text())
            self.assertEqual(receipt['shared_reference_indices'],[2,0])
            self.assertEqual(receipt['shared_reference_speaker'],'0')

    def test_regeneration_reference_puts_target_audio_and_text_before_remaining_voice_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)
            values=np.repeat(np.array([.1,.2,.3,.4],dtype=np.float32),SR)
            sf.write(p/'clean.wav',values,SR,subtype='FLOAT')
            rows=[line(0,'0'),line(1,'1'),line(2,'0'),{**line(3,'0'),'deleted':True}]
            indices=batch_tts.prepare_regeneration_reference(rows,2,p/'clean.wav',p)
            self.assertEqual(indices,[2,0])
            actual,rate=sf.read(rows[2]['_shared_reference_path'],dtype='float32')
            self.assertEqual(len(actual),2*SR)
            np.testing.assert_allclose(actual[:SR],.3,atol=1/32768)
            np.testing.assert_allclose(actual[SR:],.1,atol=1/32768)
            self.assertEqual(rows[2]['_shared_reference_text'],'original 2\noriginal 0')
            self.assertEqual(rows[2]['translated_text'],'target 2')
            self.assertFalse(rows[2]['_shared_reference_donor_tag'])

    def test_performance_cues_do_not_count_as_spoken_alignment_words(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);values=np.ones((2*SR,2),dtype=np.float32)*.1
            values[round(.4*SR):round(.8*SR)]=0
            sf.write(p/'generated.wav',values,SR,subtype='FLOAT')
            rows=[{**line(0),'translated_text':'[泣きながら]아무것도,'},
                  {**line(1),'translated_text':'[大声で]아무것도 변한 게 없잖아, 넌!'}]
            words=[{'word':'아무것도','start':.1,'end':.4},
                   {'word':'아무것도 변한 게 없잖아 넌','start':.8,'end':1.8}]
            with patch('scribe.transcribe_words',return_value={'words':words}):
                _,cuts,report=batch_tts.split_generation(p/'unused.mp3',rows,[0,1],p,
                    {'language':'ko','key':'test','keep_original_on_tts_failure':True})
            self.assertTrue(all(c is not None for c in cuts));self.assertEqual(report['kept_original'],[])
            self.assertEqual(rows[0]['translated_text'],'[泣きながら]아무것도,')

    def test_cue_only_slot_does_not_claim_neighboring_speech(self):
        self.assertEqual(batch_tts.spoken_text('[足音] ... [悲しい]'),'...')
        self.assertEqual(batch_tts.spoken_text('<|speaker:0|>[calm]Hello [quietly]there'),'Hello there')

    def test_speaker_batches_keep_known_voices_together_and_unknowns_separate(self):
        rows=[line(0,'0'),line(1,'1'),line(2,'0'),line(3,'unknown'),line(4,'unknown')]
        self.assertEqual(batch_tts.plan_speaker_batches(rows),[[0,2],[1],[3],[4]])
        self.assertEqual([len(g) for g in batch_tts.plan_speaker_batches([line(i,'0') for i in range(65)])],[60,5])
    def test_separate_source_fragments_prove_corrupted_reference_replay(self):
        words=[{'word':'alpha','start':.1,'end':.4},{'word':'beta','start':.5,'end':.8},
               {'word':'gamma','start':.9,'end':1.2},{'word':'bonjour','start':2.,'end':3.}]
        row={'start':2.,'end':3.,'word_indices':[3],'words':words[3:]}
        segments=[{'original_text':'alpha beta gamma delta epsilon zeta eta theta','translated_text':'bonjour'}]
        cuts,audit=batch_tts.exclude_unrequested_replay([(0,4000)],[row],words,segments,[0],1000)
        self.assertGreater(cuts[0][0],0)
        self.assertIn('independent_source_fragments',audit[0]['evidence'])

    def test_unmatched_line_keeps_original_without_stopping_valid_neighbor(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);sf.write(p/'generated.wav',np.ones((SR,2),dtype=np.float32)*.1,SR,subtype='FLOAT')
            rows=[{**line(0),'translated_text':'missing'}, {**line(1),'translated_text':'Hello'}]
            words=[{'word':'Hello','start':.3,'end':.8}]
            with patch('scribe.transcribe_words',return_value={'words':words}):
                _,cuts,report=batch_tts.split_generation(p/'unused.mp3',rows,[0,1],p,
                    {'language':'en','key':'test','keep_original_on_tts_failure':True})
            self.assertIsNone(cuts[0]);self.assertIsNotNone(cuts[1])
            self.assertGreaterEqual(cuts[1][0],round(.22*SR))
            self.assertEqual(report['kept_original'][0]['source_index'],0)

    def test_edge_recovery_ignores_punctuation_slot_and_requires_phonetic_evidence(self):
        missing=lambda:{'matched':False,'words':[],'word_indices':[]}
        words=[{'word':'이건','start':.1,'end':.5},{'word':'서역소','start':.6,'end':1.2},
               {'word':'다음','start':1.5,'end':2.}]
        locations=[missing(),missing(),{'matched':True,'start':1.5,'end':2.,'word_indices':[2],'words':words[2:]}]
        result,audit=batch_tts.recover_edge_anchors(locations,words,['…','이건 서약서?','다음'],np.ones((3*SR,2))*.1,SR)
        self.assertFalse(result[0]['matched']);self.assertTrue(result[1]['matched'])
        self.assertEqual(result[1]['word_indices'],[0,1]);self.assertEqual(len(audit),1)
        result,_=batch_tts.recover_edge_anchors(locations,words,['…','네.','다음'],np.ones((3*SR,2))*.1,SR)
        self.assertFalse(result[1]['matched'])

    def test_edge_recovery_accepts_stammer_elided_by_asr(self):
        words=[{'word':'저요?','start':.7,'end':1.4},{'word':'다음','start':2.,'end':3.}]
        locations=[{'matched':False,'words':[],'word_indices':[]},
            {'matched':True,'start':2.,'end':3.,'words':words[1:],'word_indices':[1]}]
        result,_=batch_tts.recover_edge_anchors(locations,words,['저, 저요?','다음'],np.ones((4*SR,2))*.1,SR)
        self.assertTrue(result[0]['matched']);self.assertEqual(result[0]['word_indices'],[0])

    def test_punctuation_only_slot_is_not_forced_into_a_speech_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);sf.write(p/'generated.wav',np.ones((SR,2),dtype=np.float32)*.1,SR,subtype='FLOAT')
            rows=[{**line(0),'translated_text':'...'}, {**line(1),'translated_text':'Hello'}]
            words=[{'word':'Hello','start':.1,'end':.8}]
            with patch('scribe.transcribe_words',return_value={'words':words}):
                _,cuts,_=batch_tts.split_generation(p/'unused.mp3',rows,[0,1],p,{'language':'en','key':'test'})
            self.assertIsNone(cuts[0]);self.assertEqual(cuts[1],(0,SR))

    def test_keep_original_slot_has_no_voice_removal_interval(self):
        rows=[{**line(0),'keep_original':True}]
        self.assertEqual(render.replacement_intervals(rows,SR,[{'lo':0,'hi':1}]),[])

    def test_keep_original_slot_preserves_mix_samples_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);x=np.linspace(-.2,.2,2*SR,dtype=np.float32).reshape(SR,2)
            sf.write(p/'original.wav',x,SR,subtype='FLOAT')
            sf.write(p/'clean.wav',x*.5,SR,subtype='FLOAT')
            sf.write(p/'background.wav',np.zeros_like(x),SR,subtype='FLOAT')
            sf.write(p/'0000-fitted.wav',np.zeros((1,2),dtype=np.float32),SR,subtype='FLOAT')
            rows=[{**line(0),'keep_original':True}]
            render.mix(p/'original.wav',p/'clean.wav',rows,p,p/'mixed.wav',lambda *a:None,background=p/'background.wav')
            actual,_=sf.read(p/'mixed.wav',dtype='float32',always_2d=True)
            np.testing.assert_array_equal(actual,x)

    def test_neighbor_transition_cannot_remove_a_kept_original_utterance(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);x=np.ones((SR,2),dtype=np.float32)*.2
            sf.write(p/'original.wav',x,SR,subtype='FLOAT')
            sf.write(p/'clean.wav',x,SR,subtype='FLOAT')
            sf.write(p/'background.wav',np.zeros_like(x),SR,subtype='FLOAT')
            sf.write(p/'0000-fitted.wav',np.ones((SR//4,2),dtype=np.float32)*.1,SR,subtype='FLOAT')
            sf.write(p/'0001-fitted.wav',np.zeros((1,2),dtype=np.float32),SR,subtype='FLOAT')
            rows=[{**line(0),'hi':SR//2,'end':.5},
                  {**line(1),'lo':SR//2,'hi':SR,'start':.5,'end':1.,'keep_original':True}]
            render.mix(p/'original.wav',p/'clean.wav',rows,p,p/'mixed.wav',lambda *a:None,background=p/'background.wav')
            actual,_=sf.read(p/'mixed.wav',dtype='float32',always_2d=True)
            np.testing.assert_array_equal(actual[SR//2:],x[SR//2:])

    def test_long_references_are_whole_and_can_exceed_sixty_seconds(self):
        rows=[{**line(i),'hi':i*SR+180*SR} for i in range(35)]
        self.assertEqual([len(x) for x in batch_tts.plan_batches(rows)],[1]*35)
        self.assertEqual(sum(batch_tts.plan_batches(rows),[]),list(range(35)))

    def test_stable_speaker_slots_on_both_transcripts(self):
        rows=[line(0,'speaker_8'),line(1,'speaker_12'),line(2,'speaker_8')]
        original,target,tags=batch_tts.tagged_text(rows,[0,1,2])
        self.assertEqual(tags,[0,1,0])
        self.assertEqual(original.count('<|speaker:0|>'),2)
        self.assertEqual(target.count('<|speaker:0|>'),2)
        self.assertEqual(batch_tts.tagged_text([line(0),line(1)],[0,1])[2],[0,1])

    def test_gemini_compact_ids_survive_final_boundaries(self):
        data={'segments':[{'start':0.,'end':1.,'original_text':'first','translated_text':'첫째','speaker_id':'speaker_3'},
            {'start':1.,'end':2.,'original_text':'second','translated_text':'둘째','speaker_id':'speaker_3'}],'excluded_intervals':[]}
        parsed=speech.parse_transcript(json.dumps(data))
        result=speech.normalize([s.model_dump() for s in parsed.segments],[],preserve_speakers=True)
        self.assertEqual([s['speaker_id'] for s in result],['speaker_3','speaker_3'])
        self.assertEqual(len(result),2)

    def test_split_uses_one_asr_and_preserves_every_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);values=np.linspace(-.4,.4,4*SR,dtype=np.float32).reshape(2*SR,2);sf.write(p/'generated.wav',values,SR,subtype='FLOAT')
            rows=[{'translated_text':'Hello there'},{'translated_text':'Goodbye now'}]
            words=[{'word':'Hello','start':.1,'end':.4},{'word':'there','start':.4,'end':.8},
                {'word':'Goodbye','start':1.2,'end':1.5},{'word':'now','start':1.5,'end':1.9}]
            with patch('scribe.transcribe_words',return_value={'words':words}) as asr:
                audio,cuts,_=batch_tts.split_generation(p/'unused.mp3',rows,[0,1],p,{'language':'en','key':'unused'})
            asr.assert_called_once();self.assertEqual(cuts[0][0],0);self.assertEqual(cuts[0][1],cuts[1][0]);self.assertEqual(cuts[-1][1],2*SR);self.assertTrue(.8*SR<=cuts[0][1]<=1.2*SR)
            self.assertEqual(asr.call_args.kwargs['attempts'],1)
            np.testing.assert_array_equal(np.concatenate([audio[a:b] for a,b in cuts]),values)
            with patch('scribe.transcribe_words',return_value={'words':words[:2]}):
                with self.assertRaisesRegex(RuntimeError,'연결하지 못했습니다'):
                    batch_tts.split_generation(p/'unused.mp3',rows,[0,1],p,{'language':'en','key':'unused'})

    def test_validated_alignment_is_not_rejected_by_second_character_cutoff(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);values=np.full((2*SR,2),.1,dtype=np.float32)
            values[round(.8*SR):round(1.2*SR)]=0
            sf.write(p/'generated.wav',values,SR,subtype='FLOAT')
            locations=[{'matched':True,'coverage':2/3,'start':.1,'end':.8,'word_indices':[0,1]},
                       {'matched':True,'coverage':1.,'start':1.2,'end':1.9,'word_indices':[2,3]}]
            rows=[{'translated_text':f'line {i}'} for i in range(8)]
            with patch('scribe.transcribe_words',return_value={'words':[]}),patch('source_alignment.locate_transcript',return_value=locations):
                _,cuts,_=batch_tts.split_generation(p/'unused.mp3',rows,[3,7],p,{'key':'unused','language':'ko'})
            self.assertEqual(cuts[0][1],cuts[1][0]);self.assertTrue(.8*SR<=cuts[0][1]<=1.2*SR)
            locations[0]['matched']=False
            with patch('scribe.transcribe_words',return_value={'words':[]}),patch('source_alignment.locate_transcript',return_value=locations):
                with self.assertRaisesRegex(RuntimeError,'4번'):
                    batch_tts.split_generation(p/'unused.mp3',rows,[3,7],p,{'key':'unused','language':'ko'})

    def test_batch_request_cache_reuses_audio_and_parallel_fit_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);clean=root/'clean.wav';sf.write(clean,np.full((3*SR,2),.1,dtype=np.float32),SR)
            rows=[line(i,'speaker_1' if i!=1 else 'speaker_2') for i in range(3)]
            def request(ref,segment,language,key,destination):
                self.assertEqual(segment['translated_text'].count('<|speaker:0|>'),2)
                sf.write(destination,np.full((3*SR,2),.1,dtype=np.float32),SR,format='WAV')
                return {'generation_id':'one-batch','tts_policy':render.tts_policy(language)}
            def split(audio,segments,indices,folder,cfg):
                return np.full((3*SR,2),.1,dtype=np.float32),[(i*SR,(i+1)*SR) for i in range(3)],{'status':'test'}
            def fit(item,cfg):
                sf.write(item['fitted'],np.full((SR,2),.1,dtype=np.float32),SR,subtype='FLOAT')
                self.assertEqual(item['generation']['tts_generation_mode'],'joint_tagged_reference')
                return {'index':item['index'],'collision_budget':item['collision_budget']}
            cfg={'language':'ko','key':'unused'}
            with patch('render.request_audio',side_effect=request) as api,patch('batch_tts.split_generation',side_effect=split),patch('render.fit_speech',side_effect=fit):
                first=batch_tts.synthesize_batches(rows,clean,root/'pieces',cfg,lambda *a:None)
                second=batch_tts.synthesize_batches(rows,clean,root/'pieces',cfg,lambda *a:None)
            self.assertEqual(api.call_count,1);self.assertEqual(first,second)
            self.assertEqual([x['index'] for x in first],[0,1,2])
            self.assertEqual(first[0]['collision_budget'],SR);self.assertIsNone(first[-1]['collision_budget'])

    def test_continuous_reference_keeps_original_gaps_and_per_turn_fit_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);clean=root/'clean.wav'
            values=np.repeat(np.linspace(.1,.3,3*SR,dtype=np.float32)[:,None],2,axis=1)
            sf.write(clean,values,SR,subtype='FLOAT')
            rows=[line(0),{**line(2),'hi':3*SR}]
            def request(ref,segment,language,key,destination):
                self.assertEqual(sf.info(ref).frames,3*SR)
                audio,_=sf.read(ref,dtype='float32')
                self.assertGreater(float(np.max(np.abs(audio[SR:2*SR]))),.1)
                self.assertIn('<|speaker:1|>target 2',segment['translated_text'])
                self.assertEqual(segment['_instructions'],render.tts_instructions('ko'))
                sf.write(destination,np.full((2*SR,2),.1,dtype=np.float32),SR,format='WAV')
                return {'generation_id':'one-batch','generated_seconds':2.}
            def split(*args):return np.full((2*SR,2),.1,dtype=np.float32),[(0,SR),(SR,2*SR)],{'status':'test'}
            def fit(item,cfg):
                return {'start':item['segment']['start'],'end':item['segment']['end'],
                        'cut_seconds':item['generation']['generated_seconds'],
                        'group_seconds':item['generation']['tts_group_generated_seconds']}
            cfg={'language':'ko','key':'unused'}
            with patch('render.request_audio',side_effect=request) as api,patch('batch_tts.split_generation',side_effect=split),patch('render.fit_speech',side_effect=fit):
                result=batch_tts.synthesize_batches(rows,clean,root/'pieces',cfg,lambda *a:None,
                    groups=[[0,1]],tag_mode='reference_slot',reference_mode='continuous_source',instructions=render.tts_instructions('ko'))
            self.assertEqual(api.call_count,1)
            self.assertEqual([(r['start'],r['end']) for r in result],[(0.,1.),(2.,3.)])
            self.assertTrue(all(r['cut_seconds']==1 and r['group_seconds']==2 for r in result))


    def test_nonadjacent_same_voice_references_exclude_other_speakers_and_restore_placement(self):
        import same_speaker_tts
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);clean=root/'clean.wav'
            source=np.concatenate([np.full((SR,2),v,np.float32) for v in (.1,.2,.3,.4)])
            sf.write(clean,source,SR,subtype='FLOAT')
            rows=[{**line(i),'_tts_group':i%2,'_tts_slot':i//2} for i in range(4)]
            def request(ref,segment,language,key,destination):
                original=segment['original_text']
                first=0 if 'original 0' in original else 1
                audio,rate=sf.read(ref,dtype='float32',always_2d=True)
                self.assertEqual(len(audio),2*SR)
                np.testing.assert_allclose(audio[:SR],np.full_like(audio[:SR],(.1,.2)[first]),atol=.0001)
                np.testing.assert_allclose(audio[SR:],np.full_like(audio[SR:],(.3,.4)[first]),atol=.0001)
                self.assertIn(f'<|speaker:1|>original {first+2}',original)
                self.assertEqual(segment['_instructions'],render.tts_instructions('ko'))
                sf.write(destination,np.full((2*SR,2),.1,np.float32),SR,format='WAV')
                return {'generated_seconds':2.}
            def split(audio,segments,indices,folder,cfg):
                self.assertIn(indices,([0,2],[1,3]))
                return np.full((2*SR,2),.1,np.float32),[(0,SR),(SR,2*SR)],{'status':'test'}
            def fit(item,cfg):
                return {'index':item['index'],'start':item['segment']['start'],'budget':item['collision_budget']}
            with patch('render.request_audio',side_effect=request) as api,patch('batch_tts.split_generation',side_effect=split) as align,patch('render.fit_speech',side_effect=fit):
                result=render.synthesize_all(rows,clean,root/'pieces',{'language':'ko','key':'unused'},lambda *a:None)
            self.assertEqual(api.call_count,2);self.assertEqual(align.call_count,2)
            self.assertEqual([r['index'] for r in result],[0,1,2,3])
            self.assertEqual([r['start'] for r in result],[0.,1.,2.,3.])
            self.assertEqual([r['budget'] for r in result],[SR,SR,SR,None])
            manifest=json.loads((root/'pieces/joint-batches.json').read_text('utf8'))
            self.assertEqual(manifest['groups'],[[0,2],[1,3]])
            self.assertEqual(manifest['reference_mode'],'concatenated')


    def test_neighbor_anchors_recover_asr_wording_mismatch_without_losing_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);audio=np.full((3*SR,2),.1,np.float32)
            audio[round(.8*SR):round(1.2*SR)]=0;audio[round(1.6*SR):round(2.2*SR)]=0
            sf.write(p/'generated.wav',audio,SR,subtype='FLOAT')
            rows=[{'translated_text':'첫째 문장'},{'translated_text':'아냐'},{'translated_text':'마지막 문장'}]
            words=[{'word':'첫째 문장','start':.1,'end':.8},
                   {'word':'안녕','start':1.2,'end':1.6},{'word':'마지막 문장','start':2.2,'end':2.8}]
            locations=[{'matched':True,'start':.1,'end':.8,'word_indices':[0]},
                       {'matched':False,'coverage':0.,'words':[],'word_indices':[]},
                       {'matched':True,'start':2.2,'end':2.8,'word_indices':[2]}]
            with patch('scribe.transcribe_words',return_value={'words':words}) as asr,patch('source_alignment.locate_transcript',return_value=locations):
                values,cuts,info=batch_tts.split_generation(p/'unused.mp3',rows,[0,1,2],p,{'key':'unused','language':'ko'})
            asr.assert_called_once();self.assertEqual(len(cuts),3)
            np.testing.assert_array_equal(np.concatenate([values[a:b] for a,b in cuts]),audio)
            self.assertEqual(info['neighbor_recovery'][0]['local_index'],1)
            self.assertEqual(info['neighbor_recovery'][0]['method'],'unclaimed_asr_between_validated_neighbors')

    def test_missing_asr_uses_audio_only_between_measured_neighbors(self):
        audio=np.zeros((3*SR,2),np.float32)
        audio[round(1.2*SR):round(1.6*SR)]=.15
        locations=[{'matched':True,'start':.1,'end':.8,'word_indices':[0]},
                   {'matched':False,'word_indices':[]},
                   {'matched':True,'start':2.2,'end':2.8,'word_indices':[1]}]
        words=[{'word':'first','start':.1,'end':.8},{'word':'last','start':2.2,'end':2.8}]
        recovered,audit=batch_tts.recover_between_anchors(locations,words,audio,SR)
        self.assertTrue(recovered[1]['matched'])
        self.assertLessEqual(recovered[1]['start'],1.2);self.assertGreaterEqual(recovered[1]['end'],1.6)
        self.assertEqual(audit[0]['method'],'measured_audio_between_validated_neighbors')
        silent,audit=batch_tts.recover_between_anchors(locations,words,np.zeros_like(audio),SR)
        self.assertFalse(silent[1]['matched']);self.assertEqual(audit,[])

    def test_neighbor_recovery_does_not_guess_multiple_missing_lines(self):
        locations=[{'matched':True,'start':0.,'end':.5,'word_indices':[0]},
                   {'matched':False,'word_indices':[]},{'matched':False,'word_indices':[]},
                   {'matched':True,'start':3.,'end':4.,'word_indices':[2]}]
        words=[{'word':'first','start':0.,'end':.5},{'word':'extra','start':1.,'end':2.},
               {'word':'last','start':3.,'end':4.}]
        result,audit=batch_tts.recover_between_anchors(locations,words,np.full((4*SR,2),.1,np.float32),SR)
        self.assertFalse(result[1]['matched']);self.assertFalse(result[2]['matched']);self.assertEqual(audit,[])


    def test_reference_replay_before_target_is_excluded_without_compressing_both(self):
        rate=100
        rows=[{'original_text':'reference dialogue','translated_text':'target dialogue'},
              {'original_text':'second reference','translated_text':'second target'}]
        words=[{'word':'reference dialogue','start':.1,'end':4.},
               {'word':'target dialogue','start':4.2,'end':7.},
               {'word':'second target','start':8.,'end':9.}]
        locations=[{'start':4.2,'end':7.,'word_indices':[1]},
                   {'start':8.,'end':9.,'word_indices':[2]}]
        cuts,audit=batch_tts.exclude_unrequested_replay([(0,750),(750,1000)],locations,words,rows,[0,1],rate)
        self.assertEqual(cuts,[(410,750),(750,1000)])
        self.assertEqual(audit[0]['excluded_frames'],410)
        self.assertEqual(audit[0]['side'],'before')

    def test_short_unmatched_opening_and_breath_silence_are_preserved(self):
        rows=[{'original_text':'long source dialogue','translated_text':'long target dialogue'}]
        words=[{'word':'uh','start':.1,'end':.3},{'word':'long target dialogue','start':.4,'end':2.}]
        cuts,audit=batch_tts.exclude_unrequested_replay([(0,300)],
            [{'start':.4,'end':2.,'word_indices':[1]}],words,rows,[0],100)
        self.assertEqual(cuts,[(0,300)]);self.assertEqual(audit,[])

    def test_reference_replay_in_gap_and_tail_does_not_enter_either_line(self):
        rows=[{'original_text':'first reference dialogue','translated_text':'first target dialogue'},
              {'original_text':'second reference dialogue','translated_text':'second target dialogue'}]
        words=[{'word':'first target dialogue','start':.1,'end':1.},
               {'word':'first reference dialogue','start':1.2,'end':3.},
               {'word':'second target dialogue','start':3.2,'end':4.},
               {'word':'second reference dialogue','start':4.2,'end':6.}]
        cuts,audit=batch_tts.exclude_unrequested_replay([(0,310),(310,650)],
            [{'start':.1,'end':1.,'word_indices':[0]},{'start':3.2,'end':4.,'word_indices':[2]}],
            words,rows,[0,1],100)
        self.assertEqual(cuts,[(0,110),(310,410)])
        self.assertEqual(len(audit),2)


    def test_acoustic_boundary_finds_quiet_before_inaccurate_asr_midpoint(self):
        rate=1000;values=np.full((2000,2),.1,np.float32);values[800:900]=0.
        cut,review=batch_tts.acoustic_cut(values,rate,.7,1.4)
        self.assertTrue(800<=cut<900)
        self.assertEqual(review['method'],'quiet_valley_between_asr_anchors')
        self.assertLess(float(np.max(abs(values[cut-5:cut+5]))),.001)

    def test_single_target_can_be_recovered_after_reference_replay_in_wrong_script(self):
        words=[{'word':'original reference dialogue','start':.1,'end':2.},
               {'word':'target in wrong script','start':2.3,'end':4.}]
        locations=[{'matched':False,'word_indices':[]}]
        result,audit=batch_tts.recover_single_after_reference(locations,words,np.full((5*SR,2),.1,np.float32),SR,
            {'original_text':'original reference dialogue','translated_text':'번역된 한국어 대사'})
        self.assertTrue(result[0]['matched']);self.assertEqual(result[0]['word_indices'],[1])
        self.assertEqual(audit[0]['method'],'remaining_target_after_verified_reference_replay')


    def test_late_asr_boundary_cannot_cut_off_target_onset(self):
        rate=1000;values=np.full((2000,2),.1,np.float32)
        values[800:1000]=.00001
        cut,review=batch_tts.acoustic_cut(values,rate,1.04,1.10)
        self.assertTrue(800<=cut<1000)
        self.assertEqual(review['method'],'waveform_quiet_valley_outside_asr_gap')
        # The next voice starts at 1.0, before ASR's 1.10 anchor.
        self.assertTrue(np.all(values[cut:1100][-100:]==.1))

    def test_uniform_loud_gap_is_never_reported_as_verified_silence(self):
        values=np.full((2000,2),.1,np.float32)
        cut,review=batch_tts.acoustic_cut(values,1000,.8,1.2)
        self.assertEqual(review['method'],'lowest_energy_unverified_asr_gap')
        self.assertTrue(800<=cut<=1200)


if __name__=='__main__':unittest.main()


def test_production_batches_send_sixty_seconds_without_count_limit_and_fill_remainder(tmp_path):
    import io
    rows=[line(i,'same-real-voice') for i in range(61)]
    clean=tmp_path/'clean.wav';sf.write(clean,np.full((61*SR,2),.1,np.float32),SR)
    folder=tmp_path/'pieces';folder.mkdir();payloads=[]
    class Reply(io.BytesIO):headers={}
    def reply(request,**kwargs):
        payload=json.loads(request.data)
        import base64
        data=payload['input_references'][0]['input_audio']['data'].split(',',1)[1]
        audio,rate=sf.read(io.BytesIO(base64.b64decode(data)),dtype='float32')
        count=len(payload['input'].splitlines())
        assert rate==SR and len(audio)==count*SR
        for j in range(count):
            np.testing.assert_allclose(audio[j*SR:(j+1)*SR],.1,atol=.0001)
        payloads.append(payload);return Reply(b'encoded audio')
    def split(audio,segments,indices,folder,cfg):
        return np.full((len(indices)*SR,2),.1,np.float32),[(i*SR,(i+1)*SR) for i in range(len(indices))],{'status':'test'}
    with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'17'}}),patch('batch_tts.split_generation',side_effect=split),patch('render.fit_speech',side_effect=lambda item,cfg:{'index':item['index']}):
        result=batch_tts.synthesize_speaker_batches(rows,clean,folder,{'language':'ko','key':'test','tts_provider_options':{'temperature':1.,'top_p':1.}},lambda *a:None)
    assert len(payloads)==2 and [r['index'] for r in result]==list(range(61))
    for payload in payloads:
        output=payload['input'].splitlines();reference=payload['input_references'][1]['text'].splitlines()
        assert len(output)==len(reference)
        for slot,(target,source) in enumerate(zip(output,reference)):
            assert '<|speaker:' not in target and '<|speaker:' not in source
        assert payload['instructions']==render.tts_instructions('ko')
        assert payload['provider']['options']['fish-audio']['condition_on_previous_chunks'] is True
        assert payload['provider']['options']['fish-audio']['temperature']==1.
    assert sorted(len(p['input'].splitlines()) for p in payloads)==[60,60]


def test_remainder_borrows_previous_same_voice_to_sixty_seconds_without_count_limit():
    rows=[{**line(i,'A'),'hi':i*SR+7*SR} for i in range(20)]
    groups=batch_tts.plan_speaker_batches(rows)
    assert list(map(len,groups))==[9,9,2]
    requests=batch_tts.speaker_request_groups(rows,groups)
    assert requests==[list(range(9)),list(range(9,18)),list(range(11,20))]
    assert all(len(g)==len(set(g)) for g in requests)
    assert sum(batch_tts.reference_seconds(rows[i]) for i in requests[-1])==63.
    rows[8]['speaker_id']='B'
    requests=batch_tts.speaker_request_groups(rows,[[16,17,18,19]])
    assert 8 not in requests[0] and all(rows[i]['speaker_id']=='A' for i in requests[0])


def test_voice_under_sixty_seconds_uses_all_available_without_repetition():
    rows=[line(i,'A') for i in range(3)]+[line(3,'unknown')]
    groups=batch_tts.plan_speaker_batches(rows)
    assert batch_tts.speaker_request_groups(rows,groups)==[[0,1,2],[3]]


def test_reference_duration_not_video_span_and_over_sixteen_lines_are_allowed():
    rows=[{**line(i,'A'),'reference_lo':i*SR,'reference_hi':i*SR+SR//2} for i in range(125)]
    groups=batch_tts.plan_speaker_batches(rows)
    assert list(map(len,groups))==[120,5]
    requests=batch_tts.speaker_request_groups(rows,groups)
    assert requests[-1]==list(range(5,125))
    assert all(sum(batch_tts.reference_seconds(rows[i]) for i in request)==60. for request in requests)


def test_borrowed_context_is_never_saved_over_its_owned_output(tmp_path):
    rows=[line(i,'A') for i in range(17)]
    clean=tmp_path/'clean.wav';sf.write(clean,np.full((17*SR,2),.1,np.float32),SR)
    fitted=[]
    def request(ref,segment,language,key,destination):
        destination.write_bytes(b'audio');return {'generated_seconds':16.}
    def split(audio,segments,indices,folder,cfg):
        return np.full((len(indices)*SR,2),.1,np.float32),[(i*SR,(i+1)*SR) for i in range(len(indices))],{'status':'test'}
    def fit(item,cfg):fitted.append(item['index']);return {'index':item['index']}
    with patch('render.request_audio',side_effect=request),patch('batch_tts.split_generation',side_effect=split),patch('render.fit_speech',side_effect=fit):
        batch_tts.synthesize_speaker_batches(rows,clean,tmp_path/'pieces',{'language':'ko','key':'test'},lambda *a:None)
    assert sorted(fitted)==list(range(17))  # Borrowed fifteen lines are not fitted/saved twice.


def test_local_phone_alignment_recovers_quiet_gap_outside_late_asr_end(tmp_path):
    values=np.full((3*SR,2),.1,np.float32);values[SR:round(1.3*SR)]=.000001
    previous={'start':.1,'end':1.4};current={'start':1.6,'end':2.8}
    hint={'status':'aligned','words':[{'label':'alpha','start':.08,'end':.98},{'label':'beta','start':1.28,'end':2.7}],
          'phones':[{'label':'a','start':.08,'end':.98},{'label':'sil','start':.98,'end':1.28},{'label':'b','start':1.28,'end':2.7}]}
    with patch('mfa_hint.alignment_hint',return_value=hint):
        boundary,review=batch_tts.recover_boundary_with_local_alignment(values,SR,previous,current,'alpha','beta',tmp_path/'hint.json',{'language':'en'})
    assert SR<=boundary<=round(1.3*SR) and review['verified']
    assert boundary<round(previous['end']*SR)


def test_local_alignment_never_authorizes_cut_through_a_phone_or_loud_audio(tmp_path):
    values=np.full((3*SR,2),.1,np.float32)
    previous={'start':.1,'end':1.4};current={'start':1.6,'end':2.8}
    hint={'status':'aligned','words':[{'label':'alpha','start':.08,'end':.98},{'label':'beta','start':1.28,'end':2.7}],
          'phones':[{'label':'a','start':.08,'end':.98},{'label':'sil','start':.98,'end':1.28},{'label':'b','start':1.28,'end':2.7}]}
    with patch('mfa_hint.alignment_hint',return_value=hint):
        boundary,review=batch_tts.recover_boundary_with_local_alignment(values,SR,previous,current,'alpha','beta',tmp_path/'hint.json',{'language':'en'})
    assert boundary is None and review['reason']=='no_acoustic_confirmation'
    hint['phones'].append({'label':'quiet_consonant','start':1.05,'end':1.15})
    values[SR:round(1.3*SR)]=0
    with patch('mfa_hint.alignment_hint',return_value=hint):
        boundary,review=batch_tts.recover_boundary_with_local_alignment(values,SR,previous,current,'alpha','beta',tmp_path/'hint.json',{'language':'en'})
    assert boundary is None and review['reason']=='phoneme_inside_proposed_gap'
