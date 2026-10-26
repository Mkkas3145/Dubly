import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import native_timeline as native

class NativeTimelineTests(unittest.TestCase):
    def test_resampled_native_eof_is_clipped_to_real_samples_without_losing_speech(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);audio=p/'source.wav';frames=1777835
            sf.write(audio,np.zeros(frames),44100)
            transcript={'boundary_authority':'nemotron_eend_vc','segments':[
                {'start':39.2,'end':round(frames/44100*16000)/16000,'source_region_id':9,
                 'original_text':'original','translated_text':'translated'}]}
            result,measurements=native.references(audio,transcript,p/'measured')
            self.assertEqual(result['reference_windows'][0]['end_frame'],frames)
            self.assertEqual(result['segments'][0]['end'],frames/44100)
            self.assertTrue(measurements[0]['clipped_to_physical_audio'])
            self.assertEqual(measurements[0]['original_end_frame'],frames+1)
            self.assertEqual(result['segments'][0]['translated_text'],'translated')

    def test_native_reference_outside_file_has_specific_error_instead_of_empty_crop(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);audio=p/'source.wav';sf.write(audio,np.zeros(44100),44100)
            with self.assertRaisesRegex(RuntimeError,'42'):
                native.references(audio,{'segments':[{'start':2.,'end':3.,'source_region_id':42}]},p/'measured')

    def test_full_audio_input_policy_never_reuses_old_compacted_translation(self):
        from source_asr import INPUT_POLICY
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);audio=p/'source.wav';sf.write(audio,np.zeros(3*44100),44100)
            output=p/'segments.json';source={'words':[{'word':'原文','start':.2,'end':.7}]}
            with patch('speech.http_json',return_value=(['번역',None,None],{})):
                native.translate(audio,p,output,{'language':'ko','key':'unused'},lambda *a:None,
                    source_transcription=source,timeline=self.timeline())
            saved=json.loads((p/'native-translations.json').read_text('utf8'))
            completed=json.loads(output.read_text('utf8'))
            old_policy={'version':1,'input':'nemotron_speech_union','clock':'restore_original_per_word'}
            saved['signature']['source_asr_input_policy']=old_policy
            completed['source_asr_input_policy']=old_policy
            native.atomic_json(p/'native-translations.json',saved);native.atomic_json(output,completed)
            with patch('source_asr.transcribe_words',return_value=source) as asr, \
                 patch('speech.http_json',return_value=(['새 번역',None,None],{})) as api:
                result=native.translate(audio,p,output,{'language':'ko','key':'unused'},lambda *a:None,timeline=self.timeline())
            asr.assert_called_once();api.assert_called_once()
            self.assertEqual(result['source_asr_input_policy'],INPUT_POLICY)
            self.assertEqual(result['segments'][0]['translated_text'],'새 번역')

    def test_fish_split_cache_counts_final_regions_not_unsplit_native_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);audio=p/'source.wav';sf.write(audio,np.zeros(3*44100),44100)
            output=p/'segments.json';source={'words':[
                {'word':'前','start':.15,'end':.3,'fish_speaker':'A'},
                {'word':'後','start':.5,'end':.7,'fish_speaker':'B'}]}
            with patch('speech.http_json',return_value=(['앞','뒤',None,None],{})):
                result=native.translate(audio,p,output,{'language':'ko','key':'unused'},lambda *a:None,
                    source_transcription=source,timeline=self.timeline())
            self.assertEqual(len(result['source_regions']),4)
            with patch('speech.http_json',side_effect=AssertionError('No extra Gemini call')), \
                 patch('source_asr.transcribe_words',side_effect=AssertionError('No extra ASR call')):
                reused=native.translate(audio,p,output,{'language':'ko','key':'unused'},lambda *a:None,timeline=self.timeline())
            self.assertEqual(reused,result)

    def test_unresolved_speaker_policy_migration_reuses_paid_translation(self):
        from fish_speaker_guard import POLICY
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);audio=p/'source.wav';sf.write(audio,np.zeros(3*44100),44100)
            output=p/'segments.json';source={'words':[
                {'word':'前','start':.2,'end':.4,'fish_speaker':'A'},
                {'word':'後','start':.4,'end':.7,'fish_speaker':'B'}]}
            with patch('speech.http_json',return_value=(['대사',None,None],{})):
                result=native.translate(audio,p,output,{'language':'ko','key':'unused'},lambda *a:None,
                    source_transcription=source,timeline=self.timeline())
            self.assertNotIn('reference_isolated',result['segments'][0])
            self.assertEqual(result['segments'][0]['speaker_review'],'no_safe_fish_speaker_cut')
            self.assertNotIn('keep_original',result['segments'][0])
            old={'version':1,'identity':POLICY['identity'],'conflict':'split_at_quiet_word_gap_or_keep_original'}
            saved=json.loads((p/'native-translations.json').read_text('utf8'))
            saved['signature']['fish_speaker_policy']=old;result['fish_speaker_policy']=old
            result['segments'][0].update(keep_original=True,keep_original_reason='unsafe_fish_speaker_boundary')
            result['segments'][0].pop('reference_isolated',None)
            native.atomic_json(p/'native-translations.json',saved);native.atomic_json(output,result)
            with patch('speech.http_json',side_effect=AssertionError('No extra Gemini call')), \
                 patch('source_asr.transcribe_words',side_effect=AssertionError('No extra ASR call')):
                migrated=native.translate(audio,p,output,{'language':'ko','key':'unused'},lambda *a:None,timeline=self.timeline())
            self.assertEqual(migrated['segments'][0]['translated_text'],'대사')
            self.assertNotIn('reference_isolated',migrated['segments'][0])
            self.assertNotIn('keep_original',migrated['segments'][0])

    def test_whisper_merged_sentence_is_split_by_native_turns(self):
        timeline={'turns':[{'start':0.,'end':1.,'speaker':'N0'},
                           {'start':1.,'end':2.,'speaker':'N1'}]}
        source={'segments':[{'start':0.,'end':2.,'text':'私は 行きます どうぞ'}],
                'words':[{'word':'私は','start':.1,'end':.4},
                         {'word':'行きます','start':.5,'end':.9},
                         {'word':'。','start':1.,'end':1.},
                         {'word':'どうぞ','start':1.1,'end':1.8}]}
        audit=[]
        result=native.text_regions(native.regions(timeline['turns'],2.),timeline,source,audit)
        self.assertEqual([r['text'] for r in result],['私は 行きます 。','どうぞ'])
        self.assertEqual([r['speaker_ids'] for r in result],[['N0'],['N1']])
        self.assertEqual([a['region_id'] for a in audit],[0,0,0,1])
        self.assertTrue(all(set(r)=={'id','speaker_ids','text'} for r in result))

    def test_short_boundary_drift_is_recovered_without_crossing_long_silence(self):
        timeline={'turns':[{'start':0.,'end':1.,'speaker':'N0'},
                           {'start':3.,'end':4.,'speaker':'N1'}]}
        source={'words':[{'word':'はい','start':1.02,'end':1.12},
                         {'word':'幻','start':2.,'end':2.1},
                         {'word':'次','start':3.2,'end':3.5}]}
        audit=[]
        result=native.text_regions(native.regions(timeline['turns'],4.),timeline,source,audit)
        self.assertEqual([r['text'] for r in result],['はい','次'])
        self.assertEqual([a['region_id'] for a in audit],[0,None,1])

    def test_zero_duration_lexical_word_at_speaker_change_goes_to_new_turn(self):
        timeline={'turns':[{'start':0.,'end':1.,'speaker':'N0'},
                           {'start':1.,'end':2.,'speaker':'N1'}]}
        source={'words':[{'word':'先','start':.1,'end':.8},
                         {'word':'え','start':1.,'end':1.}]}
        result=native.text_regions(native.regions(timeline['turns'],2.),timeline,source)
        self.assertEqual([r['text'] for r in result],['先','え'])

    def save_native_fixture(self,path):
        native.atomic_json(path,{'turns':[{'start':.1125,'end':.8125,'speaker':'D0','raw_speaker':'0'}]})
        active=np.zeros((50,1));active[5:40]=1
        np.savez_compressed(path.with_suffix('.frames.npz'),votes=active,active=active,
            frame_start=0.,frame_step=.02,frame_duration=.025)
    def timeline(self):
        return {'identity':{'boundary_model':'DiariZen-test','source_sha256':'fixture'},'duration':3.,'turns':[{'start':.1,'end':.8,'speaker':'D1'},{'start':1.,'end':1.2,'speaker':'D2'},{'start':1.5,'end':2.6,'speaker':'D1'}]}
    def test_text_input_uses_native_speakers_and_never_sends_asr_times_or_speakers(self):
        timeline={'turns':[{'start':0.,'end':1.,'speaker':'N0'},{'start':1.,'end':2.,'speaker':'N1'}]}
        source={'words':[{'word':'crossing','start':.8,'end':1.4,'speaker_id':'wrong-asr-speaker'},
                         {'word':'next','start':1.5,'end':1.8,'speaker_id':'wrong-asr-speaker'}]}
        result=native.text_regions(native.regions(timeline['turns'],2.),timeline,source)
        self.assertEqual(result,[{'id':0,'speaker_ids':['N0'],'text':''},
                                 {'id':1,'speaker_ids':['N1'],'text':'crossing next'}])
        encoded=json.dumps(result)
        self.assertNotIn('start',encoded);self.assertNotIn('end',encoded)
        self.assertNotIn('wrong-asr-speaker',encoded)

    def test_native_boundaries_without_speaker_identity_in_llm_input(self):
        rows=native.regions(self.timeline()['turns'],3.)
        self.assertEqual([(r['start'],r['end']) for r in rows],[(.1,.8),(1.,1.2),(1.5,2.6)])
        self.assertTrue(all(set(r)=={'id','start','end'} for r in rows))
    def test_overlap_union_has_no_invented_midpoint_and_touching_stays_separate(self):
        turns=[{'start':0.,'end':1.,'speaker':'VA0'},{'start':.7,'end':1.4,'speaker':'VA1'},{'start':1.4,'end':2.,'speaker':'VA2'}]
        rows=native.regions(turns,2.)
        self.assertEqual([(r['id'],r['start'],r['end']) for r in rows],[(0,0.,1.4),(1,1.4,2.)])
        self.assertNotIn('speaker_ids',rows[0])
    def test_padded_segmentation_is_clipped_to_physical_audio_only(self):
        rows=native.regions([{'start':-.02,'end':.4,'speaker':'VA0'},{'start':.8,'end':1.1325,'speaker':'VA1'},{'start':1.2,'end':1.3,'speaker':'VA2'}],1.)
        self.assertEqual([(r['start'],r['end']) for r in rows],[(0.,.4),(.8,1.)])
    def test_null_is_a_slot_not_a_missing_entry(self):
        self.assertEqual(native.parse('["앞",null,"뒤"]',3),['앞',None,'뒤'])
        for content in ['["앞","뒤"]','["앞","",null]','["앞",false,"뒤"]','{"text":"앞"}','[["원문","번역"],null,null]']:
            with self.subTest(content=content),self.assertRaises(ValueError):native.parse(content,3)
        from speech import SegmentStreamCounter
        counter=SegmentStreamCounter()
        self.assertEqual(counter.feed('["앞",nu'),1)
        self.assertEqual(counter.feed('ll,"뒤"]'),3)

    def test_translation_keeps_same_speaker_turns_for_distinct_tts_tags(self):
        turns=[{'start':0.,'end':1.,'speaker':'A'},{'start':1.4,'end':2.,'speaker':'A'}]
        self.assertEqual(len(native.regions(turns,3)),2)

    def test_one_audio_translation_null_keeps_later_region_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);audio=p/'source.wav';sf.write(audio,np.zeros((3*44100,2)),44100)
            source={'words':[{'word':'原文','start':.2,'end':.7},{'word':'後','start':1.6,'end':2.5}],'language':'ja'}
            with patch('speech.http_json',return_value=(['앞',None,'뒤'],{})) as api,patch('speech.compressed_audio',return_value=audio),patch('source_asr.transcribe_words',side_effect=AssertionError('Completed translation must skip paid ASR')) as source_api:
                result=native.translate(audio,p,p/'segments.json',{'language':'ko','key':'unused'},lambda *a:None,source_transcription=source,timeline=self.timeline())
                native.translate(audio,p,p/'segments.json',{'language':'ko','key':'unused'},lambda *a:None,source_transcription=source,timeline=self.timeline())
                cached=native.translate(audio,p,p/'segments.json',{'language':'ko','key':'unused'},lambda *a:None,timeline=self.timeline())
                self.assertEqual(cached,result)
                self.assertEqual(api.call_count,1)
                source_api.assert_not_called()
                payload=api.call_args.args[0]
                self.assertEqual(payload['response_format'],native.region_response_format([0,1,2]))
                content=payload['messages'][1]['content']
                self.assertEqual(len(content),1);self.assertNotIn('temperature',payload)
                self.assertEqual(content[0]['type'],'text')
                context=json.loads(content[0]['text'])
                self.assertTrue(all(set(r)=={'id','speaker_ids','text'} for r in context['regions']))
                self.assertEqual(context['regions'][0]['text'],source['words'][0]['word'])
                self.assertEqual([r['speaker_ids'] for r in context['regions']],[['D1'],['D2'],['D1']])
                self.assertNotIn('input_audio',json.dumps(payload))
                self.assertFalse((p/'region-audio-v1').exists())
            self.assertEqual([(s['source_region_id'],s['start'],s['end']) for s in result['segments']],[(0,.1,.8),(2,1.5,2.6)])
            self.assertEqual(result['excluded_intervals'][0]['source_region_id'],1)
            self.assertEqual(result['segments'][0]['original_text'],'原文');self.assertTrue(result['asr_used'])
            self.assertEqual(result['segments'][1]['original_text'],'後')
            self.assertEqual(result['translation_slots'],['앞',None,'뒤'])
            self.assertEqual(result['original_text_source'],'source_asr_native_region_assignment')
            self.assertEqual(result['speaker_authority'],'nemotron_fish_boundary_hints')
            self.assertTrue(all('speaker_id' not in s for s in result['segments']))
            from reference_timing import refine_references
            from render import prepare_segments
            with patch('reference_timing.source_evidence',side_effect=AssertionError('Legacy ASR alignment must not run')):
                measured,rows=refine_references(audio,result,p/'references')
            prepared=prepare_segments(measured,3*44100,source)
            self.assertEqual([(r['lo'],r['hi']) for r in prepared],[(4410,35280),(66150,114660)])
            self.assertTrue(all(not s['natural_reference_margin'] for s in prepared))
            self.assertEqual(measured['unmatched_segments'],[])

    def test_prepare_runs_only_diarizen_and_caches_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);audio=p/'source.wav';sf.write(audio,np.zeros(44100),44100)
            def worker(script,args,log,progress):
                self.assertEqual(script,'diarizen_worker.py')
                self.save_native_fixture(args[1])
            with patch.object(native,'command'),patch.object(native,'run_worker',side_effect=worker) as run:
                first=native.prepare(audio,p/'diarizen-native-v2')
                self.assertEqual(native.prepare(audio,p/'diarizen-native-v2'),first)
                self.assertEqual(run.call_count,1)
            self.assertNotIn('speaker_model',first['identity'])
            self.assertEqual(first['turns'][0]['speaker'],'D0')

    def test_switching_source_model_invalidates_completed_translation_cache(self):
        from source_asr import MODEL,CACHE_DIRECTORY
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);audio=p/'source.wav';sf.write(audio,np.zeros(3*44100),44100)
            output=p/'segments.json'
            source={'words':[{'word':'原文','start':.2,'end':.7}]}
            with patch('speech.http_json',return_value=(['번역',None,None],{})):
                native.translate(audio,p,output,{'language':'ko','key':'unused'},lambda *a:None,
                    source_transcription=source,timeline=self.timeline())
            cache=p/'native-translations.json'
            saved=json.loads(cache.read_text('utf8'));completed=json.loads(output.read_text('utf8'))
            saved['signature']['source_asr_model']='openai/whisper-large-v3-turbo'
            completed['source_asr_model']='openai/whisper-large-v3-turbo'
            native.atomic_json(cache,saved);native.atomic_json(output,completed)
            with patch('source_asr.transcribe_words',return_value=source) as asr, \
                 patch('speech.http_json',return_value=(['새 번역',None,None],{})) as api:
                result=native.translate(audio,p,output,{'language':'ko','key':'unused'},lambda *a:None,
                    timeline=self.timeline())
            asr.assert_called_once();api.assert_called_once()
            self.assertEqual(asr.call_args.args[:2],(audio,p/CACHE_DIRECTORY))
            self.assertEqual(result['source_asr_model'],MODEL)
            self.assertEqual(result['segments'][0]['translated_text'],'새 번역')

    def test_region_ids_not_json_property_order_determine_mapping(self):
        self.assertEqual(native.parse_region_keys('{"2":"뒤","0":"앞","1":null}',[0,1,2]),['앞',None,'뒤'])
        for content in ['{"0":"앞"}','{"0":null,"1":null,"3":null}','{"0":null,"0":null,"1":null,"2":null}','{"0":["前","앞"],"1":null,"2":null}']:
            with self.subTest(content=content),self.assertRaises(ValueError):native.parse_region_keys(content,[0,1,2])
        from speech import SegmentStreamCounter
        counter=SegmentStreamCounter()
        self.assertEqual(counter.feed('{"0":"'),0)
        self.assertEqual(counter.feed('앞","1":nu'),1)
        self.assertEqual(counter.feed('ll,"2":"뒤"}'),3)

    def test_old_cache_reuses_native_diarizen_only_not_va_turns(self):
        from install_diarizen import MODEL,REVISION
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);audio=p/'source.wav';sf.write(audio,np.zeros(44100),44100)
            legacy=p/'diarizen-va-v1';legacy.mkdir()
            identity={'version':1,'source_sha256':native.digest(audio),'boundary_model':MODEL,'boundary_revision':REVISION,'speaker_model':'VA'}
            turns=[{'start':.1125,'end':.8125,'speaker':'D0','raw_speaker':'0'}]
            native.atomic_json(legacy/'diarizen-identity.json',identity)
            self.save_native_fixture(legacy/'diarizen.json')
            native.atomic_json(legacy/'timeline.json',{'identity':identity,'turns':[{'start':0.,'end':1.,'speaker':'VA0'}]})
            with patch.object(native,'command',side_effect=AssertionError('cached boundaries')),patch.object(native,'run_worker',side_effect=AssertionError('VA must not run')):
                result=native.prepare(audio,p/'diarizen-native-v2')
            for actual,expected in zip(result['turns'],turns):
                self.assertAlmostEqual(actual['start'],expected['start']);self.assertAlmostEqual(actual['end'],expected['end'])
                self.assertEqual(actual['speaker'],expected['speaker'])
            self.assertEqual(result['identity']['stability_version'],1)

    def test_old_raw_cache_without_frame_votes_is_refreshed(self):
        from install_diarizen import MODEL,REVISION
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);audio=p/'source.wav';sf.write(audio,np.zeros(44100),44100)
            folder=p/'native';folder.mkdir()
            self.save_native_fixture(folder/'diarizen.json');(folder/'diarizen.frames.npz').unlink()
            native.atomic_json(folder/'diarizen-identity.json',{'source_sha256':native.digest(audio),'boundary_model':MODEL,'boundary_revision':REVISION})
            def worker(script,args,log,progress):
                self.assertEqual(script,'diarizen_worker.py');self.save_native_fixture(args[1])
            with patch.object(native,'command'),patch.object(native,'run_worker',side_effect=worker) as run:
                result=native.prepare(audio,folder)
            self.assertEqual(run.call_count,1)
            self.assertIn('stabilization',result)

class StrictRegionOutputTests(unittest.TestCase):
    def test_compact_schema_fixes_all_region_ids_and_disallows_extras(self):
        from jsonschema import Draft202012Validator
        schema=native.region_response_format(range(231))['json_schema']['schema']
        validate=Draft202012Validator(schema)
        answer={str(i):None for i in range(231)};answer['0']='번역'
        self.assertFalse(list(validate.iter_errors(answer)))
        self.assertTrue(list(validate.iter_errors({**answer,'0':['原文','번역']})))
        self.assertTrue(list(validate.iter_errors({**answer,'0':{'original':'原文','translation':'번역'}})))
        self.assertLess(len(json.dumps(schema)),500)
        self.assertEqual(schema,native.region_response_format(range(1000))['json_schema']['schema'])
        with self.assertRaises(ValueError):native.parse_region_keys(json.dumps({**answer,'231':None}),range(231))
        with self.assertRaises(ValueError):native.parse_region_keys(json.dumps({k:v for k,v in answer.items() if k!='230'}),range(231))
    def test_final_parser_reports_unexpected_region_ids(self):
        with self.assertRaisesRegex(ValueError,'unexpected'):
            native.parse_region_keys('{"0":null,"1":null,"2":null}',[0,1])

if __name__=='__main__':unittest.main()
