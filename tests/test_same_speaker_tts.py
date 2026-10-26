import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from same_speaker_tts import group_transcript,tagged_request,MAX_PARTS,restore_turns_for_alignment
import render

class SameSpeakerTTSTests(unittest.TestCase):
    def test_shared_reference_keeps_all_reference_tags_but_outputs_only_selected_slot(self):
        parts=[{'original_text':f'原文{i}','translated_text':f'번역{i}'} for i in range(3)]
        for slot in range(3):
            reference,target=tagged_request({'_tts_parts':parts,'_tts_output_slot':slot})
            self.assertEqual(reference,' '.join(f'<|speaker:{i}|>原文{i}' for i in range(3)))
            self.assertEqual(target,f'<|speaker:{slot}|>번역{slot}')
        for slot in [-1,3,True,'1']:
            with self.assertRaises(ValueError):tagged_request({'_tts_parts':parts,'_tts_output_slot':slot})

    def fixture(self,voices):
        rows=[{'id':i,'start':i*1.5,'end':i*1.5+1.} for i in range(len(voices))]
        segments=[{**{k:r[k] for k in ('start','end')},'source_region_id':r['id'],
                   'original_text':f'原文{i}','translated_text':f'번역{i}'} for i,r in enumerate(rows)]
        timeline={'turns':[{**r,'speaker':voices[i]} for i,r in enumerate(rows)]}
        return {'source_regions':rows,'segments':segments,'excluded_intervals':[]},timeline

    def test_distinct_tags_and_16_limit_preserve_endpoints(self):
        transcript,timeline=self.fixture(['A']*18)
        result=group_transcript(transcript,timeline)
        self.assertEqual([len(s['_tts_parts']) for s in result['segments']],[16,2])
        self.assertEqual((result['segments'][0]['start'],result['segments'][0]['end']),(0.,23.5))
        original,translated=tagged_request(result['segments'][0])
        for i in range(MAX_PARTS):
            self.assertIn(f'<|speaker:{i}|>原文{i}',original)
            self.assertIn(f'<|speaker:{i}|>번역{i}',translated)
        self.assertNotIn('<|speaker:16|>',original)
        self.assertEqual(len(transcript['segments']),18)
        self.assertEqual(group_transcript(result,timeline),result)

    def test_collects_returning_speaker_and_ignores_excluded_audio(self):
        transcript,timeline=self.fixture(['A','A','B','A','A'])
        result=group_transcript(transcript,timeline)
        self.assertEqual([g['source_region_ids'] for g in result['segments']],[[0,1,3,4],[2]])
        transcript,timeline=self.fixture(['A']*3);removed=transcript['segments'].pop(1)
        transcript['excluded_intervals']=[{'start':removed['start'],'end':removed['end']}]
        self.assertEqual(group_transcript(transcript,timeline)['segments'][0]['source_region_ids'],[0,2])
        transcript,timeline=self.fixture(['A']*3)
        timeline['turns'].append({'start':2.4,'end':2.6,'speaker':'B'})
        self.assertEqual([g['source_region_ids'] for g in group_transcript(transcript,timeline)['segments']],[[0,2],[1]])
        transcript,timeline=self.fixture(['unknown']*2)
        self.assertEqual(len(group_transcript(transcript,timeline)['segments']),2)

    def test_same_reference_request_receives_matching_tags(self):
        transcript,timeline=self.fixture(['A']*3)
        segment=group_transcript(transcript,timeline)['segments'][0];received=[]
        class Response(io.BytesIO):headers={}
        def reply(req,**kwargs):received.append(json.loads(req.data));return Response(b'valid audio')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'ref.wav').write_bytes(b'fixed reference')
            with patch.object(render.urllib.request,'urlopen',side_effect=reply),patch.object(render,'probe',return_value={'format':{'duration':'3'}}):
                render.request_audio(p/'ref.wav',segment,'ko','private-key',p/'tts.mp3')
        original,translated=tagged_request(segment)
        self.assertEqual(received[0]['input'],translated)
        self.assertEqual(received[0]['input_references'][1]['text'],original)
        self.assertEqual(received[0]['instructions'],render.tts_instructions('ko'))
        self.assertEqual(sum(r['type']=='input_audio' for r in received[0]['input_references']),1)
        self.assertEqual(received[0]['provider']['options']['fish-audio']['temperature'],1.)

    def test_native_reference_pipeline_preserves_individual_windows(self):
        import native_timeline,soundfile as sf,numpy as np
        transcript,timeline=self.fixture(['A','B','A'])
        restored=restore_turns_for_alignment(group_transcript(transcript,timeline))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);sf.write(p/'clean.wav',np.full((6*44100,2),.1,dtype=np.float32),44100)
            (p/'references').mkdir()
            measured,_=native_timeline.references(p/'clean.wav',restored,p/'references')
            segments=render.prepare_segments(measured,6*44100)
            self.assertEqual([(s['lo'],s['hi']) for s in segments],[(0,44100),(66150,110250),(132300,176400)])
            self.assertEqual(restored['tts_group_plan'],[[0,2],[1]])

    def test_restore_original_turns_keeps_group_membership(self):
        transcript,timeline=self.fixture(['A','A','B'])
        result=restore_turns_for_alignment(group_transcript(transcript,timeline))
        self.assertEqual(result['tts_group_plan'],[[0,1],[2]])
        self.assertEqual([s['_tts_group'] for s in result['segments']],[0,0,1])
        self.assertEqual([s['_tts_slot'] for s in result['segments']],[0,1,0])
        self.assertEqual([(s['start'],s['end']) for s in result['segments']],[(0,1),(1.5,2.5),(3,4)])

    def test_long_gaps_and_sixteen_per_returning_voice_preserve_video_order(self):
        transcript,timeline=self.fixture(['A','B']*18)
        for i in range(36):
            for row in (transcript['segments'][i],transcript['source_regions'][i],timeline['turns'][i]):
                row.update(start=i*60.,end=i*60.+1.)
        grouped=group_transcript(transcript,timeline)
        self.assertEqual([len(g['_tts_parts']) for g in grouped['segments']],[16,16,2,2])
        restored=restore_turns_for_alignment(grouped)
        self.assertEqual(restored['tts_group_plan'],[list(range(0,32,2)),list(range(1,32,2)),[32,34],[33,35]])
        self.assertEqual([s['start'] for s in restored['segments']],[i*60. for i in range(36)])
        self.assertEqual([s['source_region_id'] for s in restored['segments']],list(range(36)))

if __name__=='__main__':unittest.main()
