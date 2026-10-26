import json,unittest
import speech


class LlmVoiceIdentityTests(unittest.TestCase):
    def test_compact_schema_omits_voice_identity_and_prompt_disables_grouping(self):
        schema=speech.CompactTranscript.model_json_schema()
        fields=schema['$defs']['DialogueSegment']['properties']
        self.assertEqual(set(fields),{'start','end','original_text','translated_text'})
        self.assertIn('Do not analyze or group recurring speaker identities',speech.PROMPT)
        self.assertNotIn('Infer who is speaking',speech.PROMPT)

    def test_old_tagged_results_remain_readable_but_normalized_output_has_no_ids(self):
        rows=[dict(start=0.,end=.5,original_text='a',translated_text='A',speaker_id='speaker_1'),
              dict(start=.6,end=1.,original_text='b',translated_text='B',speaker_id='speaker_2'),
              dict(start=1.1,end=1.5,original_text='c',translated_text='C',speaker_id='unknown')]
        parsed=speech.parse_transcript(json.dumps({'segments':rows,'excluded_intervals':[]}))
        result=speech.normalize_audio_dialogue([s.model_dump() for s in parsed.segments])
        self.assertEqual([s.speaker_id for s in parsed.segments],['speaker_1','speaker_2','unknown'])
        self.assertTrue(all('speaker_id' not in r for r in result))

    def test_overlapping_distinct_voices_are_not_assigned_to_one_of_them(self):
        rows=[dict(start=0.,end=1.,original_text='a',translated_text='A',speaker_id='speaker_1'),
              dict(start=.5,end=1.5,original_text='b',translated_text='B',speaker_id='speaker_2')]
        result=speech.normalize_audio_dialogue(rows)
        self.assertEqual(len(result),1)
        self.assertNotIn('speaker_id',result[0])


if __name__=='__main__':unittest.main()
