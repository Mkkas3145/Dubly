import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import speech
from semantic_alignment import ordered_path


class UntimedDialogueTests(unittest.TestCase):
    def test_flat_pairs_validate_structure_without_rewriting_expressive_text(self):
        pairs=['I... I knew it!','난… 알고 있었어!','Hello','안녕하세요']
        self.assertEqual(speech.parse_dialogue_pairs(json.dumps(pairs)),pairs)
        self.assertEqual(speech.parse_dialogue_pairs('[]'),[])
        for invalid in [['odd'],['text',2],['text',' '],{'segments':[]}]:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                speech.parse_dialogue_pairs(json.dumps(invalid))

    def test_actual_request_schema_only_allows_translations_and_counts_each_string(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sf.write(root/'gemini-full-16k.flac',np.zeros(16000),16000)
            answer=['안녕하세요','반갑습니다...','빠…빨라!'];updates=[]
            def response(payload,key,**kwargs):
                for count in range(4):kwargs['on_segments'](count)
                return kwargs['parser'](json.dumps(answer)),{}
            with patch.object(speech,'http_json',side_effect=response) as request:
                result=speech.transcribe(root/'unused.wav',root,root/'result.json',
                    {'key':'test','language':'ko'},lambda p,d:updates.append(d),
                    source_transcription={'words':[{'word':'SECRET_ASR'}]})
            payload=request.call_args.args[0];schema=payload['response_format']['json_schema']['schema']
            self.assertEqual(schema['type'],'array');self.assertEqual(schema['items']['type'],'string')
            self.assertNotIn('start',json.dumps(schema));self.assertNotIn('end',json.dumps(schema))
            self.assertNotIn('SECRET_ASR',json.dumps(payload))
            self.assertNotIn('timestamp_origin_seconds',json.dumps(payload))
            self.assertEqual(result['segments'],[dict(translated_text=text) for text in answer])
            self.assertIn('대사 3개 수신',updates[-2])
            self.assertEqual(result['translation_output'],'translated_strings_only')
            self.assertEqual(result['source_text_authority'],'scribe')
            self.assertNotIn('original_text',json.dumps(result['segments']))
            self.assertEqual(payload['max_tokens'],65536)

    def test_translation_array_accepts_odd_counts_and_preserves_expressive_delivery(self):
        self.assertEqual(speech.parse_translated_dialogue('["빠…빨라!","안녕","잠깐…"]'),['빠…빨라!','안녕','잠깐…'])
        for invalid in ['{}','[1]','[""]','[" "]','[["text"]]']:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):speech.parse_translated_dialogue(invalid)

    def test_global_order_can_choose_second_best_similarity_to_avoid_stealing_next_line(self):
        options=[[
            dict(first=10,last=12,score=.99),dict(first=2,last=4,score=.92)],
            [dict(first=8,last=10,score=.98)]]
        path=ordered_path(options)
        self.assertEqual([r['first'] for r in path],[2,8])

    def test_asr_word_ownership_is_unique_and_missing_dialogue_stays_missing(self):
        options=[[dict(first=1,last=2,score=.99)],[],
            [dict(first=2,last=3,score=.96),dict(first=4,last=5,score=.93)]]
        path=ordered_path(options)
        self.assertIsNone(path[1]);self.assertEqual(path[2]['first'],4)


if __name__=='__main__':unittest.main()
