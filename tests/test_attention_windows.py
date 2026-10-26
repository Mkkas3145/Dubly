import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import speech

class LegacyAudioOnlyInputTests(unittest.TestCase):
    def test_asr_is_not_sent_and_changes_do_not_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            sf.write(root/'gemini-full-16k.flac',np.zeros(3*16000,dtype=np.int16),16000)
            output=root/'segments.json';cfg={'key':'unused','language':'ko'}
            source={'words':[{'start':.2,'end':.6,'word':'SECRET_TRANSCRIPT','speaker':'SECRET_ID'}],
                    'text':'SECRET_TRANSCRIPT','language':'SECRET_LANGUAGE'}
            answer=speech.parse_transcript(json.dumps({'segments':[{'start':2.,'end':2.5,'original_text':'heard','translated_text':'translated'}],'excluded_intervals':[]}))
            with patch.object(speech,'http_json',return_value=(answer,{})) as request:
                result=speech.transcribe_audio_only_legacy(root/'unused.wav',root,output,cfg,lambda *_:None,source_transcription=source)
                request.assert_called_once()
                payload=request.call_args.args[0]
                context=json.loads(payload['messages'][1]['content'][0]['text'])
                self.assertEqual(set(context),{'target_language','audio_duration_seconds','scope','timestamp_origin_seconds'})
                for forbidden in ['SECRET_TRANSCRIPT','SECRET_ID','SECRET_LANGUAGE','attention_windows','hint_text']:
                    self.assertNotIn(forbidden,json.dumps(payload))
                self.assertEqual(result['speaker_authority'],'gemini_audio_only')
                self.assertNotIn('attention_policy',result)
                self.assertFalse((root/'gemini-attention-windows.json').exists())
            changed={'words':[{'start':1.,'end':2.,'word':'different words','speaker':'different ID'}]}
            with patch.object(speech,'http_json',side_effect=AssertionError('ASR changes must not invalidate audio-only cache')):
                speech.transcribe_audio_only_legacy(root/'unused.wav',root,output,cfg,lambda *_:None,source_transcription=changed)


class ProductionAudioOnlyTests(unittest.TestCase):
    def test_production_uses_audio_without_speaker_identity_or_asr(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            sf.write(root/'gemini-full-16k.flac',np.zeros(3*16000,dtype=np.int16),16000)
            answer=['첫 화자','둘째 화자']
            with patch.object(speech,'http_json',return_value=(answer,{})) as request:
                result=speech.transcribe(root/'unused.wav',root,root/'segments.json',
                    {'key':'unused','language':'ko'},lambda *_:None,
                    source_transcription={'words':[{'word':'SECRET_ASR_TEXT','speaker':'SECRET_ASR_ID'}]})
            payload=request.call_args.args[0]
            self.assertNotIn('SECRET_ASR',json.dumps(payload))
            self.assertTrue(payload['messages'][0]['content'].startswith(speech.PROMPT))
            schema=payload['response_format']['json_schema']['schema']
            self.assertEqual(schema['type'],'array')
            self.assertEqual(schema['items']['type'],'string')
            self.assertNotIn('start',json.dumps(schema))
            self.assertNotIn('temperature',payload)
            self.assertEqual(payload['messages'][1]['content'][1]['type'],'input_audio')
            self.assertEqual(payload['max_tokens'],speech.MAX_OUTPUT_TOKENS)
            self.assertFalse(result['asr_sent_to_llm'])
            self.assertEqual(result['speaker_authority'],'gemini_audio_only')
            self.assertEqual(result['segments'],[
                {'translated_text':'첫 화자'},
                {'translated_text':'둘째 화자'}])
            self.assertEqual(result['excluded_intervals'],[])

    def test_overlap_chain_keeps_all_lines_and_separate_turns(self):
        rows=[{'start':a,'end':b,'original_text':t,'translated_text':t}
            for a,b,t in [(2.,3.,'c'),(0.,1.,'a'),(.5,2.5,'b'),(3.,4.,'d')]]
        result=speech.normalize_audio_dialogue(rows)
        self.assertEqual(result,[{'start':0.,'end':3.,'original_text':'a\nb\nc','translated_text':'a\nb\nc'},rows[-1]])

if __name__=='__main__':unittest.main()
