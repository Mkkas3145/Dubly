from pathlib import Path
import base64
import io
import json
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import speech


def parsed(segments=None):
    return speech.Transcript.model_validate({'source_language':'ja','speakers':[{'id':'speaker_1','description':'voice'}],
        'segments':segments or [],'excluded_intervals':[]})


def segment(start,end,text):
    return {'start':start,'end':end,'original_text':text,'translated_text':text,'speaker_id':'speaker_1','emotion':'neutral'}


class LegacyWholeAudioTests(unittest.TestCase):
    def test_oversize_request_retries_smaller_full_audio_without_splitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            sf.write(root/'gemini-full-16k.flac',np.zeros(16000,dtype=np.int16),16000,subtype='PCM_16')
            with patch.object(speech,'http_json',side_effect=[speech.AudioRequestTooLarge('413'),(parsed(),{})]) as request:
                result=speech.transcribe_audio_only_legacy(root/'unused.wav',root,root/'segments.json',{'key':'test','language':'ko'},lambda *a:None)
            self.assertEqual(request.call_count,2)
            self.assertTrue((root/'gemini-full-16k-64k.mp3').is_file())
            self.assertTrue((root/'gemini-full-16k-48k.mp3').is_file())
            self.assertTrue((root/'gemini-audio-64k-size-error.json').is_file())
            self.assertEqual(result['duration_seconds'],1.)
            self.assertEqual(result['request_count'],1)

    def test_compact_response_needs_no_roster_language_or_exclusion_reason(self):
        data={'segments':[segment(.1,.5,'meaningful')],
            'excluded_intervals':[{'start':.6,'end':.9}]}
        parsed=speech.parse_transcript(json.dumps(data))
        self.assertEqual(parsed.segments[0].original_text,'meaningful')
        self.assertEqual(parsed.speakers[0].id,'speaker_1')
        self.assertIsNone(parsed.excluded_intervals[0].reason)
        schema=speech.CompactTranscript.model_json_schema()
        self.assertEqual(set(schema['properties']),{'segments','excluded_intervals'})
        self.assertEqual(set(schema['$defs']['ExcludedTime']['properties']),{'start','end'})
        self.assertEqual(set(schema['$defs']['DialogueSegment']['properties']),{'start','end','original_text','translated_text'})
        data['excluded_intervals'][0]['end']=.3
        with self.assertRaises(ValueError):speech.parse_transcript(json.dumps(data))

    def test_four_field_response_keeps_llm_boundaries_without_local_short_merging(self):
        compact={'segments':[
            {'start':.1,'end':1.,'original_text':'first','translated_text':'first'},
            {'start':1.1,'end':1.6,'original_text':'next','translated_text':'next'}],
            'excluded_intervals':[]}
        result=speech.parse_transcript(json.dumps(compact))
        final=speech.normalize([s.model_dump() for s in result.segments],[])
        self.assertEqual(final,compact['segments'])

    def test_invalid_response_is_preserved_before_retrying_schema_validation(self):
        good=parsed().model_dump()
        bad={**good,'segments':[{**segment(1.,2.,'dialogue'),'emotion':'invalid-emotion'}]}
        def response(content):
            return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}]}).encode())
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            with patch.object(speech.urllib.request,'urlopen',side_effect=[response(bad),response(good)]) as request,patch.object(speech.time,'sleep'):
                result,_=speech.http_json({'model':speech.GEMINI},'test',response_directory=folder)
            self.assertEqual(request.call_count,2)
            self.assertEqual(result.segments,[])
            errors=list(folder.glob('*.validation.json'))
            self.assertEqual(len(errors),1)
            error=json.loads(errors[0].read_text())
            self.assertEqual(error['finish_reason'],'stop')
            self.assertEqual(error['errors'][0]['loc'],['segments',0,'emotion'])
            self.assertTrue(errors[0].with_name(errors[0].name.replace('.validation.json','.json')).exists())

    def test_embedded_provider_500_retries_and_preserves_partial_response(self):
        bad={'error':{'code':500,'message':'Internal Server Error'},'choices':[{'finish_reason':'error','message':{'content':'{"segments":['}}]}
        good={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(parsed().model_dump())}}]}
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            responses=[io.BytesIO(json.dumps(item).encode()) for item in (bad,good)]
            with patch.object(speech.urllib.request,'urlopen',side_effect=responses) as request,patch.object(speech.time,'sleep'):
                answer,_=speech.http_json({},'test',response_directory=folder,on_segments=lambda n:None)
            self.assertEqual(request.call_count,2)
            self.assertEqual(answer.segments,[])
            records=[json.loads(p.read_text()) for p in folder.glob('gemini-response-*.json')]
            self.assertEqual(len(records),2)
            self.assertTrue(any(r.get('error',{}).get('code')==500 for r in records))

    def test_embedded_provider_errors_are_bounded_and_nontransient_errors_not_retried(self):
        for code,attempts in [(500,3),(401,1)]:
            def response(*args,**kwargs):
                return io.BytesIO(json.dumps({'error':{'code':str(code)}}).encode())
            with patch.object(speech.urllib.request,'urlopen',side_effect=response) as request,patch.object(speech.time,'sleep'):
                with self.assertRaisesRegex(RuntimeError,str(code)):
                    speech.http_json({},'test')
                self.assertEqual(request.call_count,attempts)

    def test_stream_preserves_native_filter_reason_and_does_not_retry(self):
        event={'choices':[{'index':0,'finish_reason':'content_filter','native_finish_reason':'SAFETY','error':{'code':403,'message':'SAFETY'},'delta':{'content':'{"segments":['}}]}
        raw=('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode()
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            with patch.object(speech.urllib.request,'urlopen',return_value=io.BytesIO(raw)) as request:
                with self.assertRaisesRegex(RuntimeError,'콘텐츠 필터'):
                    speech.http_json({},'test',response_directory=folder,on_segments=lambda n:None)
            self.assertEqual(request.call_count,1)
            saved=json.loads(next(folder.glob('gemini-response-*.json')).read_text())
            self.assertEqual(saved['choices'][0]['native_finish_reason'],'SAFETY')
            self.assertEqual(saved['choices'][0]['error']['code'],403)

    def test_length_finish_is_reported_as_truncation_and_saved(self):
        response=io.BytesIO(json.dumps({'choices':[{'finish_reason':'length','message':{'content':'{"segments":['}}]}).encode())
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            with patch.object(speech.urllib.request,'urlopen',return_value=response) as request:
                with self.assertRaisesRegex(RuntimeError,'출력 한도'):
                    speech.http_json({'model':speech.GEMINI},'test',response_directory=folder)
            self.assertEqual(request.call_count,1)
            self.assertEqual(len(list(folder.glob('gemini-response-*.json'))),1)

    def test_old_fragment_translation_cache_cannot_bypass_updated_eligibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);audio=root/'gemini-full-16k.flac'
            sf.write(audio,np.zeros(16000,dtype=np.int16),16000,subtype='PCM_16')
            old_segment={**segment(.1,.5,'な'),'translated_text':'뭐,'}
            old=parsed([old_segment])
            identity={'transcription_mode':'full_audio','transcription_version':2,'model':speech.GEMINI,
                'target_language':'ko','prompt_sha256':'b6fce40e930f9ff276462657c2e57c73f732a9c2461771a3a579f395b4b5f122',
                'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),'duration_seconds':1.}
            (root/'segments.json').write_text(json.dumps({**identity,'segments':[old_segment]}),encoding='utf-8')
            (root/'gemini-full.json').write_text(json.dumps({**identity,'result':old.model_dump(),'usage':{}}),encoding='utf-8')
            revised=speech.Transcript.model_validate({'source_language':'ja','speakers':[], 'segments':[],
                'excluded_intervals':[{'start':.1,'end':.5,'reason':'nonlexical_vocalization'}]})
            with patch.object(speech,'http_json',return_value=(revised,{})) as request:
                result=speech.transcribe_audio_only_legacy(root/'unused.wav',root,root/'segments.json',{'key':'test','language':'ko'},lambda *a:None)
            request.assert_called_once()
            self.assertEqual(result['transcription_version'],speech.TRANSCRIPTION_VERSION)
            self.assertEqual(result['segments'],[])
            self.assertEqual(result['excluded_intervals'],[{'start':.1,'end':.5,'reason':'nonlexical_vocalization'}])

    def test_nonlexical_exclusion_survives_transcription_for_mixing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);sf.write(root/'gemini-full-16k.flac',np.zeros(3*16000,dtype=np.int16),16000,subtype='PCM_16')
            answer=speech.Transcript.model_validate({'source_language':'ja','speakers':[{'id':'speaker_1','description':'voice'}],
                'segments':[segment(0.,.8,'meaningful reply'),segment(1.5,2.5,'dialogue')],
                'excluded_intervals':[{'start':1.,'end':1.3,'reason':'nonlexical_vocalization'}]})
            with patch.object(speech,'http_json',return_value=(answer,{})):
                result=speech.transcribe_audio_only_legacy(root/'unused.wav',root,root/'segments.json',{'key':'test','language':'ko'},lambda *args:None)
            self.assertEqual(result['excluded_intervals'],[{'start':1.,'end':1.3,'reason':'nonlexical_vocalization'}])
            self.assertEqual(len(result['segments']),2)

    def test_long_recording_is_sent_once_with_all_samples_and_absolute_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);samples=np.zeros(301*16000,dtype=np.int16)
            samples[0]=1111;samples[120*16000]=2222;samples[240*16000]=-3333;samples[-1]=-4444
            sf.write(root/'gemini-full-16k.flac',samples,16000,subtype='PCM_16')
            # Old chunk response and old combined transcript must not satisfy a whole-audio request.
            (root/'gemini-0000.json').write_text('{"result":"old chunk"}')
            (root/'segments.json').write_text('{"segments":[],"model":"old"}')
            answer=parsed([segment(119.,122.,'경계를 지나는 대사'),segment(299.,300.9,'마지막 대사')])
            with patch.object(speech,'http_json',return_value=(answer,{'cost':.1})) as request:
                result=speech.transcribe_audio_only_legacy(root/'unused.wav',root,root/'segments.json',{'key':'test','language':'ko'},lambda *args:None)
            request.assert_called_once()
            payload=request.call_args.args[0];parts=payload['messages'][1]['content']
            context=json.loads(parts[0]['text']);self.assertEqual(context['audio_duration_seconds'],301)
            self.assertNotIn('asr_reference',context)
            self.assertNotIn('Scribe',payload['messages'][0]['content'])
            self.assertEqual(result['speaker_authority'],'gemini_audio_only')
            wire=payload['response_format']['json_schema']['schema']['$defs']['DialogueSegment']
            self.assertEqual(set(wire['properties']),{'start','end','original_text','translated_text'})
            for definition in ('DialogueSegment','ExcludedTime'):
                for field in ('start','end'):
                    self.assertEqual(payload['response_format']['json_schema']['schema']['$defs'][definition]['properties'][field]['maximum'],301)
            self.assertEqual(context['timestamp_origin_seconds'],0);self.assertEqual(context['scope'],'entire_recording')
            audio=parts[1]['input_audio'];self.assertEqual(audio['format'],'mp3')
            restored,rate=sf.read(io.BytesIO(base64.b64decode(audio['data'])),dtype='int16')
            self.assertEqual(rate,16000);self.assertEqual(len(restored),len(samples))
            for position in [0,120*16000,240*16000,len(samples)-1]:
                self.assertGreater(np.max(np.abs(restored[max(0,position-100):position+101])),0)
            self.assertEqual([(s['start'],s['end']) for s in result['segments']],[(119.,122.),(299.,300.9)])
            self.assertEqual(result['transcription_mode'],'full_audio');self.assertEqual(result['request_count'],1)
            self.assertEqual(payload['max_tokens'],65536)
            self.assertNotIn('excerpt',payload['messages'][0]['content'])

    def test_complete_response_cache_reused_after_result_file_is_lost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);sf.write(root/'gemini-full-16k.flac',np.zeros(16000,dtype=np.int16),16000,subtype='PCM_16')
            cfg={'key':'test','language':'ko'};output=root/'segments.json'
            with patch.object(speech,'http_json',return_value=(parsed(),{})) as request:
                first=speech.transcribe_audio_only_legacy(root/'unused.wav',root,output,cfg,lambda *args:None)
                self.assertEqual(request.call_count,1)
            output.unlink()
            with patch.object(speech,'http_json') as request:
                second=speech.transcribe_audio_only_legacy(root/'unused.wav',root,output,cfg,lambda *args:None)
                request.assert_not_called()
            self.assertEqual(first,second)

    def test_asr_conditioned_cache_is_not_reused_for_audio_only_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);sf.write(root/'gemini-full-16k.flac',np.zeros(16000,dtype=np.int16),16000,subtype='PCM_16')
            cfg={'key':'test','language':'ko'};output=root/'segments.json'
            with patch.object(speech,'http_json',return_value=(parsed(),{})):
                speech.transcribe_audio_only_legacy(root/'unused.wav',root,output,cfg,lambda *args:None)
            for path in [output,root/'gemini-full.json']:
                saved=json.loads(path.read_text(encoding='utf-8'))
                saved['speaker_authority']='gemini_audio_with_scribe_reference'
                path.write_text(json.dumps(saved),encoding='utf-8')
            with patch.object(speech,'http_json',return_value=(parsed(),{})) as request:
                result=speech.transcribe_audio_only_legacy(root/'unused.wav',root,output,cfg,lambda *args:None)
                request.assert_called_once()
                self.assertEqual(result['speaker_authority'],'gemini_audio_only')

    def test_truncated_response_errors_without_splitting_or_saving_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);sf.write(root/'gemini-full-16k.flac',np.zeros(16000,dtype=np.int16),16000,subtype='PCM_16')
            response=io.BytesIO(json.dumps({'choices':[{'finish_reason':'length','message':{'content':'{"segments":['}}]}).encode())
            with patch.object(speech.urllib.request,'urlopen',return_value=response) as request:
                with self.assertRaisesRegex(RuntimeError,'출력 한도'):
                    speech.transcribe_audio_only_legacy(root/'unused.wav',root,root/'segments.json',{'key':'test','language':'ko'},lambda *args:None)
                request.assert_called_once()
            self.assertFalse((root/'segments.json').exists());self.assertFalse((root/'gemini-full.json').exists())

    def test_out_of_bounds_exclusion_reports_actual_collection_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            sf.write(root/'gemini-full-16k.flac',np.zeros(2*16000,dtype=np.int16),16000,subtype='PCM_16')
            answer=parsed([segment(.1,1.,'valid')])
            answer.excluded_intervals=[speech.Excluded(start=3.,end=4.)]
            with patch.object(speech,'http_json',return_value=(answer,{})):
                with self.assertRaisesRegex(RuntimeError,'제외 구간 1번 3.000–4.000초.*2.000초'):
                    speech.transcribe_audio_only_legacy(root/'unused.wav',root,root/'segments.json',{'key':'test','language':'ko'},lambda *args:None)
            details=json.loads((root/'gemini-timing-validation.json').read_text(encoding='utf-8'))
            self.assertEqual(details['errors'],[{'collection':'excluded_intervals','index':1,'start':3.,'end':4.}])
            self.assertEqual(details['result']['segments'][0]['original_text'],'valid')
            self.assertFalse((root/'segments.json').exists())
            self.assertFalse((root/'gemini-full.json').exists())

    def test_conversion_encodes_complete_file_and_validates_end_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'voice.wav'
            sf.write(source,np.zeros((2*44100,2),dtype=np.float32),44100,subtype='FLOAT')
            with patch.object(speech,'http_json',return_value=(parsed([segment(1.,2.5,'시간 오류')]),{})) as request:
                with self.assertRaisesRegex(RuntimeError,'전체 오디오 길이'):
                    speech.transcribe_audio_only_legacy(source,root,root/'segments.json',{'key':'test','language':'ko'},lambda *args:None)
                request.assert_called_once()
            info=sf.info(root/'gemini-full-16k.flac')
            self.assertEqual(info.frames,32000);self.assertEqual(info.channels,1)
            self.assertFalse((root/'segments.json').exists());self.assertFalse((root/'gemini-full.json').exists())


if __name__=='__main__':unittest.main()
