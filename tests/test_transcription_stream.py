from pathlib import Path
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import speech


def sse(content,finish='stop',error=None):
    chunks=[': OPENROUTER PROCESSING\n\n']
    for fragment in content:
        chunks.append('data: '+json.dumps({'choices':[{'index':0,'delta':{'content':fragment}}]},ensure_ascii=False)+'\n\n')
    chunks.append('data: '+json.dumps({'choices':[{'delta':{},'finish_reason':finish}],
        'usage':{'prompt_tokens':100,'completion_tokens':50,'cost':.01},**({'error':error} if error else {})})+'\n\n')
    chunks.append('data: [DONE]\n\n')
    return io.BytesIO(''.join(chunks).encode())


class StreamingTests(unittest.TestCase):
    def test_activity_callback_alone_enables_incremental_translation_streaming(self):
        events=[]
        with patch.object(speech.urllib.request,'urlopen',return_value=sse(['{"0":"first",','"1":"second"}'])) as request:
            result,_=speech.http_json({'model':'test'},'test',parser=json.loads,on_activity=events.append,expected_region_ids=[0,1])
        self.assertTrue(json.loads(request.call_args.args[0].data)['stream'])
        self.assertEqual(result,{'0':'first','1':'second'})
        counts=[e['received_segments'] for e in events]
        self.assertIn(1,counts)
        self.assertIn(2,counts)

    def test_translation_tag_objects_count_only_after_each_complete_region(self):
        counter=speech.SegmentStreamCounter([0,1])
        self.assertEqual(counter.feed('{"0":{"t":"안녕.","c":["shouting"]'),0)
        self.assertEqual(counter.feed('},"1":{"t":"네.","c":[]}}'),2)

    def test_voice_numbers_wait_for_complete_numeric_value(self):
        counter=speech.SegmentStreamCounter([0,1])
        self.assertEqual(counter.feed('{"0":1'),0)
        self.assertEqual(counter.feed('2,"1":null}'),2)
        self.assertEqual(counter.seen,{'0','1'})

    def test_voice_entry_arrays_count_only_known_unique_ids(self):
        counter=speech.SegmentStreamCounter([0,1])
        count=counter.feed('[{"id":0,"speaker":2},{"id":1,"speaker":null},{"id":0,"speaker":2},{"id":9,"speaker":3}]')
        self.assertEqual(count,2)
        self.assertEqual(counter.duplicate_keys,['0'])
        self.assertEqual(counter.rejected_keys,['9'])

    def test_voice_entries_with_echoed_locator_times_count_incrementally(self):
        counter=speech.SegmentStreamCounter([0,1])
        self.assertEqual(counter.feed('[{"id":0,"start":1,"end":2,"speaker":3},'),1)
        self.assertEqual(counter.feed('{"id":1,"start":2,"end":3,"speaker":null}]'),2)

    def test_explicit_region_and_speaker_id_aliases_count_live(self):
        counter=speech.SegmentStreamCounter([0,1])
        self.assertEqual(counter.feed('[{"region_id":0,"speaker_id":12},'),1)
        self.assertEqual(counter.feed('{"region_id":1,"speaker_id":null}]'),2)

    def test_echoed_transcripts_are_not_reported_as_speaker_assignments(self):
        counter=speech.SegmentStreamCounter([0,1])
        self.assertEqual(counter.feed('[{"id":0,"start":1,"end":2,"text":"copied input"},'),0)
        self.assertEqual(counter.unrecognized_region_ids,{'0'})

    def test_http_400_preserves_provider_reason_and_redacts_key(self):
        import urllib.error
        error=urllib.error.HTTPError('https://example.invalid',400,'Bad Request',{},
            io.BytesIO(b'{"error":{"message":"Invalid schema: secret-test-key"}}'))
        with tempfile.TemporaryDirectory() as folder,patch('speech.urllib.request.urlopen',side_effect=error):
            with self.assertRaisesRegex(RuntimeError,'Invalid schema') as raised:
                speech.http_json({'model':'test'},'secret-test-key',response_directory=folder)
            self.assertNotIn('secret-test-key',str(raised.exception))
            saved=list(Path(folder).glob('gemini-http-error-*.json'))
            self.assertEqual(len(saved),1)
            data=json.loads(saved[0].read_text('utf8'))
            self.assertEqual(data['status'],400)
            self.assertNotIn('secret-test-key',data['response'])

    def test_incremental_count_ignores_partial_objects_escaped_text_and_exclusions(self):
        line={'start':0.,'end':1.,'original_text':'말 { "segments": [] }','translated_text':'번역'}
        encoded=json.dumps(line,ensure_ascii=False)
        counter=speech.SegmentStreamCounter()
        self.assertEqual(counter.feed('{"segments":['+encoded[:-1]),0)
        self.assertEqual(counter.feed('},'),1)
        self.assertEqual(counter.feed(encoded+'],"excluded_intervals":[{"start":2,"end":3}]}'),2)

    def test_stream_counts_live_preserves_raw_response_and_usage(self):
        line=json.dumps({'start':0.,'end':1.,'original_text':'안녕','translated_text':'안녕'},ensure_ascii=False)
        fragments=['{"segments":[',line[:-1],'},',line,'],"excluded_intervals":[]}']
        counts=[]
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            with patch.object(speech.urllib.request,'urlopen',return_value=sse(fragments)) as request:
                result,usage=speech.http_json({'model':speech.GEMINI},'test',folder,counts.append)
            payload=json.loads(request.call_args.args[0].data)
            self.assertTrue(payload['stream'])
            self.assertEqual(counts,[0,1,2])
            self.assertEqual(len(result.segments),2)
            self.assertEqual(usage['cost'],.01)
            self.assertEqual(len(list(folder.glob('*.sse'))),1)
            saved=json.loads(next(folder.glob('gemini-response-*.json')).read_text(encoding='utf-8'))
            self.assertEqual(saved['choices'][0]['message']['content'],''.join(fragments))

    def test_partial_stream_never_becomes_successful_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            with patch.object(speech.urllib.request,'urlopen',return_value=sse(['{"segments":['],finish=None)):
                with self.assertRaisesRegex(RuntimeError,'response is incomplete'):
                    speech.http_json({},'test',folder,lambda count:None)
            self.assertTrue(list(folder.glob('*.sse')))

    def test_stream_error_is_failure_and_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            with patch.object(speech.urllib.request,'urlopen',return_value=sse([],finish='error',error={'message':'failure'})):
                with self.assertRaisesRegex(RuntimeError,'제공자 오류'):
                    speech.http_json({},'test',folder,lambda count:None)
            self.assertIn('error',json.loads(next(folder.glob('gemini-response-*.json')).read_text(encoding='utf-8')))


    def test_processing_comments_report_activity_but_never_fake_translation_count(self):
        events=[];counts=[]
        response=sse(['{"0":["source",','"target"]}'])
        data=speech.read_stream(response,counts.append,on_activity=events.append)
        self.assertIn('provider_processing',[e['stage'] for e in events])
        waiting=next(e for e in events if e['stage']=='provider_processing')
        self.assertEqual(waiting['received_segments'],0)
        self.assertEqual(waiting['content_characters'],0)
        self.assertEqual(counts,[1])
        self.assertEqual(data['choices'][0]['message']['content'],'{"0":["source","target"]}')

    def test_activity_callback_keeps_one_request_and_emits_request_start(self):
        events=[]
        with patch.object(speech.urllib.request,'urlopen',return_value=sse(['{"0":["a","b"]}'])) as api:
            result,_=speech.http_json({},'test',on_segments=lambda n:None,parser=json.loads,on_activity=events.append)
        api.assert_called_once()
        self.assertEqual(events[0]['stage'],'request_start')
        self.assertIn('connected',[e['stage'] for e in events])
        self.assertEqual(result,{'0':['a','b']})


    def test_counter_never_counts_unrequested_or_duplicate_region_ids(self):
        counter=speech.SegmentStreamCounter(range(231))
        content=json.dumps({str(i):None for i in range(302)})
        self.assertEqual(counter.feed(content),231)
        self.assertEqual(len(counter.rejected_keys),71)
        duplicate=speech.SegmentStreamCounter([0,1])
        self.assertEqual(duplicate.feed('{"0":null,"0":["a","b"],"1":null}'),2)
        self.assertEqual(duplicate.duplicate_keys,['0'])


if __name__=='__main__':unittest.main()
