from pathlib import Path
import base64,hashlib,io,json,sys,tempfile,unittest,urllib.error
from unittest.mock import patch
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scribe
from source_alignment import locate_transcript,source_evidence,reclaim_isolated_prefixes
from timestamp_calibration import calibrate_word_times

class ScribeTests(unittest.TestCase):
    def test_large_flac_is_sent_as_complete_compressed_mp3(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'voice.wav';folder=root/'stt'
            sf.write(source,np.random.default_rng(7).normal(0,.1,2*16000),16000)
            answer={'words':[{'word':'hello','start':.2,'end':1.5}]}
            with patch.object(scribe,'AUDIO_BYTE_BUDGET',26000),patch.object(scribe.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(answer).encode())) as request:
                result=scribe.transcribe_words(source,folder,'test')
            payload=json.loads(request.call_args.args[0].data)
            self.assertEqual(payload['input_audio']['format'],'mp3')
            audio=base64.b64decode(payload['input_audio']['data'])
            self.assertLessEqual(len(audio),26000)
            with sf.SoundFile(io.BytesIO(audio)) as decoded:
                self.assertAlmostEqual(decoded.frames/decoded.samplerate,2.,places=2)
                self.assertEqual(decoded.channels,1)
            meta=json.loads((folder/'scribe-input-compression.json').read_text())
            self.assertEqual(meta['bitrate_kbps'],64)
            self.assertFalse(meta['split']);self.assertFalse(meta['silence_removed'])
            self.assertEqual(result['duration'],2.)

    def test_compression_selects_lower_bitrate_when_needed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'voice.wav';folder=root/'stt';folder.mkdir()
            sf.write(source,np.random.default_rng(8).normal(0,.1,10*16000),16000)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(scribe,'AUDIO_BYTE_BUDGET',30000):
                path,fmt,duration=scribe.prepare_input(source,folder,digest)
            self.assertEqual(fmt,'mp3');self.assertEqual(duration,10.)
            self.assertIn('-16k-1ch-16k.mp3',path.name)
            self.assertLessEqual(path.stat().st_size,30000)
            self.assertAlmostEqual(sf.info(path).duration,10.,places=2)

    def test_isolated_prefix_can_restore_an_omitted_whole_line_without_duplicate_words(self):
        for prefix,body in [('え','いやいやこの子はやめた方が'),('yes','wecanbringtheotherstomorrow')]:
            with self.subTest(language=prefix):
                words=[{'word':prefix,'start':1.,'end':1.2},
                    {'word':body,'start':4.,'end':5.}]
                targets=[prefix,prefix+prefix,prefix+body]
                missing=lambda:{'matched':False,'words':[],'word_indices':[],'coverage':0.}
                locations=[missing(),missing(),{'matched':True,'start':1.,'end':5.,
                    'coverage':1.,'words':words,'word_indices':[0,1]}]
                result=reclaim_isolated_prefixes(targets,words,locations)
                self.assertTrue(result[0]['matched'])
                self.assertFalse(result[1]['matched'])
                self.assertEqual(result[0]['word_indices'],[0])
                self.assertEqual(result[2]['word_indices'],[1])
                self.assertEqual(result[2]['start'],4.)

    def test_prefix_repair_does_not_split_continuous_speech_or_unsupported_text(self):
        for gap,target,body in [(.02,'yes','wecanbringtheotherstomorrow'),
                                (2.,'no','wecanbringtheotherstomorrow'),(2.,'yes','x')]:
            with self.subTest(gap=gap,target=target,body=body):
                words=[{'word':'yes','start':1.,'end':1.2},
                    {'word':body,'start':1.2+gap,'end':2.2+gap}]
                locations=[{'matched':False,'words':[],'word_indices':[]},
                    {'matched':True,'start':1.,'end':2.2+gap,'words':words,'word_indices':[0,1]}]
                result=reclaim_isolated_prefixes([target,'yes'+body],words,locations)
                self.assertFalse(result[0]['matched'])
                self.assertEqual(result[1]['word_indices'],[0,1])

    def test_cached_raw_response_restores_punctuation_without_new_asr_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'voice.wav';sf.write(source,np.zeros(16000),16000)
            raw={'words':[{'word':'声','start':.1,'end':.4},{'word':'。','start':.4,'end':.4}]}
            with patch('scribe.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(raw).encode())):
                scribe.transcribe_words(source,root/'asr','test')
            cache=root/'asr/scribe.json';old=json.loads(cache.read_text(encoding='utf8'))
            old['words']=old['words'][:1];cache.write_text(json.dumps(old),encoding='utf8')
            with patch('scribe.urllib.request.urlopen',side_effect=AssertionError('No duplicate ASR')):
                refreshed=scribe.transcribe_words(source,root/'asr','test')
            self.assertEqual([w['word'] for w in refreshed['words']],['声','。'])

    def test_single_attempt_does_not_repeat_a_failed_tts_asr_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'tts.wav';sf.write(source,np.zeros(16000,dtype=np.float32),16000)
            error=urllib.error.HTTPError(scribe.URL,503,'unavailable',{},None)
            with patch.object(scribe.urllib.request,'urlopen',side_effect=error) as request:
                with self.assertRaises(RuntimeError):scribe.transcribe_words(source,root/'asr','test','ko',attempts=1)
            self.assertEqual(request.call_count,1)

    def test_verbose_word_request_sends_complete_audio_and_success_is_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'voice.wav';folder=root/'stt'
            sf.write(source,np.zeros(2*16000,dtype=np.float32),16000)
            answer={'text':'hello world','words':[{'word':'hello','start':.2,'end':.5,'confidence':.9},{'word':'world','start':1.,'end':1.5}], 'usage':{'seconds':2.,'cost':.0001}}
            with patch.object(scribe.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(answer).encode())) as request:
                first=scribe.transcribe_words(source,folder,'test','en')
            payload=json.loads(request.call_args.args[0].data)
            self.assertEqual(payload['model'],'elevenlabs/scribe-v2');self.assertEqual(payload['response_format'],'verbose_json')
            self.assertEqual(payload['timestamp_granularities'],['word'])
            self.assertNotIn('provider',payload)
            self.assertNotIn('diarization_threshold',first['signature'])
            with sf.SoundFile(io.BytesIO(base64.b64decode(payload['input_audio']['data']))) as audio:self.assertEqual(audio.frames,32000)
            with patch.object(scribe.urllib.request,'urlopen',side_effect=AssertionError('Duplicate paid request')):
                second=scribe.transcribe_words(source,folder,'test','en')
            self.assertEqual(first,second)

    def test_invalid_word_times_are_rejected_and_zero_duration_text_is_retained(self):
        words,events=scribe.parse_words({'words':[
            {'word':'a','start':0.,'end':.1,'confidence':0.},
            {'word':'b','start':.1,'end':.1},
            {'word':'(laugh)','start':.2,'end':.3},
            {'word':'invalid','start':float('nan'),'end':.4},
            {'word':'negative','start':-.1,'end':.3}]},1.)
        self.assertEqual([w['word'] for w in words],['a','b']);self.assertEqual(words[0]['probability'],0.)
        self.assertEqual(len(events),1)
        with self.assertRaises(RuntimeError):scribe.parse_words({'text':'no times'},1.)

    def test_twenty_second_wrong_llm_times_do_not_affect_full_recording_matching(self):
        segments=[{'start':210.,'end':211.,'original_text':'alpha unique'},{'start':244.,'end':245.,'original_text':'beta different'}]
        words=[{'word':'alpha','start':10.,'end':10.5},{'word':'unique','start':10.5,'end':11.},
            {'word':'beta','start':20.,'end':20.5},{'word':'different','start':20.5,'end':21.}]
        result=locate_transcript(segments,words,250.)
        self.assertEqual([(r['start'],r['end']) for r in result],[(10.,11.),(20.,21.)])

    def test_japanese_character_tokens_match_complete_phrases(self):
        segments=[{'original_text':'こんにちは。','start':100.,'end':101.}]
        words=[{'word':c,'start':i*.1,'end':(i+1)*.1} for i,c in enumerate('こんにちは')]
        result=locate_transcript(segments,words,200.)[0]
        self.assertTrue(result['matched']);self.assertEqual(result['start'],0.)

    def test_repeated_line_cannot_join_first_character_to_a_later_repetition(self):
        texts=['ガビル様こそ無事でよかった','お前たち','もういい']
        parts=[(texts[0],17.),('お',21.),('お前たち',22.3),
               ('我ら生まれた時は違えど死ぬ時は一緒だぞ',26.7),('ガビルしゃまー',31.3),
               ('お前たち',34.4),(texts[2],36.7)]
        words=[{'word':c,'start':start+i*.1,'end':start+(i+1)*.1} for text,start in parts for i,c in enumerate(text)]
        segments=[{'original_text':text,'start':200.,'end':201.} for text in texts]
        result=locate_transcript(segments,words,250.)
        self.assertTrue(all(r['matched'] for r in result))
        self.assertAlmostEqual(result[1]['start'],22.3)
        self.assertAlmostEqual(result[1]['end'],22.7)
        self.assertEqual(''.join(w['word'] for w in result[1]['words']),'お前たち')

    def test_two_requested_identical_lines_keep_distinct_ordered_occurrences(self):
        text='お前たち';words=[]
        for start in [3.,12.]:
            words.extend({'word':c,'start':start+i*.1,'end':start+(i+1)*.1} for i,c in enumerate(text))
        result=locate_transcript([{'original_text':text},{'original_text':text}],words,20.)
        self.assertEqual([r['start'] for r in result],[3.,12.])
        self.assertTrue(all(r['matched'] for r in result))

    def test_genuine_long_internal_pause_is_not_cut_by_a_duration_limit(self):
        words=[{'word':'alpha','start':1.,'end':1.4},{'word':'beta','start':14.,'end':14.5}]
        result=locate_transcript([{'original_text':'alpha beta'}],words,20.)[0]
        self.assertTrue(result['matched']);self.assertEqual((result['start'],result['end']),(1.,14.5))

    def test_local_alignment_keeps_partial_asr_spelling_errors(self):
        words=[{'word':c,'start':i*.1,'end':(i+1)*.1} for i,c in enumerate('ガビルしゃまー')]
        result=locate_transcript([{'original_text':'ガビル様ー'}],words,3.)[0]
        self.assertTrue(result['matched']);self.assertGreaterEqual(result['coverage'],.6)

    def test_one_unmatched_line_is_reported_without_failing_valid_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'voice.wav';source.write_bytes(b'test')
            transcript=[{'original_text':'hello world','translated_text':'hello world','start':1.,'end':2.},{'original_text':'zzzzzz','translated_text':'zzzzzz','start':2.,'end':3.}]
            data={'signature':{'model':scribe.MODEL,'source_sha256':'test'},'words':[
                {'word':'hello','start':.2,'end':.5},{'word':'world','start':.5,'end':.9}],
                'audio_events':[],'duration':3.,'usage':{}}
            with patch('source_alignment.transcribe_words',return_value=data):
                result=source_evidence(source,transcript,root/'timing','en',key='test')
            self.assertEqual(len(result['segments']),1);self.assertEqual(result['unmatched_segments'][0]['source_index'],1)

    def test_timestamp_bias_is_measured_from_audio_not_a_fixed_offset(self):
        sr=16000;t=np.arange(10*sr)/sr;x=np.zeros(len(t),dtype=np.float32);rows=[]
        for i in range(8):
            start=.5+i;mask=(t>=start)&(t<start+.2);x[mask]=.2*np.sin(t[mask]*2*np.pi*220)
            rows.append({'start':start+.13,'end':start+.33,'words':[{'word':'a','start':start+.13,'end':start+.33}]})
        _,report=calibrate_word_times(x,sr,rows)
        self.assertAlmostEqual(report['shift_seconds'],-.13,delta=.02)
        _,short=calibrate_word_times(x,sr,rows[:2]);self.assertEqual(short['shift_seconds'],0.)

if __name__=='__main__':unittest.main()
