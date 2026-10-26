from pathlib import Path
import base64
import io
import json
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import render
import speech
from storage import Store
from queue_worker import VideoQueue
from core import SR


def line(start=0.,end=1.,speaker='speaker_1',emotion='neutral',text='dialogue'):
    return dict(start=start,end=end,speaker_id=speaker,emotion=emotion,original_text=text,translated_text='translated')


class PipelineTests(unittest.TestCase):
    def test_videos_are_serial_and_failure_keeps_next_queued(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=Store(Path(temporary))
            for n in range(3):store.add(str(n),f'{n}.mp4')
            order=[];active=0;peak=0
            def process(job):
                nonlocal active,peak
                active+=1;peak=max(peak,active);order.append(job['id'])
                try:
                    if job['id']=='1':raise RuntimeError('bad file')
                    store.update(job['id'],state='done',phase='완료',progress=100)
                finally:active-=1
            queue=VideoQueue(store,process);queue.start()
            thread=queue.thread
            if thread:thread.join(5)
            self.assertFalse(queue.running)
            self.assertEqual(order,['0','1'])
            self.assertEqual(peak,1)
            self.assertEqual([j['state'] for j in store.list()],['done','error','queued'])

    def test_pause_after_current_video_and_restart_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=Store(Path(temporary));store.add('a','a.mp4');store.add('b','b.mp4')
            entered=threading.Event();release=threading.Event()
            def process(job):
                entered.set();release.wait(5);store.update(job['id'],state='done')
            queue=VideoQueue(store,process);queue.start();self.assertTrue(entered.wait(5))
            thread=queue.thread;queue.pause();release.set();thread.join(5)
            self.assertEqual([j['state'] for j in store.list()],['done','queued'])
            store.update('b',state='running');store.recover()
            self.assertEqual(store.get('b')['state'],'queued')

    def test_tts_configured_slots_and_ordered_results(self):
        lock=threading.Lock();barrier=threading.Barrier(render.TTS_CONCURRENCY);active=0;peak=0;updates=[]
        def process(index,value):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            try:
                if index<render.TTS_CONCURRENCY:barrier.wait(5)
                return index
            finally:
                with lock:active-=1
        result=render.pipelined_queue([None]*23,process,lambda value:value,lambda *v:updates.append(v))
        self.assertEqual(peak,render.TTS_CONCURRENCY);self.assertEqual(result,list(range(23)))
        self.assertEqual(updates[0],(0,render.TTS_CONCURRENCY,23-render.TTS_CONCURRENCY,0,0));self.assertEqual(updates[-1],(23,0,0,0,0))

    def test_next_tts_starts_while_both_fitting_slots_are_busy(self):
        both_fitting=threading.Event();next_tts=threading.Event();lock=threading.Lock()
        fitting=0;peak=0;updates=[]
        def generate(index,value):
            if index>=render.TTS_CONCURRENCY:
                if not both_fitting.wait(5):raise RuntimeError('Fitting did not overlap TTS')
                next_tts.set()
            return index
        def fit(index):
            nonlocal fitting,peak
            with lock:
                fitting+=1;peak=max(peak,fitting)
                if fitting==render.FIT_CONCURRENCY:both_fitting.set()
            try:
                if not next_tts.wait(5):raise RuntimeError('Fitting occupied the TTS slots')
                return index
            finally:
                with lock:fitting-=1
        with patch.object(render,'FIT_CONCURRENCY',2):
            results=render.pipelined_queue([None]*30,generate,fit,lambda *v:updates.append(v))
            self.assertEqual(peak,render.FIT_CONCURRENCY)
        self.assertEqual(results,list(range(30)))
        self.assertTrue(any(running and fitting for _,running,_,fitting,_ in updates))
        self.assertEqual(updates[-1],(30,0,0,0,0))

    def test_fitting_backlog_is_bounded_and_results_keep_original_order(self):
        release=threading.Event();backpressure=threading.Event();reports=[];errors=[]
        def progress(done,running,queued,fitting,waiting):
            if waiting==2*render.TTS_CONCURRENCY:
                backpressure.set()
            self.assertLessEqual(waiting+running,2*render.TTS_CONCURRENCY)
        def fit(index):
            if not release.wait(5):raise RuntimeError('Test fitting release timed out')
            return index
        def run():
            try:reports.extend(render.pipelined_queue([None]*100,lambda n,v:n,fit,progress))
            except Exception as error:errors.append(error)
        thread=threading.Thread(target=run);thread.start()
        try:self.assertTrue(backpressure.wait(5))
        finally:release.set();thread.join(10)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        self.assertEqual(reports,list(range(100)))

    def test_generation_error_drains_started_work_and_stops_new_submissions(self):
        allow_generation=threading.Event();generated=[];fitted=[]
        def generate(index,value):
            generated.append(index)
            if index==0:raise RuntimeError('generation failed')
            if not allow_generation.wait(5):raise RuntimeError('Test generation release timed out')
            return index
        def progress(done,running,queued,fitting,waiting):
            # Once the failing future is consumed, release the other in-flight work.
            if running<render.TTS_CONCURRENCY:allow_generation.set()
        with self.assertRaisesRegex(RuntimeError,'TTS 1번 구간: generation failed'):
            render.pipelined_queue([None]*100,generate,lambda n:fitted.append(n),progress)
        self.assertEqual(sorted(generated),list(range(render.TTS_CONCURRENCY)))
        self.assertEqual(sorted(fitted),list(range(1,render.TTS_CONCURRENCY)))

    def test_primary_and_brief_requests_share_configured_api_slots(self):
        lock=threading.Lock();barrier=threading.Barrier(render.TTS_CONCURRENCY);active=0;peak=0;entered=0
        def request(*args):
            nonlocal active,peak,entered
            with lock:
                active+=1;peak=max(peak,active);entered+=1;first_batch=entered<=render.TTS_CONCURRENCY
            try:
                if first_batch:barrier.wait(5)
                return args[-1]
            finally:
                with lock:active-=1
        with patch.object(render,'_request_audio',side_effect=request),render.ThreadPoolExecutor(max_workers=render.TTS_CONCURRENCY+2) as pool:
            futures=[pool.submit(render.request_audio,None,None,'ko','test',None,brief=bool(n%2)) for n in range(40)]
            values=[f.result(10) for f in futures]
        self.assertEqual(peak,render.TTS_CONCURRENCY);self.assertEqual(values,[bool(n%2) for n in range(40)])

    def test_empty_pipeline_finishes_without_workers(self):
        updates=[]
        self.assertEqual(render.pipelined_queue([],lambda *a:None,lambda *a:None,lambda *v:updates.append(v)),[])
        self.assertEqual(updates,[(0,0,0,0,0)])

    def test_tts_has_no_fade_and_no_original_voice_at_either_edge(self):
        clean=np.full((100,2),.1,dtype=np.float32)
        original=clean+.2;translated=np.full((100,2),-.05,dtype=np.float32)
        np.testing.assert_allclose(render.blend(original,clean,translated),.15,atol=1e-7)

    def test_mix_preserves_outside_and_length(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            original=np.full((3*SR,2),.3,dtype=np.float32);clean=np.full_like(original,.1)
            sf.write(folder/'original.wav',original,SR,subtype='FLOAT')
            sf.write(folder/'clean.wav',clean,SR,subtype='FLOAT')
            sf.write(folder/'0000-fitted.wav',np.full((SR,2),-.05,dtype=np.float32),SR,subtype='FLOAT')
            render.mix(folder/'original.wav',folder/'clean.wav',[dict(lo=SR,hi=2*SR)],folder,folder/'mix.wav',lambda *v:None)
            mixed,_=sf.read(folder/'mix.wav',dtype='float32',always_2d=True)
            np.testing.assert_array_equal(mixed[:round(.3*SR)],original[:round(.3*SR)]);np.testing.assert_array_equal(mixed[round(2.7*SR):],original[round(2.7*SR):])
            np.testing.assert_allclose(mixed[SR:2*SR],.2-.05*render.TTS_GAIN,atol=1e-7)
            np.testing.assert_allclose(mixed[round(.8*SR):SR],.2,atol=1e-7)
            np.testing.assert_allclose(mixed[2*SR:round(2.2*SR)],.2,atol=1e-7)
            np.testing.assert_array_equal(mixed[:round(.5*SR)],original[:round(.5*SR)])
            np.testing.assert_array_equal(mixed[round(2.5*SR):],original[round(2.5*SR):])
            np.testing.assert_allclose(mixed[round(.65*SR)],.275,atol=1e-7)
            np.testing.assert_allclose(mixed[round(2.35*SR)],.275,atol=1e-7)

    def test_replacement_margin_clips_to_video_edges_and_shares_short_gaps(self):
        segments=[{'lo':0,'hi':SR//4},{'lo':SR//4+101,'hi':SR//2},{'lo':3*SR//4,'hi':SR}]
        before=[dict(s) for s in segments]
        intervals=render.replacement_intervals(segments,SR)
        self.assertEqual(segments,before)
        self.assertEqual(intervals[0]['lo'],0);self.assertEqual(intervals[-1]['hi'],SR)
        midpoint=SR//4+50
        self.assertEqual(intervals[0]['hi'],midpoint);self.assertEqual(intervals[1]['lo'],midpoint)
        self.assertEqual(intervals[1]['hi'],(SR//2+3*SR//4)//2)
        for original,padded in zip(segments,intervals):
            self.assertLessEqual(padded['lo'],original['lo']);self.assertGreaterEqual(padded['hi'],original['hi'])
        weights=render.transition_weights(0,SR,intervals)
        np.testing.assert_array_equal(weights[segments[0]['hi']:segments[1]['lo']],1)
        self.assertLessEqual(float(weights.max()),1)

    def test_external_fades_overlap_without_attenuating_tts_or_double_removal(self):
        segments=[{'lo':SR,'hi':2*SR},{'lo':round(2.2*SR),'hi':round(3.2*SR)}]
        weights=render.transition_weights(0,4*SR,segments)
        np.testing.assert_array_equal(weights[SR:2*SR],1)
        np.testing.assert_array_equal(weights[round(2.2*SR):round(3.2*SR)],1)
        self.assertAlmostEqual(float(weights[round(2.1*SR)]),.5,places=6)
        self.assertLessEqual(float(weights.max()),1)
        self.assertEqual(weights[0],0)
        self.assertEqual(weights[-1],0)

    def test_overlapping_and_touching_fades_across_mix_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            original=np.full((12*SR,2),.3,dtype=np.float32)
            sf.write(folder/'original.wav',original,SR,subtype='FLOAT')
            sf.write(folder/'clean.wav',np.full_like(original,.1),SR,subtype='FLOAT')
            segments=[{'lo':round(a*SR),'hi':round(b*SR)} for a,b in [(9.8,10.),(10.2,10.4),(10.4,10.6)]]
            for n,segment in enumerate(segments):
                sf.write(folder/f'{n:04d}-fitted.wav',np.full((segment['hi']-segment['lo'],2),-.05 if n!=1 else .05,dtype=np.float32),SR,subtype='FLOAT')
            render.mix(folder/'original.wav',folder/'clean.wav',segments,folder,folder/'mix.wav',lambda *v:None)
            mixed,_=sf.read(folder/'mix.wav',dtype='float32',always_2d=True)
            self.assertEqual(len(mixed),len(original))
            for n,segment in enumerate(segments):
                np.testing.assert_allclose(mixed[segment['lo']:segment['hi']],.2+(.05 if n==1 else -.05)*render.TTS_GAIN,atol=1e-7)
            self.assertAlmostEqual(float(mixed[round(10.1*SR),0]),.2,places=6)
            np.testing.assert_array_equal(mixed[:9*SR],original[:9*SR])
            np.testing.assert_array_equal(mixed[round(11.3*SR):],original[round(11.3*SR):])

    def test_fades_at_video_edges_keep_dubbed_intervals_full(self):
        segments=[{'lo':0,'hi':SR//4},{'lo':3*SR//4,'hi':SR}]
        weights=render.transition_weights(0,SR,segments)
        np.testing.assert_array_equal(weights[:SR//4],1)
        np.testing.assert_array_equal(weights[3*SR//4:],1)
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreaterEqual(float(weights.min()),0)
        self.assertLessEqual(float(weights.max()),1)

    def test_excluded_reactions_do_not_interrupt_standard_200ms_fades(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            original=np.full((3*SR,2),.3,dtype=np.float32)
            sf.write(folder/'original.wav',original,SR,subtype='FLOAT')
            sf.write(folder/'clean.wav',np.full_like(original,.1),SR,subtype='FLOAT')
            sf.write(folder/'0000-fitted.wav',np.full((SR,2),-.05,dtype=np.float32),SR,subtype='FLOAT')
            excluded=[dict(start=.65,end=.85,reason='nonlexical_vocalization'),dict(start=2.15,end=2.3,reason='nonlexical_vocalization')]
            render.mix(folder/'original.wav',folder/'clean.wav',[dict(lo=SR,hi=2*SR)],folder,folder/'mix.wav',lambda *v:None,excluded_intervals=excluded)
            mixed,_=sf.read(folder/'mix.wav',dtype='float32',always_2d=True)
            np.testing.assert_array_equal(mixed[:round(.3*SR)],original[:round(.3*SR)])
            np.testing.assert_array_equal(mixed[round(2.7*SR):],original[round(2.7*SR):])
            np.testing.assert_allclose(mixed[SR:2*SR],.2-.05*render.TTS_GAIN,atol=1e-7)
            self.assertAlmostEqual(float(mixed[round(.75*SR),0]),.3-.1*.75,places=4)
            self.assertAlmostEqual(float(mixed[round(2.25*SR),0]),.3-.1*.75,places=4)

    def test_duplicate_full_recording_dialogue_is_kept_once(self):
        complete=line(118.06,121.26,emotion='angry',text='勝つ！あいつは俺を怒らせた。')
        repeated={**complete,'start':119.279,'end':121.039}
        result=speech.normalize([complete,repeated],[])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['original_text'],complete['original_text'])
        self.assertEqual(result[0]['start'],118.06)
        self.assertEqual(result[0]['end'],121.26)
        self.assertEqual(set(result[0]),{'start','end','original_text','translated_text'})
        self.assertEqual(len(speech.normalize([complete,{**repeated,'start':121.3,'end':123.1}],[])),2)
        with self.assertRaises(RuntimeError):
            speech.normalize([complete,{**repeated,'original_text':'別の台詞です。'}],[])

    def test_speaker_and_emotion_changes_prevent_short_merge(self):
        self.assertEqual(len(speech.normalize([line(0.,.5,text='네'),line(.6,1.,text='그래요')],[])),1)
        self.assertEqual(len(speech.normalize([line(0.,.5),line(.6,1.,speaker='speaker_2')],[])),2)
        self.assertEqual(len(speech.normalize([line(0.,.5),line(.6,1.,emotion='angry')],[])),2)
        self.assertEqual(speech.normalize([line(0.,2.),line(1.,3.,speaker='speaker_2')],[]),[])

    def test_legacy_single_llm_call_for_whole_audio_and_public_fields(self):
        parsed=speech.Transcript.model_validate(dict(source_language='en',speakers=[dict(id='speaker_1',description='voice')],
            segments=[line(.1,.8)],excluded_intervals=[]))
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            sf.write(folder/'gemini-full-16k.flac',np.zeros(16000,dtype=np.int16),16000,subtype='PCM_16')
            with patch.object(speech,'http_json',return_value=(parsed,{})) as request:
                result=speech.transcribe_audio_only_legacy(folder/'unused.wav',folder,folder/'segments.json',{'key':'test','language':'ko'},lambda *v:None)
                self.assertEqual(request.call_count,1)
                payload=request.call_args.args[0]
                self.assertIn('reverb/echo decay',payload['messages'][0]['content'])
                self.assertEqual(payload['model'],speech.GEMINI)
            self.assertEqual(set(result['segments'][0]),{'start','end','original_text','translated_text'})
            with patch.object(speech,'http_json') as request:
                speech.transcribe_audio_only_legacy(folder/'unused.wav',folder,folder/'segments.json',{'key':'test','language':'ko'},lambda *v:None)
                request.assert_not_called()

    def test_fish_uses_clean_reference_and_reference_transcript(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);sf.write(folder/'clean.wav',np.full((SR,2),.1,dtype=np.float32),SR,subtype='FLOAT')
            segment={**line(),'lo':0,'hi':SR}
            render.reference_audio(folder/'clean.wav',segment,folder/'reference.wav')
            received=[]
            class Response(io.BytesIO):
                headers={'X-Generation-Id':'test-generation'}
            def fake(request,**kwargs):received.append(json.loads(request.data));return Response(b'a'*128)
            with patch.object(render.urllib.request,'urlopen',side_effect=fake),patch.object(render,'probe',return_value={'format':{'duration':'1'}}):
                render.request_audio(folder/'reference.wav',segment,'ko','test',folder/'tts.mp3')
            payload=received[0];self.assertEqual(payload['input'],segment['translated_text'])
            self.assertEqual(payload['input_references'][1]['text'],segment['original_text'])
            data=base64.b64decode(payload['input_references'][0]['input_audio']['data'].split(',')[1])
            values,_=sf.read(io.BytesIO(data));np.testing.assert_allclose(values,.1,atol=1/32768)

    def test_generation_rejects_foreign_reference_cache_and_uses_measured_vocals(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            sf.write(folder/'clean.wav',np.full((2*SR,2),.13,dtype=np.float32),SR,subtype='FLOAT')
            sf.write(folder/'0000-tts.mp3',np.full(SR,.4,dtype=np.float32),SR,format='WAV',subtype='FLOAT')
            (folder/'0000-generation.json').write_text(json.dumps({
                'tts_policy':render.tts_policy('ko'),'reference_source':'different_reference_model'}),encoding='utf-8')
            segment={**line(1.,2.),'lo':SR,'hi':2*SR,'reference_lo':SR+100,'reference_hi':2*SR-200}
            received=[]
            def request(reference,seg,language,key,destination):
                values,rate=sf.read(reference,dtype='float32')
                received.append((values,rate))
                sf.write(destination,np.full(SR,.12,dtype=np.float32),SR,format='WAV',subtype='FLOAT')
                return {'generation_id':'new-generation'}
            with patch.object(render,'request_audio',side_effect=request):
                result=render.generate_speech(0,segment,folder/'clean.wav',folder,{'language':'ko','key':'test'})
                render.generate_speech(0,segment,folder/'clean.wav',folder,{'language':'ko','key':'test'})
            self.assertEqual(len(received),1)
            values,rate=received[0]
            self.assertEqual(rate,SR);self.assertEqual(len(values),SR-300)
            np.testing.assert_allclose(values,.13,atol=1/32768)
            self.assertEqual(sf.info(result['timing_reference']).frames,SR)
            self.assertEqual(result['generation']['reference_source'],'separated_dialogue')
            # Same words and timing with a different source voice must never
            # reuse the previous generation merely because its text matches.
            sf.write(folder/'clean.wav',np.full((2*SR,2),.21,dtype=np.float32),SR,subtype='FLOAT')
            with patch.object(render,'request_audio',side_effect=request):
                render.generate_speech(0,segment,folder/'clean.wav',folder,{'language':'ko','key':'test'})
            self.assertEqual(len(received),2)
            np.testing.assert_allclose(received[1][0],.21,atol=1/32768)


if __name__=='__main__':unittest.main()
