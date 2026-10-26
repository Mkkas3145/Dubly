"""Rerender a saved dubbing script with the current timing engine, offline."""
from pathlib import Path
from unittest.mock import patch
import hashlib,json,shutil,time
import soundfile as sf
from core import ROOT,SR,FIT_CONCURRENCY,atomic_json,command,config,probe

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def copy_cached_inputs(source_cache,cache,index):
    """Include the actual alignment filename derived from NNNN-fitted.wav."""
    for suffix in ['-tts.mp3','-generation.json','-reference.wav','-timing-reference.wav','-mfa.json','-fitted-mfa.json']:
        src=Path(source_cache)/f'{index:04d}{suffix}';dest=Path(cache)/src.name
        if src.exists() and not dest.exists():shutil.copy2(src,dest)

def execute(video_id,store):
    import render
    from timing import FIT_VERSION,TIMING_POLICY
    from stretch import ENGINE,VERSION,close_worker
    from mfa_hint import close_worker as close_mfa
    job=ROOT/'work/videos'/video_id;output=ROOT/'outputs'/video_id
    marker=job/'audio-refresh.json';manifest=json.loads(marker.read_text(encoding='utf-8'))
    backup=Path(manifest['backup_directory']);source_cache=Path(manifest['source_cache']);cache=Path(manifest['cache'])
    clean=Path(manifest['clean']);background=Path(manifest['background']) if manifest.get('background') else None
    transcript=json.loads((backup/'segments.json').read_text(encoding='utf-8'))
    previous=json.loads((backup/'render.json').read_text(encoding='utf-8'))
    started=time.perf_counter()
    try:
        if manifest['engine']!=ENGINE or manifest['fit_version']!=FIT_VERSION:raise RuntimeError('The requested time-stretch engine version has changed.')
        cfg=config();cfg['language']=transcript['target_language']
        # This entry point promises a strictly offline timing-only refresh.
        cfg['voice_similarity']=False
        cache.mkdir(parents=True,exist_ok=True)
        segments=render.prepare_segments(transcript,sf.info(job/'original.wav').frames)
        before={}
        for n in range(len(segments)):
            for suffix in ['-tts.mp3','-generation.json','-reference.wav']:
                src=source_cache/f'{n:04d}{suffix}'
                if not src.exists():raise RuntimeError(f'Existing audio cache is missing for dialogue {n+1}.')
                before[str(src)]=digest(src)
            generation=json.loads((source_cache/f'{n:04d}-generation.json').read_text(encoding='utf-8'))
            if generation.get('tts_policy')!=render.tts_policy(cfg['language']):raise RuntimeError('Cached TTS instructions differ from current instructions. Stopped without regenerating.')
            copy_cached_inputs(source_cache,cache,n)
        def status(p,detail):
            store.update(video_id,phase='Google Speedy 보정',phase_progress=round(p,1),progress=round(p*.85,1),detail=detail)
            print(json.dumps({'phase':'speedy','percent':round(p,1),'detail':detail},ensure_ascii=False),flush=True)
        with patch('render.request_audio',side_effect=RuntimeError('갱신 중 TTS 재요청은 허용하지 않습니다.')), \
             patch('urllib.request.urlopen',side_effect=RuntimeError('갱신 중 외부 API를 호출하지 않습니다.')):
            reports=render.synthesize_individual_all(segments,clean,cache,cfg,lambda *a:None,status=status)
        for n,r in enumerate(reports):
            if r['tempo_engine'] not in {'none',ENGINE} or r['tempo']<1:raise RuntimeError('Time-stretch engine or speed validation failed.')
            if r['fit_version']!=FIT_VERSION or r['timing_method']!=TIMING_POLICY:raise RuntimeError('Time-stretch cache version validation failed.')
            if r.get('overlap_after_speedup_frames',0):raise RuntimeError('The result overlaps the next dialogue.')
            if r['collision_budget_frames'] is not None and r['output_frames']>r['collision_budget_frames']:raise RuntimeError('The result extends past the next dialogue start.')
            for suffix in ['-tts.mp3','-generation.json','-reference.wav']:
                if digest(cache/f'{n:04d}{suffix}')!=before[str(source_cache/f'{n:04d}{suffix}')]:raise RuntimeError('Cached TTS or reference audio has changed.')
        store.update(video_id,phase='영상 합성·검증',progress=88,phase_progress=0,detail='원본 영상 스트림 유지')
        mixed=cache/'refreshed-mixed.wav';final=cache/'refreshed.mp4'
        render.mix(job/'original.wav',clean,segments,cache,mixed,
          lambda p,d:store.update(video_id,progress=round(88+p*.05,1),phase_progress=p),
          excluded_intervals=transcript.get('excluded_intervals',[]),background=background)
        store.update(video_id,progress=94,detail='MP4 저장·전체 디코딩 확인')
        render.mux(job/'source.mp4',mixed,final,cfg['language'],float(probe(job/'source.mp4')['format']['duration']))
        def video_hash(p):return command(['ffmpeg','-nostdin','-v','error','-i',str(p),'-map','0:v:0','-c','copy','-f','hash','-hash','sha256','-']).strip()
        if video_hash(final)!=video_hash(job/'source.mp4'):raise RuntimeError('The source video stream has changed.')
        store.update(video_id,progress=97,detail='MP4 저장 완료 · 전체 재생 오류 검증')
        command(['ffmpeg','-nostdin','-v','error','-i',str(final),'-f','null','-'])
        for p,h in before.items():
            if digest(p)!=h:raise RuntimeError('The source audio cache has changed.')
        placements=render.speech_placements(segments,cache)
        audit={'reason':'speedy_timing_refresh','segments':len(reports),'tts_reused':len(reports),'tts_api_requests':0,
          'asr_api_requests':0,'translation_unchanged':True,'references_unchanged':True,
          'speedy_segments':sum(r['tempo_engine']==ENGINE for r in reports),
          'no_slowdown':True,'no_next_dialogue_overlap':True,'video_stream_preserved':True,'fully_decodable':True,
          'elapsed_seconds':time.perf_counter()-started}
        report={**previous,'segments':reports,'timing_fit_version':FIT_VERSION,'timing_fit_policy':TIMING_POLICY,
          'speed_correction_engine':ENGINE,'speed_correction_engine_version':VERSION,'tts_cache_directory':str(cache),
          'speed_processing':'google_speedy_only_for_next_dialogue_overlap','speed_scheduling':'offline_wsl_cpu_per_utterance',
          'fit_concurrency':FIT_CONCURRENCY,'speed_concurrency':FIT_CONCURRENCY,'audio_prepare_concurrency':FIT_CONCURRENCY,
          'pause_planning':'algorithmic_pause_only','pause_planning_api_requests':0,
          'tts_overlap_allowed':False,'tts_duration_policy':'pause_first_keep_duration_until_next_dialogue_then_speedy',
          'tts_placements':[{'start':p['lo']/SR,'end':p['hi']/SR} for p in placements],
          'replacement_intervals':[{'start':s['lo']/SR,'end':s['hi']/SR} for s in render.replacement_intervals(segments,sf.info(job/'original.wav').frames,placements)],
          'fade_seconds':render.TRANSITION_SECONDS,'replacement_margin_seconds':render.REPLACEMENT_MARGIN_SECONDS,
          'output_audio_seconds':sf.info(mixed).duration,'refresh':audit}
        atomic_json(cache/'refreshed-render.json',report)
        shutil.copy2(mixed,job/'mixed.wav')
        staging=output/'speedy-validated.mp4';shutil.copy2(final,staging);staging.replace(output/'final.mp4')
        atomic_json(output/'render.json',report)
        if digest(output/'segments.json')!=digest(backup/'segments.json'):raise RuntimeError('The translation or timing boundary file has changed.')
        atomic_json(marker,{**manifest,'complete':True,'verification':audit})
        unmatched=len(transcript.get('unmatched_segments',[]))
        detail=f'대사 연결 확인 필요 {unmatched}개 · 해당 대사는 원음 유지 (segments.json 참고)' if unmatched else ''
        store.update(video_id,state='done',phase='완료',progress=100,phase_progress=100,detail=detail,error=None)
        print(json.dumps({'complete':True,**audit},ensure_ascii=False),flush=True)
        return 0
    except Exception as error:
        store.update(video_id,state='error',phase='실패',error=str(error),detail='기존 완성본과 새 보정 캐시 보관')
        print(str(error),flush=True)
        return 1
    finally:
        try:close_worker()
        finally:close_mfa()
