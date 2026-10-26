"""Insert validated recovered dialogue while reusing every unchanged TTS clip."""
from pathlib import Path
import copy,hashlib,json,os,shutil,time
import soundfile as sf
from core import ROOT,SR,atomic_json,config,probe,command


def identity(segment):
    return tuple(segment[k] for k in ('start','end','original_text','translated_text'))


def link_or_copy(source,destination):
    if destination.exists():return
    try:os.link(source,destination)
    except OSError:shutil.copy2(source,destination)


def execute(video_id,store):
    import render
    from reference_edges import acoustic_boundaries
    from reference_clean import reference_windows
    from stretch import close_worker
    from mfa_hint import close_worker as close_mfa
    job=ROOT/'work/videos'/video_id;output=ROOT/'outputs'/video_id
    marker=job/'dialogue-refresh.json';manifest=json.loads(marker.read_text(encoding='utf8'))
    started=time.perf_counter();backup=Path(manifest['backup']);cache=Path(manifest['cache'])
    source_cache=Path(manifest['source_cache']);clean=Path(manifest['clean']);background=Path(manifest['background'])
    try:
        backup.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
        for name in ['segments.json','render.json','mux-validation.json','final.mp4']:
            if (output/name).exists():link_or_copy(output/name,backup/name)
        link_or_copy(job/'mixed.wav',backup/'mixed.wav')
        old=json.loads((backup/'segments.json').read_text(encoding='utf8'))
        previous=json.loads((backup/'render.json').read_text(encoding='utf8'))
        repaired=json.loads(Path(manifest['validation']).read_text(encoding='utf8'))['rows']
        values,sr=sf.read(clean,dtype='float32',always_2d=True)
        next_start=min(s['start'] for s in old['segments'])
        if any(r['end']>next_start for r in repaired):raise RuntimeError('Inserted dialogue overlaps existing dialogue.')
        for r in repaired:r.update(source_word_floor=0.,source_word_ceiling=next_start)
        store.update(video_id,phase='누락 대사 갱신',phase_progress=5,progress=5,detail='기존 음성 재사용 · 복구한 참조 경계 확인')
        measurements=acoustic_boundaries(values,sr,repaired);windows=reference_windows(values,sr,measurements)
        current=copy.deepcopy(old);insertions=[]
        from source_recovery import VERSION as RECOVERY_VERSION
        current['asr_recovery_version']=RECOVERY_VERSION
        for r,m,w in zip(repaired,measurements,windows):
            m.update(start=w['start'],end=w['end'],start_frame=w['start_frame'],end_frame=w['end_frame'],
                     duration=w['duration'],source_index=r['source_index'],original_start=r['located_start'],original_end=r['located_end'],tts_reference=w)
            insertions.append(dict(start=w['start'],end=w['end'],original_text=r['recognized_text'],
                                   translated_text=r['translated_text'],asr_speaker_label=r.get('asr_speaker_label'),speaker_boundary_split=False))
        current['segments']=insertions+old['segments'];current['reference_windows']=windows+old['reference_windows']
        restored={r['source_index'] for r in repaired}
        current['unmatched_segments']=[r for r in old.get('unmatched_segments',[]) if r['source_index'] not in restored]
        for report in current.get('asr_recovery',[]):
            if report.get('source_index') in restored and report.get('kind')=='unmatched':
                report.update(status='recovered',reason='complete_candidate_verified_before_ranking',new_api_requests=0)
        current['reference_tail_intervals']=old.get('reference_tail_intervals',[])+[p for w in windows for p in w.get('preserved_intervals',[])]
        source_asr=json.loads(Path(manifest['scribe']).read_text(encoding='utf8'))
        frames=sf.info(job/'original.wav').frames
        segments=render.prepare_segments(current,frames,source_asr)
        old_lookup={identity(s):i for i,s in enumerate(old['segments'])}
        if len(old_lookup)!=len(old['segments']):raise RuntimeError('Existing dialogue cache contains duplicates.')
        cfg=config();cfg['language']=current['target_language'];reports=[];generated=0;reused=0;hashes={}
        for n,s in enumerate(segments):
            old_index=old_lookup.get(identity(s))
            if old_index is not None:
                for suffix in ['-fitted.wav','-tts.mp3','-reference.wav','-timing-reference.wav','-generation.json','-mfa.json','-fitted-mfa.json']:
                    src=source_cache/f'{old_index:04d}{suffix}'
                    if src.exists():link_or_copy(src,cache/f'{n:04d}{suffix}')
                if not (cache/f'{n:04d}-fitted.wav').exists():raise RuntimeError('Existing dubbing cache is missing.')
                hashes[n]=hashlib.sha256((cache/f'{n:04d}-fitted.wav').read_bytes()).hexdigest()
                report=dict(previous['segments'][old_index],index=n)
                atomic_json(cache/f'{n:04d}.json',report);reused+=1
            else:
                store.update(video_id,phase_progress=20,progress=20,detail=f'복구된 대사 {n+1}번만 TTS 생성·길이 보정')
                s['next_dialogue_start_frame']=segments[n+1]['lo'] if n+1<len(segments) else None
                report=render.synthesize(n,s,clean,cache,cfg);generated+=1
            reports.append(report)
        if reused!=len(old['segments']):raise RuntimeError('Not all existing audio was reused.')
        store.update(video_id,phase='영상 갱신',phase_progress=0,progress=45,detail=f'TTS {reused}개 재사용 · {generated}개 추가 · 오디오 합성')
        mixed=cache/'updated-mixed.wav'
        render.mix(job/'original.wav',clean,segments,cache,mixed,
                   lambda p,d:store.update(video_id,phase_progress=p,progress=round(45+p*.15,1)),
                   excluded_intervals=current.get('excluded_intervals',[]),background=background)
        store.update(video_id,phase_progress=60,progress=65,detail='기존 영상 스트림 유지 · MP4 저장 중')
        final=cache/'updated.mp4';render.mux(job/'source.mp4',mixed,final,cfg['language'],float(probe(job/'source.mp4')['format']['duration']))
        store.update(video_id,phase_progress=85,progress=90,detail='전체 영상·음성 디코딩 검증')
        command(['ffmpeg','-nostdin','-v','error','-i',str(final),'-map','0:v:0','-map','0:a:0','-f','null','-'])
        for n,h in hashes.items():
            if hashlib.sha256((cache/f'{n:04d}-fitted.wav').read_bytes()).hexdigest()!=h:raise RuntimeError('Reused audio has changed.')
        placements=render.speech_placements(segments,cache)
        if previous.get('tts_placements')!=[{'start':p['lo']/SR,'end':p['hi']/SR} for p in placements[len(insertions):]]:
            raise RuntimeError('Existing dubbed audio placement has changed.')
        audit=dict(reason='recovered_dialogue_insert',tts_reused=reused,tts_added=generated,
                   asr_api_requests=0,translation_api_requests=0,existing_audio_unchanged=True,
                   fully_decodable=True,elapsed_seconds=round(time.perf_counter()-started,2))
        report=dict(previous,segments=reports,reference_measurements=measurements+previous['reference_measurements'],
                    unmatched_source_segments=current['unmatched_segments'],tts_cache_directory=str(cache),
                    tts_placements=[{'start':p['lo']/SR,'end':p['hi']/SR} for p in placements],
                    replacement_intervals=[{'start':p['lo']/SR,'end':p['hi']/SR} for p in render.replacement_intervals(segments,frames,placements)],
                    output_audio_seconds=sf.info(mixed).duration,refresh=audit)
        for index,measurement in enumerate(report['reference_measurements']):measurement['index']=index
        # Commit only after the replacement video and unchanged cache pass verification.
        staging=output/'dialogue-validated.mp4';link_or_copy(final,staging);staging.replace(output/'final.mp4')
        staging=job/'mixed.ready.wav';link_or_copy(mixed,staging);staging.replace(job/'mixed.wav')
        atomic_json(output/'segments.json',current);atomic_json(output/'render.json',report)
        atomic_json(output/'mux-validation.json',json.loads((cache/'mux-validation.json').read_text(encoding='utf8')))
        atomic_json(marker,dict(manifest,complete=True,verification=audit))
        detail=f'대사 연결 확인 필요 {len(current["unmatched_segments"])}개' if current['unmatched_segments'] else ''
        store.update(video_id,state='done',phase='완료',phase_progress=100,progress=100,detail=detail,error=None)
        print(json.dumps(audit,ensure_ascii=False),flush=True);return 0
    except Exception as error:
        store.update(video_id,state='error',phase='갱신 실패',error=str(error),detail='기존 완성본과 갱신 캐시 보관')
        print(type(error).__name__+': '+str(error),flush=True);return 1
    finally:
        try:close_worker()
        finally:close_mfa()
