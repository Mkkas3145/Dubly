from __future__ import annotations
import json
import sys
import soundfile as sf
from core import ROOT, SR, GEMINI, QWEN, FISH, TTS_CONCURRENCY, FIT_CONCURRENCY, config, command, probe, atomic_json, LANGUAGES, TARGET_LANGUAGES,video_languages
from storage import Store
from local_resources import local_resources
from separation import separate,cache_directory,MODELS,model_ready
from speech import transcribe,TRANSCRIPTION_VERSION
from render import prepare_segments, synthesize_all, mix, mux, speech_placements, TRANSITION_SECONDS, REPLACEMENT_MARGIN_SECONDS, replacement_intervals, MIX_VERSION, output_guard_policy
from timing import FIT_VERSION, TIMING_POLICY, NATURAL_MIN_TEMPO, NATURAL_MAX_TEMPO, MAX_SLOW_DURATION_FACTOR
from reference_timing import refine_references, VERSION as REFERENCE_TIMING_VERSION
from stretch import close_worker, ENGINE as TEMPO_ENGINE, VERSION as TEMPO_VERSION
from mfa_hint import close_worker as close_mfa_worker
from scribe import transcribe_words


def run(video_id,action='full',segment_index=None):
    store=Store()
    job=store.get(video_id)
    if job is None:raise RuntimeError('Video not found.')
    if action in {'tts','render','mix','watch'}:
        from studio_workflow import generate
        return generate(video_id,action,segment_index)
    work=ROOT/'work'/'videos'/video_id
    if action=='analyze' and (work/'editor/draft.json').exists() and (work/'editor/resume-completed.json').exists():
        from studio_workflow import generate
        return generate(video_id,'preview')
    output=ROOT/'outputs'/video_id
    output.mkdir(parents=True,exist_ok=True)
    recovered_refresh=work/'dialogue-refresh.json'
    if action=='full' and recovered_refresh.exists() and not json.loads(recovered_refresh.read_text(encoding='utf8')).get('complete'):
        from dialogue_refresh import execute
        return execute(video_id,store)
    refresh=work/'audio-refresh.json'
    if action=='full' and refresh.exists() and not json.loads(refresh.read_text(encoding='utf-8')).get('complete'):
        from audio_refresh import execute
        return execute(video_id,store)
    def phase(label,low,high):
        def update(percent,detail=''):
            store.update(video_id,phase=label,phase_progress=round(percent,1),progress=round(low+(high-low)*percent/100,1),detail=detail)
        return update
    try:
        cfg=config()
        from audio_tracks import selected
        selected_audio_stream=selected(video_id)
        # Browser playback must not depend on the upload's original MP4 layout.
        # Keep this optional preparation from interrupting the dubbing job.
        try:
            from browser_preview import prepare as prepare_browser_preview
            prepare_browser_preview(video_id)
        except Exception as error:
            atomic_json(work/'browser-preview-error.json',{'error':str(error)[:500]})
        # Rebuilds can retain the separation used by an approved preview
        # without changing the defaults for other videos.
        run_config=work/'run-config.json'
        if run_config.exists():
            settings=json.loads(run_config.read_text(encoding='utf-8'))
            if not isinstance(settings,dict) or not set(settings)<={'language','separator','background'}:
                raise RuntimeError('Invalid per-video regeneration configuration.')
            cfg.update(settings)
        cfg=video_languages(cfg,job)
        if not cfg['key']:raise RuntimeError('OPENROUTER_API_KEY is not configured.')
        if cfg['language'] not in TARGET_LANGUAGES:raise RuntimeError('Invalid TARGET_LANGUAGE configuration.')
        if cfg['separator'] not in MODELS:raise RuntimeError('Invalid SEPARATION_MODEL configuration.')
        if not model_ready(cfg['separator']):raise RuntimeError('The selected separation model is not installed.')
        stage_work=cache_directory(work,cfg['separator'])
        source=work/'source.mp4'
        info=probe(source)
        if not any(s['codec_type']=='audio' for s in info['streams']):raise RuntimeError('No audio track was found.')
        duration=float(info['format']['duration'])
        original,clean=work/'original.wav',stage_work/'dialogue.wav'
        if cfg['background'] not in {'leap-xe-vocals','mel-band-roformer-deux','hyperace-v2-instrumental','residual'}:raise RuntimeError('Invalid BACKGROUND_MODEL configuration.')
        dedicated=cfg['background']!='residual'
        leap_pair=cfg['separator']==cfg['background']=='leap-xe-vocals'
        unified=cfg['separator']==cfg['background']=='mel-band-roformer-deux'
        update=phase('1/4 음성 분리',0,32);update(0)
        with local_resources():
            if not original.is_file():
                from audio_tracks import extract
                extract(source,original,selected_audio_stream,duration)
            atomic_json(work/'waveform-source.json',{'path':str(original),'kind':'original'})
            from separation_process import prepare as prepare_separation
            clean,background=prepare_separation(original,stage_work,work,cfg,update)
            if sf.info(original).frames!=sf.info(clean).frames:raise RuntimeError('Separated audio duration does not match.')
            atomic_json(work/'waveform-source.json',{'path':str(clean),'kind':'clean'})
            update(100)
        transcription_model=GEMINI
        transcription_tag=cfg['diarizer']+'-native-fixed-text-speakers-gemini-translation-v1'
        dialogue_work=stage_work/f'full-audio-v{TRANSCRIPTION_VERSION}-{cfg["language"]}-{transcription_tag}'
        dialogue_work.mkdir(exist_ok=True)
        transcript=transcribe(clean,dialogue_work,dialogue_work/'segments.json',cfg,
            phase('2/4 전체 음성 전사·번역',32,52))
        if transcript.get('boundary_authority') in ('scribe_v2','voxtral_transcribe_3','diarizen_native','nemotron_eend_vc','mai_transcribe_2','fish_transcribe_1_pro','gemini_native_audio','qwen_omni_audio','fish_qwen_gemini','moss_transcribe_diarize'):source_transcription={}
        else:
            source_cache=stage_work/'scribe-source-v1';source_cache_file=source_cache/'scribe.json'
            source_language=json.loads(source_cache_file.read_text(encoding='utf-8')).get('signature',{}).get('language') if source_cache_file.exists() else None
            source_transcription=transcribe_words(clean,source_cache,cfg['key'],source_language,phase('2/4 참조 경계 측정용 ASR',52,55))
        phase('2/4 전사·번역',32,55)(100)
        transcript,reference_measurements=refine_references(clean,transcript,dialogue_work/f'reference-timing-v{REFERENCE_TIMING_VERSION}',
            phase('3/4 MOSS 원본 경계 사용' if cfg['diarizer']=='moss-gemini' else '3/4 원본 음성 구간 보정',55,60),source_transcription=source_transcription)
        from tts_prompt import with_transcript_languages
        cfg=with_transcript_languages(cfg,transcript)
        if action=='analyze':
            from studio_workflow import checkpoint
            checkpoint(video_id,transcript,{'source':str(source),'original':str(original),'clean':str(clean),
                'background':str(background) if background else None,'duration':duration,'language':cfg['language'],'audio_stream_index':selected_audio_stream})
            from studio_workflow import generate
            return generate(video_id,'preview')
        atomic_json(output/'segments.json',transcript)
        cfg['language']=transcript['target_language']
        segments=prepare_segments(transcript,sf.info(original).frames,source_transcription)
        boundary_gaps=[]
        # Keep each TTS model's generations and fitting caches independently.
        shared_references=cfg.get('tts_reference_mode') in ('same_speaker_shared_untagged','same_speaker_shared_tagged','same_speaker_shared_donor_tag')
        reference_mode='same-speaker-context-count-unlimited-target-first-donors-speaker1-per-line-language-wrapped-acting-instructions-gap05-outer05-individual-v29' if cfg.get('tts_reference_mode')=='same_speaker_tagged_context' else 'same-speaker-batch-min60seconds-unlimited-count-filled-untagged-natural-instructions-gap0-v12' if cfg.get('tts_reference_mode')=='same_speaker_batch' else 'same-speaker-context-min2-quality-donor-target-first-untagged-gap0-v8' if cfg.get('tts_reference_mode')=='same_speaker_context' else 'shared-donor-tag-target-first-v1' if cfg.get('tts_reference_mode')=='same_speaker_shared_donor_tag' else 'shared-tags-target-first-v1' if cfg.get('tts_reference_mode')=='same_speaker_shared_tagged' else 'shared-no-tags-v1' if shared_references else 'individual'
        pieces=dialogue_work/f'tts-{FISH.split("/")[-1]}-{reference_mode}-reference-v{REFERENCE_TIMING_VERSION}';pieces.mkdir(exist_ok=True)
        reference_plan=None;reference_bank_audit=None
        if shared_references and segments:
            if cfg.get('speaker_embeddings',False):
                from reference_bank_guard import prepare as verify_reference_banks
                reference_bank_audit=verify_reference_banks(segments,clean,pieces/'reference-bank-guard',
                    phase('3/4 참조 묶음 화자 검증',60,60),preserve_speaker_ids=bool(transcript.get('preserve_source_speaker_ids')))
                for decision in reference_bank_audit.get('changes',[]):
                    i=decision['index']
                    if segments[i].get('_speaker_locked'):continue
                    voice=str(decision['bank_speaker'])
                    segments[i]['speaker_id']=voice
                    transcript['segments'][i]['_asr_speaker_id']=transcript['segments'][i].get('_asr_speaker_id',transcript['segments'][i]['speaker_id'])
                    transcript['segments'][i]['speaker_id']=voice
                atomic_json(output/'segments.json',transcript)
            else:
                for segment in segments:
                    for key in ('_reference_bank_speaker','_reference_donor_indices'):segment.pop(key,None)
            from native_timeline import prepare_selected
            from shared_reference import prepare as prepare_shared_references
            timeline=transcript.get('speaker_timeline') or prepare_selected(clean,cfg)
            reference_plan=prepare_shared_references(segments,clean,pieces,timeline,
                phase('3/4 같은 화자 참조 준비',60,60))
        update=phase('3/4 번역 음성·쉼·겹침 보정',60,87)
        reports=synthesize_all(segments,clean,pieces,cfg,
            lambda done,running,queued,fitting,fit_waiting:update(done/max(1,len(segments))*100,
                f'완료 {done}/{len(segments)} · TTS {running}/{TTS_CONCURRENCY} · 쉼·겹침 보정 {fitting} · 대기 {queued+fit_waiting}'),status=update)
        original_fallbacks=[{'index':r['index']+1,'reason':r.get('reason'),'start':r['start'],'end':r['end']}
            for r in reports if r.get('keep_original')]
        output_reviews=[{'index':r['index']+1,'reason':r['tts_output_check'].get('reason'),
            'start':r['start'],'end':r['end']} for r in reports
            if not r.get('keep_original') and r.get('tts_output_check',{}).get('action')=='keep_uncertain']
        for item in original_fallbacks:
            transcript['segments'][item['index']-1].update(keep_original=True,keep_original_reason=item['reason'])
        atomic_json(output/'segments.json',transcript)
        update(100,f'{len(segments)}개 구간 완료')
        mixed=work/'mixed.wav'
        update=phase('4/4 영상 저장',87,100)
        mix(original,clean,segments,pieces,mixed,lambda p,d:update(p*.7,d),excluded_intervals=transcript.get('excluded_intervals',[]),
            background=background)
        update(75,'MP4 저장 중')
        mux(source,mixed,output/'final.mp4',cfg['language'],duration)
        atomic_json(output/'render.json',{'transcription_model':transcript.get('source_transcription_model',transcript.get('model',transcription_model)),'tts_model':FISH,'tts_concurrency':TTS_CONCURRENCY,
            'model_pipeline':transcript.get('models'),'source_request_counts':transcript.get('request_counts'),
            'source_boundary_policy':transcript.get('boundary_policy'),
            'fit_concurrency':FIT_CONCURRENCY,'tts_scheduling':'separate_generation_and_fitting_queues',
            'tts_generation_mode':cfg.get('tts_mode','individual'),
            'unresolved_collision_policy':'keep_original_per_segment_never_mix_long_overrun',
            'tts_generated_alignment':'waveform_reference_comparison_and_quiet_gap_crop' if cfg.get('tts_output_guard')=='waveform' else ('scribe_v2_verified_target_and_quiet_gap_crop' if cfg.get('tts_output_guard') else None),
            'tts_output_guard_policy':output_guard_policy(cfg),
            'tts_output_guard_batches':('individual_local_checks_parallel_to_generation_and_fitting' if cfg.get('tts_output_guard')=='waveform' else 'up_to_16_outputs_per_request_parallel_to_generation_and_fitting') if cfg.get('tts_output_guard') else None,
            'tts_output_guard_asr_requests':0 if cfg.get('tts_output_guard') in (None,'waveform') else None,
            'source_asr_model':transcript.get('source_asr_model'),
            'source_asr_input_policy':transcript.get('source_asr_input_policy'),
            'fish_speaker_policy':transcript.get('fish_speaker_policy'),
            'tts_reference_mode':cfg.get('tts_reference_mode','individual'),
            'tts_reference_policy':reference_plan['policy'] if reference_plan else None,
            'tts_reference_bank_verification':reference_bank_audit,
            'tts_reference_groups':[{'speaker':g['speaker'],'indices':g['indices'],
                'reference_indices':g['reference_indices'],'shared':g['shared']}
                for g in reference_plan['groups']] if reference_plan else [],
            'tts_grouping_policy':None,
            'tts_grouping_counts':None,
            'transcription_mode':transcript.get('transcription_mode'),'transcription_version':TRANSCRIPTION_VERSION,
            'translation_output':transcript.get('translation_output'),'source_segments':transcript.get('boundary_authority','gemini_audio'),
            'original_text_source':transcript.get('original_text_source','gemini'),
            'reference_speaker_boundaries':transcript.get('reference_speaker_boundaries'),
            'reference_tail_policy':transcript.get('reference_tail_policy'),
            'llm_timestamps':transcript.get('llm_timestamps',False),'alignment_similarity_model':transcript.get('similarity_model'),
            'asr_sent_to_llm':transcript.get('asr_sent_to_llm',True),'overlapping_dialogue':'shared_segment','original_audio_fallbacks':original_fallbacks,
            'tts_output_reviews':output_reviews,
            'speaker_authority':transcript.get('speaker_authority','gemini_audio_only'),
            'speaker_linking_model':transcript.get('speaker_timeline',{}).get('external_embedding_model'),
            'speaker_linking_policy':transcript.get('speaker_timeline',{}).get('identity',{}).get('packed_native_linking',transcript.get('speaker_timeline',{}).get('identity',{}).get('policy')),
            'emotion_segmentation':'disabled',
            'separation_model':MODELS[cfg['separator']],'separation_model_id':cfg['separator'],
            'background_model':cfg['background'],'background_source':'original_minus_leap_xe_vocals' if cfg['background']=='leap-xe-vocals' else ('dedicated_instrumental' if dedicated else 'original_minus_dialogue'),
            'separation_mode':'leap_xe_vocals_and_residual_single_pass' if leap_pair else ('deux_dual_single_pass' if unified else 'independent_models'),
            'timing_fit_version':FIT_VERSION,'timing_fit_policy':TIMING_POLICY,
            'pause_adjustment_enabled':True,'mfa_used':any(r.get('mfa_status')=='aligned' for r in reports),
            'pause_planning':'algorithmic_pause_only','pause_planning_api_requests':0,
            'mfa_scheduling':'single_resident_worker_cached_alignment','speed_processing':'google_speedy_only_for_next_dialogue_overlap',
            'speed_scheduling':'offline_wsl_cpu_per_utterance','speed_concurrency':FIT_CONCURRENCY,'audio_prepare_concurrency':FIT_CONCURRENCY,
            'source_boundary_gaps':boundary_gaps,
            'reference_timing_version':REFERENCE_TIMING_VERSION,'reference_measurements':reference_measurements,
            'reference_timing_model':transcript.get('timing_model'),'reference_timing_usage':transcript.get('timing_usage'),
            'source_prefix_recovery':transcript.get('source_prefix_recovery',[]),
            'source_prefix_recovery_version':transcript.get('source_prefix_recovery_version'),
            'unmatched_source_segments':transcript.get('unmatched_segments',[]),
            'speed_correction_engine':TEMPO_ENGINE,'speed_correction_engine_version':TEMPO_VERSION,
            'speed_tempo_limits':[NATURAL_MIN_TEMPO,NATURAL_MAX_TEMPO],
            'max_slowdown_duration_factor':MAX_SLOW_DURATION_FACTOR,'speedup_unlimited':True,'slowdown_unlimited':False,
            'tts_overlap_allowed':any(r.get('overlap_allowed',False) for r in reports),
            'tts_duration_policy':'pause_first_then_speedy_on_collision_keep_valid_audio_on_speedy_failure',
            'speed_correction_skipped_segments':[r['index']+1 for r in reports if r.get('speed_correction_status')=='skipped_keep_pause_audio'],
            'tts_placements':[{'start':p['lo']/SR,'end':p['hi']/SR} for p in speech_placements(segments,pieces)],
            'output_audio_seconds':sf.info(mixed).duration,
            'video_save':json.loads((output/'mux-validation.json').read_text(encoding='utf8')),
            'mix_version':MIX_VERSION,'replacement_end_policy':'max_reference_end_and_fitted_tts_end',
            'fade_seconds':TRANSITION_SECONDS,'fade_placement':'outside_padded_replacement_interval','tts_fade_seconds':0,
            'replacement_margin_seconds':REPLACEMENT_MARGIN_SECONDS,
            'replacement_intervals':[{'start':s['lo']/SR,'end':s['hi']/SR} for s in replacement_intervals(segments,sf.info(original).frames,speech_placements(segments,pieces))],
            'inside_replacement_margin':'background_only',
            'overlapping_margins':'maximum_voice_removal','reference_source':'separated_dialogue','outside_transition_regions':'original_audio',
            'reference_separator':cfg['separator'],
            'inside_segments':'background_plus_tts','segments':reports})
        unmatched=len(transcript.get('unmatched_segments',[]))
        kept_failed=sum(r['reason']!='no_lexical_dubbing_text' for r in original_fallbacks)
        detail=f'원음 유지 {unmatched+kept_failed}개 · 나머지 더빙 완료 (segments.json·render.json 참고)' if unmatched+kept_failed else ''
        store.update(video_id,state='done',phase='완료',phase_progress=100,progress=100,detail=detail,error=None)
    except Exception as error:
        import traceback
        traceback.print_exc()
        message=str(error) if isinstance(error,RuntimeError) else '처리 중 오류가 발생했습니다. 로그와 입력 파일을 확인해 주세요.'
        store.update(video_id,state='error',phase='실패',detail='',error=message)
        return 1
    finally:
        try:close_worker()
        finally:close_mfa_worker()
    return 0


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('video_id');parser.add_argument('--action',choices=['full','analyze','tts','render','mix','watch'],default='full');parser.add_argument('--segment',type=int)
    args=parser.parse_args();sys.exit(run(args.video_id,args.action,args.segment))
