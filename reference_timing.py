"""Measure replacement/reference boundaries from aligned source audio.

No per-video overrides, segment numbers or absolute-time exceptions. Unmatched
text is preserved; timestamps come exclusively from ASR and measured audio.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from local_resources import local_stage
import soundfile as sf
from core import atomic_json

VERSION=26
LANGUAGES={'ja':'Japanese','japanese':'Japanese','en':'English','english':'English',
    'ko':'Korean','korean':'Korean','zh':'Chinese','chinese':'Chinese','yue':'Cantonese','cantonese':'Cantonese',
    'fr':'French','french':'French','de':'German','german':'German','it':'Italian','italian':'Italian',
    'pt':'Portuguese','portuguese':'Portuguese','ru':'Russian','russian':'Russian','es':'Spanish','spanish':'Spanish'}

from reference_edges import acoustic_boundaries,validated_boundaries
from source_alignment import source_evidence,VERSION as ALIGNMENT_VERSION
from semantic_alignment import VERSION as SEMANTIC_VERSION
from reference_clean import reference_windows
from timestamp_calibration import calibrate_word_times
from source_prefix import recover_prefixes,VERSION as PREFIX_VERSION
from translation_alignment import VERSION as TRANSLATION_ALIGNMENT_VERSION
from source_recovery import recover as recover_source,VERSION as RECOVERY_VERSION

def exclusions_outside_references(intervals,windows,total_frames,sr):
    """Keep estimated exclusions outside the authoritative final references."""
    ranges=sorted((w['start_frame'],w['end_frame']) for w in windows)
    result=[];adjustments=[]
    for interval in intervals:
        lo=max(0,round(interval['start']*sr));hi=min(total_frames,round(interval['end']*sr))
        if hi<=lo:continue
        cursor=lo;parts=[]
        for start,end in ranges:
            if end<=cursor:continue
            if start>=hi:break
            if start>cursor:parts.append((cursor,min(start,hi)))
            cursor=max(cursor,end)
            if cursor>=hi:break
        if cursor<hi:parts.append((cursor,hi))
        kept=[{**interval,'start':start/sr,'end':end/sr} for start,end in parts if end>start]
        result.extend(kept)
        if parts!=[(lo,hi)]:
            adjustments.append({'estimated_interval':dict(interval),'kept_intervals':kept,
                'reason':'final_reference_has_priority'})
    return sorted(result,key=lambda x:(x['start'],x['end'])),adjustments


@local_stage
def refine_references(clean,transcript,folder,progress=lambda *args:None,source_transcription=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if transcript.get('boundary_authority') in ('moss_transcribe_diarize','voxtral_transcribe_3'):
        atomic_json(folder/'measured.json',{'transcript':transcript,'measurements':[]})
        progress(100,'원본 전사 시작·끝 그대로 사용 · 추가 경계 보정 없음')
        return transcript,[]
    if transcript.get('boundary_authority') in ('scribe_v2','mai_transcribe_2','fish_transcribe_1_pro','gemini_native_audio','qwen_omni_audio','fish_qwen_gemini'):
        measurements=transcript.get('boundary_measurements',[])
        atomic_json(folder/'measured.json',{'transcript':transcript,'measurements':measurements})
        progress(100,'파형 경계 보정 완료 · 측정한 참조 경계 사용')
        return transcript,measurements
    if transcript.get('boundary_authority') in ('diarizen_native','nemotron_eend_vc'):
        from native_timeline import references
        return references(clean,transcript,folder,progress)
    if not transcript['segments']:return dict(transcript),[]
    languages={LANGUAGES.get(x.lower()) for x in transcript.get('source_languages',[])}
    language=languages.pop() if len(languages)==1 else None
    identity=hashlib.sha256(Path(clean).read_bytes()+json.dumps([VERSION,ALIGNMENT_VERSION,SEMANTIC_VERSION,PREFIX_VERSION,TRANSLATION_ALIGNMENT_VERSION,RECOVERY_VERSION,'standard_external_fade',transcript],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    cached=folder/'measured.json'
    if cached.exists():
        data=json.loads(cached.read_text(encoding='utf-8'))
        if data.get('identity')==identity:
            result=dict(data['transcript']);result['transition_policy']='standard_external_fade'
            return result,data['measurements']
    input_file=folder/'input.json';atomic_json(input_file,transcript)
    progress(0,'원문과 분리 음성 정렬')
    code={'Japanese':'ja','English':'en','Korean':'ko','Chinese':'zh','Cantonese':'yue','French':'fr','German':'de','Italian':'it','Portuguese':'pt','Russian':'ru','Spanish':'es'}.get(language)
    evidence=source_evidence(Path(clean),transcript['segments'],folder,code,lambda p,d:progress(p*.85,d),transcription=source_transcription)
    values,sr=sf.read(clean,dtype='float32',always_2d=True)
    calibrated,calibration=calibrate_word_times(values,sr,evidence['segments'])
    translation_only=transcript.get('translation_output')=='translated_strings_only'
    unmatched=list(evidence['unmatched_segments']);cascade_reports=[];measurement_failures=[]
    def keep_unmeasured(failures,stage):
        known={m['source_index'] for m in unmatched}
        for failure in failures:
            measurement_failures.append(dict(failure,stage=stage))
            index=failure['source_index']
            if index not in known:
                unmatched.append(dict(source_index=index,original_text='',translated_text=transcript['segments'][index]['translated_text'],reason=failure['reason']))
                known.add(index)
        atomic_json(folder/'boundary-review.json',{'failures':measurement_failures})
    if translation_only:
        prefix_reports=[]
        calibrated,preliminary,failed=validated_boundaries(values,sr,calibrated,progress)
        keep_unmeasured(failed,'initial_acoustic_measurement')
        calibrated,unmatched,cascade_reports=recover_source(clean,values,sr,transcript,calibrated,
            preliminary,unmatched,evidence.get('asr',[]),folder/'asr-recovery',
            (source_transcription or {}).get('language') or language,
            lambda p,d:progress(85+p*.04,d))
    else:
        try:neighbor_measurements=acoustic_boundaries(values,sr,calibrated)
        except RuntimeError:neighbor_measurements=None
        alignment_language=(source_transcription or {}).get('language') or language
        calibrated,prefix_reports=recover_prefixes(clean,values,sr,transcript['segments'],calibrated,
            folder/'source-prefix',alignment_language,lambda p,d:progress(85+p*.04,d),
            neighbor_measurements=neighbor_measurements)
    progress(90,'발성·잔향·인접 대사 경계 측정')
    fallbacks=[]
    atomic_json(folder/'waveform-fallback.json',{'reports':fallbacks,'unresolved':unmatched})
    calibrated,measured,failed=validated_boundaries(values,sr,calibrated,progress)
    keep_unmeasured(failed,'final_acoustic_measurement')
    eligible=[{**transcript['segments'][a['source_index']],
        **({'original_text':a['recognized_text'],'translated_text':a['translated_text'],
            'asr_speaker_label':a.get('asr_speaker_label'),
            'speaker_boundary_split':a.get('speaker_boundary_split',False)} if translation_only else {})} for a in calibrated]
    if len(measured)!=len(eligible):raise RuntimeError('Linked dialogue counts do not match.')
    result=dict(transcript);result['segments']=[]
    for segment,row,source in zip(eligible,measured,calibrated):
        row['source_index']=source['source_index']
        row.update(original_start=source.get('located_start',source['start']),
            original_end=source.get('located_end',source['end']))
        # Gemini exclusions are estimates; resolve their overlap only after
        # cleanup, without cutting or rejecting the measured reference.
        if row['end']<=row['start']:raise RuntimeError('Invalid measured reference duration.')
        row['start_frame']=max(0,round(row['start']*sr));row['end_frame']=min(len(values),round(row['end']*sr))
        row['start']=row['start_frame']/sr;row['end']=row['end_frame']/sr;row['duration']=(row['end_frame']-row['start_frame'])/sr
        result['segments'].append({**segment,'start':row['start'],'end':row['end']})
    result['timing_method']='translated_dialogue_to_scribe_cross_language_semantic_alignment_and_acoustic_boundaries' if translation_only else 'scribe_semantic_alignment_text_supported_onset_recovery_and_acoustic_boundaries'
    result['original_text_source']='scribe' if translation_only else 'gemini'
    result['reference_speaker_boundaries']='scribe' if translation_only else 'legacy_alignment'
    result['similarity_model']=evidence.get('similarity_model')
    result['timing_model']=evidence['model'];result['timing_usage']=evidence['usage']
    result['timestamp_calibration']=calibration
    result['source_prefix_recovery']=prefix_reports
    result['source_prefix_recovery_version']=PREFIX_VERSION
    result['asr_recovery']=cascade_reports
    result['asr_recovery_version']=RECOVERY_VERSION
    result['boundary_measurement_reviews']=measurement_failures
    result['unmatched_segments']=unmatched
    result['waveform_fallbacks']=fallbacks
    result['reference_timing_version']=VERSION
    result['reference_windows']=reference_windows(values,sr,measured)
    preserved=[]
    for segment,row,window in zip(result['segments'],measured,result['reference_windows']):
        row['measured_start_before_cleanup']=row['start'];row['measured_end_before_cleanup']=row['end']
        row.update(start=window['start'],end=window['end'],start_frame=window['start_frame'],
            end_frame=window['end_frame'],duration=window['duration'],boundary_policy=window['boundary_policy'])
        segment.update(start=window['start'],end=window['end'])
        row['tts_reference']=window
        preserved.extend(window['preserved_intervals'])
    result['boundary_policy']='shared_clean_interval'
    result['excluded_intervals'],result['excluded_interval_adjustments']=exclusions_outside_references(
        transcript.get('excluded_intervals',[]),result['reference_windows'],len(values),sr)
    result['reference_tail_intervals']=preserved
    result['transition_policy']='standard_external_fade'
    result['excluded_timing_policy']='final_reference_has_priority'
    atomic_json(cached,{'identity':identity,'transcript':result,'measurements':measured,'gemini_timestamp_fallback_count':0})
    progress(100,'원본 음성 구간 보정 완료')
    return result,measured
