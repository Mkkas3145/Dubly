from __future__ import annotations
import base64
from contextlib import ExitStack
import hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import re
import os
from pathlib import Path
from local_resources import local_stage
import threading
import time
import urllib.error
import urllib.request
import numpy as np
import soundfile as sf
from core import SR, FISH, LANGUAGES,LANGUAGE_CATALOG, TTS_CONCURRENCY, FIT_CONCURRENCY, atomic_json, command, probe, config
from timing import FIT_VERSION, TIMING_POLICY, bounded_timing_audio
from reference_clean import edge_guard,VERSION as REFERENCE_CLEAN_VERSION
from mix_levels import TTS_GAIN
from audio_edges import reference_fade_edges,FADE_SECONDS

TRANSITION_SECONDS = .1
REPLACEMENT_MARGIN_SECONDS = 0.
MIX_VERSION = 11
MUX_AUDIO_ENCODERS=('aac_mf','aac') if os.name=='nt' else ('aac',)
_TTS_SLOTS = threading.BoundedSemaphore(TTS_CONCURRENCY)
REFERENCE_MARGIN_POLICY='zero-padding-acoustic-voice-release-no-reference-fades-v3'
# Keep sampling defaults and the normal quality-oriented latency mode.
TTS_PROVIDER_OPTIONS={'latency':'normal','temperature':1.0,'top_p':1.0,'condition_on_previous_chunks':True}


def pipelined_queue(items, generator, fitter, progress):
    """Separate network and fitting slots, with a bounded on-disk fitting backlog."""
    pending=iter(enumerate(items))
    generating,fitting,results,errors={},{},{},[]
    ready=deque()
    submitted=0
    with ThreadPoolExecutor(max_workers=TTS_CONCURRENCY,thread_name_prefix='fish') as tts_pool, \
         ThreadPoolExecutor(max_workers=FIT_CONCURRENCY,thread_name_prefix='timing') as fit_pool:
        def fill():
            nonlocal submitted
            while ready and len(fitting)<FIT_CONCURRENCY:
                index,prepared=ready.popleft()
                fitting[fit_pool.submit(fitter,prepared)]=index
            while (not errors and len(generating)<TTS_CONCURRENCY and
                   len(ready)+len(generating)<2*TTS_CONCURRENCY):
                item=next(pending,None)
                if item is None:break
                index,value=item
                generating[tts_pool.submit(generator,index,value)]=index
                submitted+=1
        def update():progress(len(results),len(generating),len(items)-submitted,len(fitting),len(ready))
        fill();update()
        while generating or fitting:
            completed,_=wait([*generating,*fitting],return_when=FIRST_COMPLETED)
            for future in completed:
                is_generation=future in generating
                index=(generating if is_generation else fitting).pop(future)
                try:
                    value=future.result()
                    if is_generation:ready.append((index,value))
                    else:results[index]=value
                except Exception as error:errors.append((index,error,'음성 생성' if is_generation else '오디오 후처리·저장'))
            fill();update()
    if errors:
        index,error,stage=errors[0]
        if isinstance(error,RuntimeError):detail=str(error)
        elif isinstance(error,PermissionError):detail=f'{stage} 중 파일 접근이 거부됐습니다. 잠금·권한을 확인해 주세요.'
        elif isinstance(error,FileNotFoundError):detail=f'{stage} 중 필요한 작업 파일을 찾지 못했습니다.'
        elif isinstance(error,json.JSONDecodeError):detail=f'{stage} 중 저장된 JSON을 읽지 못했습니다.'
        else:detail=f'{stage} 오류 ({type(error).__name__}): {error}'
        raise RuntimeError(f'TTS dialogue {index+1}: {detail}') from error
    return [results[n] for n in range(len(items))]


def tts_instructions(language,brief=False,source_language=None,structured=False):
    from tts_prompt import instructions,structured_instructions
    return (structured_instructions if structured else instructions)(language,source_language)


def tts_policy(language,provider_options=None,source_language=None,instructions=None):
    from rich_transcript import TTS_CUE_POLICY
    from same_speaker_tts import POLICY as GROUPING_POLICY
    from shared_reference import POLICY as SHARED_POLICY
    from batch_tts import CONTEXT_REFERENCE_POLICY,BATCH_REFERENCE_POLICY
    from tagged_context_reference import POLICY as TAGGED_CONTEXT_POLICY
    policy=[FISH,language,instructions if instructions is not None else tts_instructions(language,source_language=source_language),REFERENCE_MARGIN_POLICY,provider_options or TTS_PROVIDER_OPTIONS,GROUPING_POLICY,SHARED_POLICY,config()['tts_reference_mode'],CONTEXT_REFERENCE_POLICY,BATCH_REFERENCE_POLICY,TAGGED_CONTEXT_POLICY,config().get('tts_delivery_cue'),config().get('tts_omit_instructions',False)]
    return hashlib.sha256(json.dumps([*policy,TTS_CUE_POLICY],ensure_ascii=False).encode()).hexdigest()


def delivery_text(text,cue):
    if not cue:return text
    marker='['+cue+']'
    lines=[]
    for line in text.split('\n'):
        if not line.strip():lines.append(line);continue
        match=re.match(r'^(<\|speaker:\d+\|>)?(.*)$',line)
        prefix=match[1] or '';body=match[2]
        lines.append(prefix+(body if body.startswith(marker) else marker+' '+body))
    return '\n'.join(lines)


def request_audio(reference,segment,language,key,destination,brief=False):
    # All requests share the configured API concurrency limit.
    with _TTS_SLOTS:
        return _request_audio(reference,segment,language,key,destination,brief)


def _request_audio(reference,segment,language,key,destination,brief=False):
    from same_speaker_tts import tagged_request
    reference_text,translated_text=tagged_request(segment)
    if '_shared_reference_text' in segment:
        reference_text=segment['_shared_reference_text']
        translated_text=segment['translated_text']
        if segment.get('_shared_reference_output_tag')==0:
            from rich_transcript import without_speakers
            translated_text='<|speaker:0|>'+without_speakers(translated_text)
    from rich_transcript import without_speakers
    paired_tags=bool(segment.get('_preserve_speaker_tags'))
    if not segment.get('_preserve_reference_tags',paired_tags) and not segment.get('_shared_reference_donor_tag'):
        reference_text=without_speakers(reference_text)
    if not paired_tags:
        translated_text=without_speakers(translated_text)
    translated_text=delivery_text(translated_text,segment.get('_delivery_cue'))
    if segment.get('_reference_language_wrappers'):
        from tagged_context_reference import request_texts
        reference_text,translated_text=request_texts(reference_text,translated_text,segment.get('_source_language'),language,segment.get('_reference_language_parts'))
    provider_options=dict(TTS_PROVIDER_OPTIONS)
    for k,v in segment.get('_provider_options',{}).items():
        if k not in ('temperature','top_p','condition_on_previous_chunks'):continue
        if k=='condition_on_previous_chunks' and v is not None and type(v) is not bool:
            raise ValueError('condition_on_previous_chunks must be boolean')
        if v is None:provider_options.pop(k,None)
        else:provider_options[k]=v
    payload={'model':FISH,'input':translated_text,'response_format':'mp3',
        'input_references':[{'type':'input_audio','input_audio':{'data':'data:audio/wav;base64,'+base64.b64encode(reference.read_bytes()).decode()}},
                            {'type':'text','text':reference_text}],
        'provider':{'options':{'fish-audio':provider_options}}}
    payload['instructions']=segment.get('_instructions',tts_instructions(language,brief=brief,source_language=segment.get('_source_language'),structured=bool(segment.get('_structured_instructions',segment.get('_reference_language_wrappers')))))
    if segment.get('_omit_instructions'):payload.pop('instructions',None)
    if not reference_text.strip():
        payload['input_references']=[item for item in payload['input_references'] if item['type']!='text']
    request=urllib.request.Request('https://openrouter.ai/api/v1/audio/speech',data=json.dumps(payload).encode(),method='POST',
        headers={'Authorization':f'Bearer {key}','Content-Type':'application/json','X-Title':'Fast Dubbing'})
    temporary=destination.with_suffix('.download')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=240) as response,temporary.open('wb') as target:
                size=0
                while block:=response.read(1024*1024):
                    size+=len(block)
                    if size>64*1024**2:raise RuntimeError('TTS audio is too large.')
                    target.write(block)
                generation=response.headers.get('X-Generation-Id')
            duration=float(probe(temporary)['format'].get('duration') or 0)
            if not math.isfinite(duration) or duration<=0 or (not segment.get('_joint_generation') and duration>120):raise RuntimeError('Invalid TTS audio duration.')
            temporary.replace(destination)
            return {'generation_id':generation,'generated_seconds':duration,'tts_policy':tts_policy(language,provider_options,segment.get('_source_language'),payload.get('instructions')),
                    'tts_provider_options':provider_options,'tts_delivery_cue':segment.get('_delivery_cue'),'instructions':payload.get('instructions')}
        except urllib.error.HTTPError as error:
            code=error.code
            body=error.read().decode('utf8','replace').replace(key,'[REDACTED]');error.close()
            atomic_json(destination.with_suffix('.http-error.json'),{'status':code,'body':body[:16000]})
            if (code==429 or code>=500) and attempt<2:
                time.sleep(2**(attempt+1));continue
            raise RuntimeError(f'Fish Audio HTTP {code}. Check API key, credits, and model access.') from None
        except (urllib.error.URLError,TimeoutError):
            raise RuntimeError('Fish Audio connection or response timed out. Completed segments were preserved.') from None


def reference_audio(clean,segment,destination):
    lo=segment.get('reference_lo',segment['lo']);hi=segment.get('reference_hi',segment['hi'])
    if not segment['lo']<=lo<hi<=segment['hi']:raise RuntimeError('TTS reference exceeds the dialogue replacement interval.')
    with sf.SoundFile(clean) as audio:
        audio.seek(lo)
        values=audio.read(hi-lo,dtype='float32',always_2d=True)
    if not len(values) or np.max(np.abs(values))<1e-6:raise RuntimeError('No audio exists in the reference interval.')
    mono=values.mean(axis=1)
    channel=None
    if np.sqrt(np.mean(mono**2))<.25*np.sqrt(np.mean(values**2)):
        channel=int(np.argmax(np.mean(values**2,axis=0)))
        mono=values[:,channel]
    mono=reference_fade_edges(mono,SR)
    sf.write(destination,mono,SR,subtype='PCM_16')


def output_guard_policy(cfg):
    if cfg.get('tts_output_guard')=='waveform':
        from waveform_output_guard import POLICY
        return POLICY
    if cfg.get('tts_output_guard')=='scribe-v2':
        from tts_output_guard import POLICY
        return POLICY
    return None


def fits_next_dialogue(report,collision_budget=None):
    budget=report.get('collision_budget_frames') if collision_budget is None else collision_budget
    return budget is None or report.get('output_frames',0)<=budget+round(.02*SR)

def generate_speech(index,segment,clean,folder,cfg,force_prepared=False):
    segment={**segment,'_omit_instructions':segment.get('_omit_instructions',cfg.get('tts_omit_instructions',False)),'_delivery_cue':cfg.get('tts_delivery_cue'),'_provider_options':{**cfg.get('tts_provider_options',{}),**segment.get('_provider_options',{})}}
    segment.setdefault('_source_language',cfg.get('source_language'))
    segment.setdefault('_instructions',tts_instructions(cfg['language'],source_language=segment['_source_language'],structured=bool(segment.get('_structured_instructions',segment.get('_reference_language_wrappers')))))
    reference=folder/f'{index:04d}-reference.wav'
    generated=folder/f'{index:04d}-tts.mp3'
    fitted=folder/f'{index:04d}-fitted.wav'
    metadata=folder/f'{index:04d}.json'
    effective_options={**TTS_PROVIDER_OPTIONS,**segment['_provider_options']}
    policy=tts_policy(cfg['language'],effective_options,segment['_source_language'],segment['_instructions'])
    generation_cache=folder/f'{index:04d}-generation.json'
    generation=json.loads(generation_cache.read_text(encoding='utf-8')) if generation_cache.is_file() else {}
    reference_audio(clean,segment,reference)
    request_reference=Path(segment.get('_shared_reference_path',reference))
    reference_source='shared_same_speaker_separated_dialogue' if '_shared_reference_path' in segment else 'separated_dialogue'
    with request_reference.open('rb') as source:reference_sha256=hashlib.file_digest(source,'sha256').hexdigest()
    generation_input=hashlib.sha256(json.dumps({**{k:segment[k] for k in ['original_text','translated_text']},
        **({'reference_parts':segment['_tts_parts']} if '_tts_parts' in segment else {}),
        'reference_text':segment.get('_shared_reference_text',segment['original_text']),
        'reference_sha256':reference_sha256},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    reusable_generation=(generated.is_file() and generation.get('tts_policy')==policy and
        generation.get('generation_input')==generation_input and
        generation.get('reference_source','separated_dialogue')==reference_source)
    window=[segment.get('speech_lo',segment['lo'])-segment['lo'],segment.get('speech_hi',segment['hi'])-segment['lo']]
    frames=segment['hi']-segment['lo']
    next_start=segment.get('next_dialogue_start_frame')
    collision_budget=None if next_start is None else next_start-segment['lo']
    if not force_prepared and metadata.is_file() and fitted.is_file():
        cached=json.loads(metadata.read_text(encoding='utf-8'))
        audio=sf.info(fitted)
        if (reusable_generation and cached.get('tts_policy')==policy and
            cached.get('tts_output_guard_policy')==output_guard_policy(cfg) and
            cached.get('fit_version')==FIT_VERSION and cached.get('timing_method')==TIMING_POLICY and
            cached.get('output_frames')==audio.frames and audio.samplerate==SR and audio.channels==2 and
            cached.get('speech_window_frames')==window and
            cached.get('collision_budget_frames')==collision_budget and
            fits_next_dialogue(cached,collision_budget)):return {'cached':cached}
    timing_reference=reference
    if 'reference_lo' in segment or segment.get('natural_reference_margin'):
        timing_reference=folder/f'{index:04d}-timing-reference.wav'
        reference_audio(clean,{'lo':segment['lo'],'hi':segment['hi']},timing_reference)
    if not reusable_generation:
        generation=request_audio(request_reference,segment,cfg['language'],cfg['key'],generated)
        generation['tts_policy']=policy
        generation['reference_source']=reference_source
        atomic_json(generation_cache,generation)
    generation['generation_input']=generation_input
    generation['reference_sha256']=reference_sha256
    if '_shared_reference_indices' in segment:
        generation['shared_reference_indices']=segment['_shared_reference_indices']
        generation['shared_reference_speaker']=segment['_shared_reference_speaker']
        generation['reference_tags_preserved']=bool(segment.get('_preserve_reference_tags') or segment.get('_preserve_speaker_tags'))
        generation['output_tags_preserved']=bool(segment.get('_preserve_speaker_tags'))
    if '_tts_parts' in segment:generation['reference_parts']=segment['_tts_parts']
    atomic_json(generation_cache,generation)
    return dict(index=index,segment=segment,folder=folder,reference=reference,timing_reference=timing_reference,
        generated=generated,fitted=fitted,metadata=metadata,generation_cache=generation_cache,
        generation=generation,policy=policy,frames=frames,window=window,collision_budget=collision_budget)


@local_stage
def fit_speech(prepared,cfg):
    if 'cached' in prepared:
        if not fits_next_dialogue(prepared['cached']):raise RuntimeError('Cached dubbing overlaps the next dialogue. Preserving source audio for this dialogue.')
        return prepared['cached']
    if prepared.get('_guard_error'):raise RuntimeError('생성 음성 검증: '+prepared['_guard_error'])
    index,segment,folder=(prepared[k] for k in ('index','segment','folder'))
    reference,timing_reference,generated,fitted=(prepared[k] for k in ('reference','timing_reference','generated','fitted'))
    generation,generation_cache,policy=(prepared[k] for k in ('generation','generation_cache','policy'))
    frames,window=prepared['frames'],prepared['window']
    from rich_transcript import spoken
    timing=bounded_timing_audio(generated,fitted,frames,timing_reference,spoken(segment['translated_text']),cfg['language'],speech_window=window,
        collision_budget_frames=prepared['collision_budget'],preserve_outer_edges=True,
        protected_spoken_end_frames=generation.get('tts_protected_spoken_end_frames'),preserve_tail=True)
    if not fits_next_dialogue(timing,prepared['collision_budget']):
        atomic_json(folder/f'{index:04d}-rejected-timing.json',timing)
        raise RuntimeError('Dubbed audio still overlaps the next dialogue after duration correction. Preserving source audio for this dialogue instead of overlapping playback.')
    timing.update(clip_edge_fade_seconds=FADE_SECONDS,clip_padding_seconds=0.)
    result={'index':index,'start':segment['start'],'end':segment['end'],**timing,**generation,
        'reference_clean_version':REFERENCE_CLEAN_VERSION,'reference_window_frames':[segment.get('reference_lo',segment['lo']),segment.get('reference_hi',segment['hi'])]}
    atomic_json(prepared['metadata'],result)
    return result


def synthesize(index,segment,clean,folder,cfg):
    return fit_speech(generate_speech(index,segment,clean,folder,cfg),cfg)


def synthesize_all(segments,clean,folder,cfg,progress,status=None):
    if cfg.get('tts_reference_mode')=='same_speaker_tagged_context':
        from tagged_context_reference import prepare
        prepare(segments,clean,folder)
        return synthesize_individual_all(segments,clean,folder,cfg,progress,status)
    if cfg.get('tts_reference_mode')=='same_speaker_context':
        from batch_tts import prepare_context_references
        prepare_context_references(segments,clean,folder)
        return synthesize_individual_all(segments,clean,folder,cfg,progress,status)
    if cfg.get('tts_mode')=='same_speaker_batches':
        from batch_tts import synthesize_speaker_batches
        return synthesize_speaker_batches(segments,clean,folder,cfg,progress,status)
    if cfg.get('tts_mode')=='individual':
        return synthesize_individual_all(segments,clean,folder,cfg,progress,status)
    if segments and all('_tts_group' in s for s in segments):
        from same_speaker_tts import synthesize_aligned
        return synthesize_aligned(segments,clean,folder,cfg,progress,status)
    return synthesize_individual_all(segments,clean,folder,cfg,progress,status)


def synthesize_individual_all(segments,clean,folder,cfg,progress,status=None):
    # Derive deadlines from neighboring timeline starts before parallel fitting.
    items=[{**s,'next_dialogue_start_frame':min((other['lo'] for other in segments if not other.get('deleted') and other['lo']>s['lo']),default=None)}
        for i,s in enumerate(segments)]
    def update(done,running,queued,fitting,waiting):
        generating_label='TTS 묶음' if any(len(s.get('_tts_parts',[]))>1 for s in segments) else 'TTS'
        if status:status(done/max(1,len(segments))*100,
            f'완료 {done}/{len(segments)} · {generating_label} {running}/{TTS_CONCURRENCY} · 쉼·겹침 보정 {fitting} · 대기 {queued+waiting}')
        else:progress(done,running,queued,fitting,waiting)
    def keep_original(index,reason):
        segment=segments[index];segment['keep_original']=True
        sf.write(folder/f'{index:04d}-fitted.wav',np.zeros((1,2),dtype=np.float32),SR,subtype='FLOAT')
        report={'index':index,'start':segment['start'],'end':segment['end'],
            'keep_original':True,'reason':reason,'output_frames':1}
        atomic_json(folder/f'{index:04d}.json',report)
        return {'index':index,'cached':report}
    def generate(index,segment):
        if segment.get('keep_original'):return keep_original(index,segment.get('keep_original_reason','source_speaker_boundary_uncertain'))
        if not any(c.isalnum() for c in segment['translated_text']):return keep_original(index,'no_lexical_dubbing_text')
        try:return generate_speech(index,segment,clean,folder,cfg)
        except RuntimeError as error:
            if not cfg.get('keep_original_on_tts_failure'):raise
            atomic_json(folder/f'{index:04d}-generation-failure.json',{'error':str(error)})
            return keep_original(index,'tts_generation_failed')
    def fit(prepared):
        try:
            report=fit_speech(prepared,cfg)
            if report.get('keep_original'):segments[report['index']]['keep_original']=True
            return report
        except RuntimeError as error:
            if not cfg.get('keep_original_on_tts_failure'):raise
            index=prepared['index'];atomic_json(folder/f'{index:04d}-fit-failure.json',{'error':str(error)})
            reason='tts_output_rejected' if prepared.get('_guard_error') else 'tts_timing_failed'
            return keep_original(index,reason)['cached']
    if cfg.get('tts_output_guard') in ('scribe-v2','waveform'):
        from checked_tts_queue import run
        if cfg['tts_output_guard']=='waveform':
            from waveform_output_guard import check_batch
            checking_label='참조 파형·음소 비교'
        else:
            from tts_output_guard import check_batch
            checking_label='Scribe 생성음 확인'
        def checked_update(done,running,queued,fitting,waiting,checking):
            if status:status(done/max(1,len(items))*100,
                f'완료 {done}/{len(items)} · TTS {running}/{TTS_CONCURRENCY} · {checking_label} {checking} · 쉼·겹침 보정 {fitting} · 대기 {queued+waiting}')
            else:progress(done,running,queued,fitting,waiting)
        try:
            return run(items,generate,fit,lambda values:check_batch(values,cfg,folder/'output-checks'),checked_update,
                batch_size=output_guard_policy(cfg)['batch_size'])
        finally:
            if cfg['tts_output_guard']=='waveform':
                from phonetic_output_guard import release_runtime
                release_runtime()
    reports=pipelined_queue(items,generate,fit,update)
    from tts_collision import relax_collisions
    return relax_collisions(segments,reports,clean,folder,cfg)


def blend(original,clean,translated):
    # Fitted TTS already carries the configured clip edge fades.
    return np.clip(original-clean+translated,-1,1)


def replacement_intervals(segments,frames,placements=None):
    """Keep source dialogue removed through both reference and actual TTS ends."""
    if placements is not None and len(placements)!=len(segments):raise RuntimeError('Dubbed playback interval counts do not match.')
    ends=[]
    for i,segment in enumerate(segments):
        end=segment['hi']
        if placements is not None:
            p=placements[i]
            if p['lo']!=segment.get('output_start_frame',segment['lo']) or p['lo']<0 or p['hi']<=p['lo']:raise RuntimeError('Invalid dubbed playback interval.')
            end=max(end,p['hi'])
        ends.append(end)
    margin=round(REPLACEMENT_MARGIN_SECONDS*SR)
    intervals=[]
    previous=0
    for index,segment in enumerate(segments):
        lo,reference_hi=segment['lo'],segment['hi']
        if not previous<=lo<reference_hi<=frames:raise RuntimeError('Replacement intervals overlap or exceed source duration.')
        hi=ends[index]
        # Midpoints share free margins only. They must never cut into a TTS
        # coverage interval, including a legacy overrun into later segments.
        left=0 if index==0 else min(lo,(ends[index-1]+lo)//2)
        right=max(frames,hi) if index+1==len(segments) else max(hi,(hi+segments[index+1]['lo'])//2)
        if not segment.get('keep_original'):
            start=min(lo,placements[index]['lo']) if placements is not None else lo
            intervals.append({'lo':max(0,start-margin) if start<lo else max(left,lo-margin),'hi':min(right,hi+margin)})
        previous=reference_hi
    return intervals


def transition_weights(start,frames,segments,excluded=()):
    positions=np.arange(start,start+frames,dtype=np.float64)
    weights=np.zeros(frames,dtype=np.float32)
    fade=round(TRANSITION_SECONDS*SR)
    for segment in segments:
        lo,hi=segment['lo'],segment['hi']
        if hi+fade<=start:continue
        if lo-fade>=start+frames:break
        # Max, not sum: overlapping margins must never remove the voice twice.
        fade_start,fade_end=lo-fade,hi+fade
        ramp=np.minimum((positions-fade_start)/max(1,lo-fade_start),(fade_end-positions)/max(1,fade_end-hi))
        ramp[(positions>=lo)&(positions<hi)]=1
        np.maximum(weights,np.clip(ramp,0,1).astype(np.float32),out=weights)
    return weights


def prepare_segments(transcript,frames,source_transcription=None):
    result=[];previous=0
    windows=transcript.get('reference_windows')
    if windows is not None and len(windows)!=len(transcript['segments']):raise RuntimeError('TTS reference interval counts do not match.')
    for index,segment in enumerate(transcript['segments']):
        lo=round(segment['start']*SR);hi=min(frames,round(segment['end']*SR))
        if lo<previous or hi<=lo or segment['end']>frames/SR+.02:
            raise RuntimeError('Dialogue intervals overlap or exceed source duration.')
        prepared={**segment,'lo':lo,'hi':hi}
        if windows is not None:
            window=windows[index]
            prepared.update(reference_lo=round(window['start']*SR),reference_hi=round(window['end']*SR),
                reference_fade_in=window['fade_in_frames'],reference_fade_out=window['fade_out_frames'])
            if not lo<=prepared['reference_lo']<prepared['reference_hi']<=hi:raise RuntimeError('TTS reference interval exceeds the source replacement interval.')
            if windows[index].get('boundary_policy')=='shared_clean_interval' and (prepared['reference_lo']!=lo or prepared['reference_hi']!=hi):
                raise RuntimeError('Final replacement interval does not match the TTS reference interval.')
        result.append(prepared);previous=hi
    excluded=[*transcript.get('excluded_intervals',[]),*transcript.get('reference_tail_intervals',[])]
    asr=(source_transcription or {}).get('words',[])
    for index,segment in enumerate(result):
        lo=segment.get('reference_lo',segment['lo']);hi=segment.get('reference_hi',segment['hi'])
        left=result[index-1]['hi'] if index else 0
        right=result[index+1]['lo'] if index+1<len(result) else frames
        for interval in excluded:
            a=round(interval['start']*SR);b=round(interval['end']*SR)
            if b<=lo:left=max(left,b)
            elif a<lo<b:left=lo
            if a>=hi:right=min(right,a)
            elif a<hi<b:right=hi
        for interval in asr:
            a=round(interval['start']*SR);b=round(interval['end']*SR)
            if b<=lo:left=max(left,b)
            if a>=hi:right=min(right,a)
        segment.update(natural_reference_margin=transcript.get('boundary_authority') not in ('scribe_v2','diarizen_native','nemotron_eend_vc','mai_transcribe_2','fish_transcribe_1_pro','gemini_native_audio','qwen_omni_audio','fish_qwen_gemini','moss_transcribe_diarize'),reference_margin_left=left,reference_margin_right=right)
    return result


def speech_placements(segments,folder):
    result=[]
    for n,segment in enumerate(segments):
        audio=sf.info(folder/f'{n:04d}-fitted.wav')
        if audio.samplerate!=SR or audio.channels!=2 or audio.frames<1:
            raise RuntimeError('Invalid TTS format.')
        start=segment.get('output_start_frame',segment['lo'])
        result.append({'lo':start,'hi':start+audio.frames})
    return result


@local_stage
def mix(original,clean,segments,folder,destination,progress,excluded_intervals=(),*,background=None,replacement_override=None):
    temporary=destination.with_name('mix.partial.wav')
    with ExitStack() as stack:
        src=stack.enter_context(sf.SoundFile(original))
        voice=stack.enter_context(sf.SoundFile(clean))
        bg=stack.enter_context(sf.SoundFile(background)) if background is not None else None
        for track in [voice,*([bg] if bg is not None else [])]:
            if (len(track),track.samplerate,track.channels)!=(len(src),SR,2):
                raise RuntimeError('Source and separated audio durations, channels, or sample rates do not match.')
        if src.samplerate!=SR or src.channels!=2:raise RuntimeError('Invalid source audio format.')
        out=stack.enter_context(sf.SoundFile(temporary,'w',samplerate=SR,channels=2,subtype='FLOAT',format='RF64'))
        placements=speech_placements(segments,folder)
        replacements=replacement_intervals(segments,len(src),placements) if replacement_override is None else replacement_override
        output_frames=max(len(src),max((p['hi'] for p in placements),default=0))
        for n,segment in enumerate(segments):
            generated,rate=sf.read(folder/f'{n:04d}-fitted.wav',dtype='float32',always_2d=True)
            if rate!=SR or generated.shape[1]!=2 or not np.isfinite(generated).all():raise RuntimeError('Invalid TTS format.')
        voice.seek(0)
        start=0
        while start<output_frames:
            count=min(10*SR,output_frames-start)
            def read_padded(track):
                part=track.read(count,dtype='float32',always_2d=True)
                return np.pad(part,((0,count-len(part)),(0,0)))
            values=read_padded(src)
            vocals=read_padded(voice)
            bed=read_padded(bg) if bg is not None else values-vocals
            if not np.isfinite(bed).all():raise RuntimeError('Invalid background audio values.')
            weights=transition_weights(start,len(values),replacements)
            preserved=np.zeros(len(values),dtype=bool)
            for segment in segments:
                if segment.get('keep_original'):
                    a,b=max(start,segment['lo']),min(start+len(values),segment['hi'])
                    if b>a:preserved[a-start:b-start]=True
            weights[preserved]=0
            translated=np.zeros_like(values)
            for n,segment in enumerate(placements):
                if segments[n].get('keep_original'):continue
                lo,hi=max(start,segment['lo']),min(start+len(values),segment['hi'])
                if hi<=lo:continue
                with sf.SoundFile(folder/f'{n:04d}-fitted.wav') as tts:
                    tts.seek(lo-segment['lo'])
                    translated[lo-start:hi-start]+=tts.read(hi-lo,dtype='float32',always_2d=True)*TTS_GAIN
            translated[preserved]=0
            # Transition only source speech <-> background. TTS uses one fixed gain.
            changed=(weights>0)|np.any(translated!=0,axis=1)
            mixed=values.copy()
            if bg is None:
                # Keep the legacy path bit-identical for before/after comparisons.
                mixed[changed]=np.clip(values[changed]-vocals[changed]*weights[changed,None]+translated[changed],-1,1)
            else:
                mixed[changed]=np.clip(values[changed]*(1-weights[changed,None])+bed[changed]*weights[changed,None]+translated[changed],-1,1)
            out.write(mixed)
            start+=count
            progress(start/output_frames*100,'')
        if out.frames!=output_frames:raise RuntimeError('Mixed audio duration validation failed.')
    temporary.replace(destination)


def validate_mux(info,source_info,duration):
    """Validate media streams separately: copied subtitles can extend the container."""
    def times(stream):
        start=float(stream.get('start_time') or 0)
        value=stream.get('duration')
        if value in (None,'N/A') and stream.get('duration_ts') is not None:
            from fractions import Fraction
            value=float(stream['duration_ts'])*float(Fraction(stream['time_base']))
        length=float(value) if value not in (None,'N/A') else float('nan')
        if not math.isfinite(start) or not math.isfinite(length) or length<=0:
            raise RuntimeError('Final MP4 validation failed: cannot determine video or audio track duration.')
        return start,start+length
    streams=info.get('streams',[])
    video=next((s for s in streams if s.get('codec_type')=='video'),None)
    audio=next((s for s in streams if s.get('codec_type')=='audio'),None)
    source_video=next((s for s in source_info.get('streams',[]) if s.get('codec_type')=='video'),None)
    if video is None or audio is None or source_video is None:
        raise RuntimeError('Final MP4 validation failed: video or audio track is missing.')
    video_start,video_end=times(video);audio_start,audio_end=times(audio)
    source_start,source_end=times(source_video)
    origin=float(source_info.get('format',{}).get('start_time') or 0)
    expected_video_start=max(0.,source_start-origin)
    expected_video_end=min(duration,source_end-origin)
    report={'expected_audio_end':duration,'actual_audio_start':audio_start,'actual_audio_end':audio_end,
        'expected_video_start':expected_video_start,'expected_video_end':expected_video_end,
        'actual_video_start':video_start,'actual_video_end':video_end,
        'container_duration':info.get('format',{}).get('duration'),'tolerance_seconds':.15,
        'policy':'validate_audio_and_video_streams_independently_of_subtitle_duration'}
    errors=[]
    if abs(audio_start)>.15 or abs(audio_end-duration)>.15:errors.append(f'오디오 {audio_start:.3f}–{audio_end:.3f}초, 목표 끝 {duration:.3f}초')
    if abs(video_start-expected_video_start)>.15 or abs(video_end-expected_video_end)>.15:
        errors.append(f'영상 {video_start:.3f}–{video_end:.3f}초, 예상 {expected_video_start:.3f}–{expected_video_end:.3f}초')
    return {**report,'passed':not errors,'errors':errors}


@local_stage
def mux(source,mixed,destination,language,duration):
    source,mixed,destination=source.resolve(),mixed.resolve(),destination.resolve()
    temporary=destination.with_name('final.partial.mp4')
    # Preserve a natural TTS tail past the video boundary rather than truncating it.
    duration=max(duration,sf.info(mixed).duration)
    iso=LANGUAGE_CATALOG['languages'][language]['iso639_3']
    source_info=probe(source);attempts=[];started=time.perf_counter()
    for encoder in MUX_AUDIO_ENCODERS:
        attempt_start=time.perf_counter();report=None
        try:
            command(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-i',str(mixed),'-map','0:v:0','-map','1:a:0','-map','0:s?',
                '-map_metadata','0','-c:v','copy','-c:s','copy','-c:a',encoder,'-b:a','320k','-ar',str(SR),'-ac','2',
                '-af','apad','-t',str(duration),'-metadata:s:a:0',f'language={iso}','-movflags','+faststart',str(temporary)])
            info=probe(temporary)
            report=validate_mux(info,source_info,duration)
            if not report['passed']:raise RuntimeError('최종 MP4 검증 실패: '+'; '.join(report['errors']))
        except RuntimeError as error:
            attempts.append({'encoder':encoder,'passed':False,'seconds':time.perf_counter()-attempt_start,
                'error':str(error),'validation':report})
            atomic_json(destination.with_name('mux-validation.json'),{'passed':False,'attempts':attempts})
            continue
        attempts.append({'encoder':encoder,'passed':True,'seconds':time.perf_counter()-attempt_start})
        atomic_json(destination.with_name('mux-validation.json'),{**report,'audio_encoder':encoder,
            'audio_bitrate':'320k','video_codec':'copy','attempts':attempts,
            'fallback_used':len(attempts)>1,'save_seconds':time.perf_counter()-started})
        from media_output import publish
        return publish(temporary,destination)
    raise RuntimeError('MP4 저장 실패: '+attempts[-1]['error'])
