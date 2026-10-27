"""Use actual next-clip voice onset as the deadline, not its leading silence.

This never relocates source words or fabricates a boundary. A following clip's
verified silence may overlap another clip without overlapping their voices.
"""
from pathlib import Path
import json
import soundfile as sf
from core import SR
from asr_word_edges import speech_hints

VERSION=1

def relax_collisions(segments,reports,clean,folder,cfg,wanted=None):
    from render import fit_speech,reference_audio
    folder=Path(folder);by_id={r['index']:r for r in reports}
    wanted=set(by_id) if wanted is None else set(wanted)
    order=sorted((i for i,s in enumerate(segments) if not s.get('deleted') and not s.get('keep_original')),key=lambda i:segments[i]['lo'])
    for slot in range(len(order)-2,-1,-1):
        i,j=order[slot:slot+2];segment=segments[i];next_segment=segments[j]
        if i not in wanted or segment.get('audio_edit') or next_segment.get('audio_edit'):continue
        metadata=folder/f'{i:04d}.json';raw=folder/f'{i:04d}-tts.mp3';following=folder/f'{j:04d}-fitted.wav'
        generation_file=folder/f'{i:04d}-generation.json'
        if not all(p.exists() for p in [metadata,raw,following,generation_file]):continue
        prior=json.loads(metadata.read_text('utf8'))
        if prior.get('tempo',1)<=1.001:continue
        values,rate=sf.read(following,dtype='float32',always_2d=True)
        try:voiced=speech_hints(values,rate)
        except Exception:continue  # No reliable silence evidence: retain timing.
        if not voiced:continue
        quiet_frames=max(0,round(voiced[0][0]*SR))
        if not quiet_frames:continue
        old_budget=next_segment['lo']-segment['lo']
        budget=old_budget+quiet_frames
        if budget<=prior.get('collision_budget_frames',old_budget):continue
        reference=folder/f'{i:04d}-timing-reference.wav'
        reference_audio(clean,{'lo':segment['lo'],'hi':segment['hi']},reference)
        generation=json.loads(generation_file.read_text('utf8'))
        prepared={'index':i,'segment':segment,'folder':folder,'reference':reference,'timing_reference':reference,
            'generated':raw,'fitted':folder/f'{i:04d}-fitted.wav','metadata':metadata,'generation_cache':generation_file,
            'generation':generation,'policy':generation.get('tts_policy'),'frames':segment['hi']-segment['lo'],
            'window':[0,segment['hi']-segment['lo']],'collision_budget':budget}
        report=fit_speech(prepared,cfg)
        report.update(collision_deadline_method='next_tts_audible_onset',next_clip_silence_frames=quiet_frames)
        from core import atomic_json
        atomic_json(metadata,report);by_id[i]=report
    return [by_id[r['index']] for r in reports]
