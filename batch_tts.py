"""Joint Drama 3 generation: one audio reference, paired tagged transcripts.

Batch output is aligned once, split without changing samples, then passed to
the existing per-dialogue timing/mixing pipeline. No individual-TTS fallback.
"""
from __future__ import annotations
import hashlib,json,shutil,re
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import soundfile as sf
from core import SR,TTS_CONCURRENCY,FIT_CONCURRENCY,atomic_json,command

VERSION=1
REFERENCE_TARGET_SECONDS=60.
BATCH_REFERENCE_POLICY={'version':16,'target_seconds':REFERENCE_TARGET_SECONDS,
    'speaker_tags':'none',
    'instructions':'brief_native_target_pronunciation_intonation_reference_voice_emotion_dynamics','reference_gap_seconds':0.,'condition_on_previous_chunks':True,'delivery_cue':None,
    'remainder':'fill_from_nearest_previous_same_speaker_whole_lines_no_repetition'}
CONTEXT_REFERENCE_POLICY={'version':12,'minimum_seconds':2.,'target_seconds':None,'max_parts':16,
    'selection':'current_alone_if_sufficient_else_quality_ranked_same_speaker_single_donor_first',
    'order':'target_first_then_selected_donors_in_video_order','speaker_tags':'none',
    'short_speaker':'all_available_without_repetition','boundaries':'whole_source_utterances'}


def batch_instructions(language):
    import render
    return render.tts_instructions(language)+' Each speaker tag represents a different speaker; use its matching reference voice consistently.'


def reference_seconds(segment):
    if 'lo' in segment and 'hi' in segment:
        return max(0.,(segment.get('reference_hi',segment['hi'])-segment.get('reference_lo',segment['lo']))/SR)
    return max(0.,segment.get('end',0.)-segment.get('start',0.))


def plan_batches(segments):
    batches=[];current=[];seconds=0.
    for i,segment in enumerate(segments):
        if current and seconds>=BATCH_REFERENCE_POLICY['target_seconds']:
            batches.append(current);current=[];seconds=0.
        current.append(i)
        seconds+=reference_seconds(segment)
    if current:batches.append(current)
    return batches


def plan_speaker_batches(segments):
    """Reach sixty reference seconds per voice without splitting utterances or limiting count."""
    groups=[];opened={};durations={}
    for i,segment in enumerate(segments):
        voice=segment.get('speaker_id')
        known=voice is not None and str(voice) not in ('','unknown')
        key=str(voice) if known else None
        group=opened.get(key) if known else None
        if group is None or durations[key]>=BATCH_REFERENCE_POLICY['target_seconds']:
            group=[];groups.append(group)
            if known:opened[key]=group;durations[key]=0.
        group.append(i)
        if known:durations[key]+=reference_seconds(segment)
    return groups


def speaker_request_groups(segments,groups):
    """Pad short requests to sixty seconds with earlier same-voice lines, without repetition."""
    requests=[]
    for group in groups:
        members=list(group)
        seconds=sum(reference_seconds(segments[i]) for i in members)
        voice=segments[group[0]].get('speaker_id')
        if voice is not None and str(voice) not in ('','unknown'):
            donors=[i for i in range(min(group)) if str(segments[i].get('speaker_id'))==str(voice)
                and not segments[i].get('deleted') and not segments[i].get('keep_original')
                and any(c.isalnum() for c in segments[i].get('translated_text',''))]
            for i in reversed(donors):
                if seconds>=BATCH_REFERENCE_POLICY['target_seconds']:break
                members.append(i)
                seconds+=reference_seconds(segments[i])
        requests.append(sorted(members))
    return requests


def synthesize_speaker_batches(segments,clean,folder,cfg,progress,status=None):
    import render
    groups=cfg.get('_speaker_batch_groups')
    if groups is None:groups=plan_speaker_batches(segments)
    requests=cfg.get('_speaker_batch_requests')
    if requests is None:requests=speaker_request_groups(segments,groups)
    return synthesize_batches(segments,clean,folder,{**cfg,'tts_omit_instructions':cfg.get('tts_omit_instructions',False)},progress,status,
        groups=groups,request_groups=requests,tag_mode='none',mode='same_speaker_batches',
        reference_mode='concatenated',reference_gap_seconds=BATCH_REFERENCE_POLICY['reference_gap_seconds'],
        instructions=render.tts_instructions(cfg['language']))


def prepare_regeneration_reference(segments,index,clean,folder):
    """Keep the current line alone, supplementing only to reach two content seconds."""
    from render import reference_audio
    from rich_transcript import without_speakers
    voice=segments[index].get('speaker_id')
    known=voice is not None and str(voice) not in ('','unknown')
    eligible=[i for i,s in enumerate(segments) if i!=index and known and s.get('speaker_id')==voice
        and not s.get('deleted') and s.get('original_text','').strip()]
    # A reliable identity here means the saved same-speaker assignment and
    # a non-overlapping, usable recording. No embedding reassignment is enabled.
    eligible=[i for i in eligible if not any(j!=i and str(other.get('speaker_id'))!=str(voice)
        and not other.get('deleted') and min(segments[i]['hi'],other['hi'])>max(segments[i]['lo'],other['lo'])
        for j,other in enumerate(segments))]
    bank=Path(folder)/'regeneration-references';bank.mkdir(parents=True,exist_ok=True)
    members={};total_frames=0;minimum=round(CONTEXT_REFERENCE_POLICY['minimum_seconds']*SR)
    def read_member(i):
        path=bank/f'{i:04d}-reference.wav';reference_audio(clean,segments[i],path)
        values,rate=sf.read(path,dtype='int16',always_2d=True)
        if rate!=SR or values.shape[1]!=1:raise RuntimeError('Invalid grouped reference audio format.')
        if not len(values) or not np.isfinite(values).all():raise RuntimeError('No audio exists in the reference interval.')
        normalized=values.astype(np.float32)/32768.
        hop=max(1,round(.02*SR));frames=normalized[:len(values)//hop*hop]
        levels=np.sqrt(np.mean(frames.reshape(-1,hop)**2,axis=1)) if len(frames) else np.array([np.sqrt(np.mean(normalized**2))])
        peak=float(np.max(np.abs(normalized)))
        if peak<1e-6:raise RuntimeError('No audio exists in the reference interval.')
        active=float(np.mean(levels>max(.0001,float(np.percentile(levels,90))*.05)))
        clipping=float(np.mean(np.abs(normalized)>=.99))
        quality=active-10*clipping
        return values,{'index':i,'original_text':segments[i]['original_text'],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'activity_fraction':active,
            'clipping_fraction':clipping,'recording_quality_score':quality}
    members[index]=read_member(index);total_frames=len(members[index][0])
    rejected=[]
    if total_frames<minimum:
        candidates=[]
        for i in eligible:
            try:
                member=read_member(i)
                if member[1]['activity_fraction']<.1:raise RuntimeError('Reference speech coverage is too low.')
                candidates.append((i,member))
            except RuntimeError as error:rejected.append({'index':i,'reason':str(error)})
        while candidates and total_frames<minimum and len(members)<CONTEXT_REFERENCE_POLICY['max_parts']:
            # Prefer one complete usable donor over collecting many short clips.
            candidates.sort(key=lambda item:(
                len(item[1][0])<minimum-total_frames,
                not bool(segments[item[0]].get('_speaker_locked')),
                -item[1][1]['recording_quality_score'],
                abs(segments[item[0]]['lo']-segments[index]['lo']),item[0]))
            i,member=candidates.pop(0);members[i]=member;total_frames+=len(member[0])
    indices=[index,*sorted(i for i in members if i!=index)]
    arrays=[]
    for i in indices:
        arrays.append(members[i][0])
    parts=[members[i][1] for i in indices]
    identity={**CONTEXT_REFERENCE_POLICY,'parts':parts,'gap_seconds':0.,'instructions':None}
    key=hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    path=bank/f'{key}.wav';sf.write(path,np.concatenate(arrays),SR,subtype='PCM_16')
    text='\n'.join(without_speakers(segments[i]['original_text']) for i in indices)
    atomic_json(path.with_suffix('.json'),{**identity,'target_index':index,'reference_text':text,'reference_seconds':total_frames/SR,
        'content_seconds':total_frames/SR,'minimum_met':total_frames>=minimum,'rejected_donors':rejected})
    segments[index].update(_shared_reference_path=str(path),_shared_reference_text=text,
        _shared_reference_indices=indices,_shared_reference_speaker=segments[index].get('speaker_id'),_shared_reference_donor_tag=False,_preserve_speaker_tags=False,_shared_reference_output_tag=None,_omit_instructions=True)
    return indices


def prepare_context_references(segments,clean,folder,only_indices=None):
    selected=set(range(len(segments)) if only_indices is None else only_indices)
    selected={i for i in selected if not segments[i].get('deleted') and not segments[i].get('keep_original')}
    for index in sorted(selected):
        prepare_regeneration_reference(segments,index,clean,folder)


def tagged_text(segments,indices,tag_mode='speaker'):
    if tag_mode=='reference_slot_omit_first':
        from rich_transcript import without_speakers
        if len(indices)>16:raise RuntimeError('Batch TTS exceeds the limit of 16 dialogue lines.')
        tags=[None]+list(range(1,len(indices))) if indices else []
        def text(field):
            return '\n'.join(('' if tag is None else f'<|speaker:{tag}|>')+without_speakers(segments[i][field])
                for i,tag in zip(indices,tags))
        return text('original_text'),text('translated_text'),tags
    if tag_mode=='none':
        from rich_transcript import without_speakers
        return tuple('\n'.join(without_speakers(segments[i][field]) for i in indices) for field in ('original_text','translated_text'))+([None]*len(indices),)
    mapping={};tags=[]
    for i in indices:
        # Old four-field caches have no claimed identities: never merge them.
        identity=f'reference_slot_{i}' if tag_mode=='reference_slot' else segments[i].get('speaker_id') or f'unknown_turn_{i}'
        if identity not in mapping:mapping[identity]=len(mapping)
        tags.append(mapping[identity])
    if len(mapping)>16:raise RuntimeError('Batch TTS exceeds the limit of 16 speakers.')
    def text(field):return '\n'.join(f'<|speaker:{tag}|>{segments[i][field]}' for i,tag in zip(indices,tags))
    return text('original_text'),text('translated_text'),tags


ALIGNMENT_VERSION=14


def spoken_text(text):
    """Performance cues condition TTS but are not words it must pronounce."""
    return re.sub(r'\[[^\[\]]*\]|<\|speaker:[^|]*\|>', '', text).strip()

def recover_edge_anchors(locations,words,texts,values,rate):
    """Recover a single edge line using unclaimed ASR and phonetic spelling.

    Ignore empty script slots when finding neighbors. Require both recall and
    precision in decomposed Unicode; a different repeated sentence cannot pass.
    """
    from copy import deepcopy
    import unicodedata,re
    from source_alignment import compact_match,match_key
    result=deepcopy(locations);audit=[]
    active=[i for i,t in enumerate(texts) if match_key(t)]
    claimed={i for r in result if r.get('matched') for i in r['word_indices']}
    for edge in (0,-1):
        if len(active)<2:break
        index=active[edge];neighbor=active[1 if edge==0 else -2]
        if result[index].get('matched') or not result[neighbor].get('matched'):continue
        anchor=result[neighbor]
        if not anchor['word_indices']:continue
        candidates=[i for i,w in enumerate(words) if i not in claimed and w['end']>w['start'] and match_key(w['word'])
            and (i<anchor['word_indices'][0] and w['end']<=anchor['start']+.002 if edge==0
                 else i>anchor['word_indices'][-1] and w['start']>=anchor['end']-.002)]
        if not candidates:continue
        original_target=match_key(texts[index])
        target=unicodedata.normalize('NFD',original_target)
        heard=unicodedata.normalize('NFD',''.join(match_key(words[i]['word']) for i in candidates))
        hits=compact_match(target,heard)
        # ASR often writes a stammer once. Only collapse adjacent repetitions
        # for this already bounded single-line comparison; keep its audio whole.
        if len(hits)/max(1,len(target))<.7:
            collapsed=unicodedata.normalize('NFD',re.sub(r'(.)\1+',r'\1',original_target))
            if collapsed!=target:
                alternate=compact_match(collapsed,heard)
                if len(alternate)/max(1,len(collapsed))>=.7 and len(alternate)/max(1,len(heard))>=.6:
                    target,hits=collapsed,alternate
        if len(hits)/max(1,len(target))<.7 or len(hits)/max(1,len(heard))<.6:continue
        start,end=words[candidates[0]]['start'],words[candidates[-1]]['end']
        block=values[max(0,round(start*rate)):min(len(values),round(end*rate))]
        if not len(block) or float(np.max(np.abs(block)))<1e-5:continue
        result[index]={'matched':True,'coverage':len(hits)/len(target),'start':start,'end':end,
            'words':[words[i] for i in candidates],'word_indices':candidates,
            'recovery':'phonetic_unclaimed_asr_at_batch_edge','lexical_match_verified':True}
        claimed.update(candidates)
        audit.append({'local_index':index,'start':start,'end':end,'method':result[index]['recovery']})
    return result,audit

def recover_between_anchors(locations,words,values,rate):
    """Recover one omitted line bounded by already validated neighboring lines.

    Prefer unclaimed ASR words. If ASR omitted the line entirely, require a
    measurable speech island inside those anchors. Never divide a gap among
    multiple unknown lines, or copy the source video's timing into TTS audio.
    """
    from copy import deepcopy
    from source_alignment import match_key
    result=deepcopy(locations);audit=[]
    claimed={i for row in result if row.get('matched') for i in row['word_indices']}
    for index,row in enumerate(result):
        if row.get('matched'):continue
        # Both immediate neighbors must be trustworthy; a chain of missing
        # lines has no unique owner and is intentionally not guessed.
        if index==0 or index+1==len(result):continue
        previous,following=result[index-1],result[index+1]
        if not previous.get('matched') or not following.get('matched'):continue
        if previous.get('recovery') or following.get('recovery'):continue
        left,right=previous['end'],following['start']
        left_word,right_word=previous['word_indices'][-1],following['word_indices'][0]
        candidates=[i for i in range(left_word+1,right_word)
                    if i not in claimed and match_key(words[i]['word'])
                    and left<=words[i]['start']<words[i]['end']<=right]
        if candidates:
            start,end=words[candidates[0]]['start'],words[candidates[-1]]['end']
            # Confirm that the remaining ASR span actually contains audio.
            block=values[max(0,round(start*rate)):min(len(values),round(end*rate))]
            if not len(block) or float(np.max(np.abs(block)))<1e-5:continue
            result[index]={'matched':True,'coverage':None,'start':start,'end':end,
                'words':[words[i] for i in candidates],'word_indices':candidates,
                'recovery':'unclaimed_asr_between_validated_neighbors',
                'recognized_text':' '.join(words[i]['word'] for i in candidates),
                'lexical_match_verified':False}
            claimed.update(candidates)
        else:
            # Small inner guard excludes neighboring word releases/reverb.
            lo=round((left+.08)*rate);hi=round((right-.08)*rate)
            if hi-lo<round(.12*rate):continue
            hop=max(1,round(.01*rate));block=values[max(0,lo):min(len(values),hi)]
            count=len(block)//hop
            if count<12:continue
            energy=np.sqrt(np.mean(block[:count*hop].reshape(count,hop,-1)**2,axis=1)).max(axis=1)
            global_rms=float(np.sqrt(np.mean(values**2)))
            threshold=max(1e-5,global_rms*.12,float(energy.max())*.06)
            mask=energy>threshold
            active=np.flatnonzero(mask)
            if len(active)<12:continue
            # Require quiet separation from both already measured neighbors.
            if active[0]<3 or count-1-active[-1]<3:continue
            start=(lo+max(0,int(active[0])-2)*hop)/rate
            end=(lo+min(count,int(active[-1])+3)*hop)/rate
            result[index]={'matched':True,'coverage':None,'start':start,'end':end,
                'words':[],'word_indices':[],
                'recovery':'measured_audio_between_validated_neighbors',
                'lexical_match_verified':False}
        audit.append({'local_index':index,'start':result[index]['start'],'end':result[index]['end'],
            'method':result[index]['recovery'],'left_anchor':index-1,'right_anchor':index+1})
    return result,audit


def exclude_unrequested_replay(cuts,locations,words,segments,indices,rate,values=None):
    """Remove ASR-confirmed source/repeated dialogue outside aligned target lines.

    Silence, breaths and short unmatched openings stay intact. Only a substantial
    lexical replay of known reference or target text qualifies for exclusion.
    No language, dialogue number or source time is special-cased.
    """
    from source_alignment import match_key,compact_match
    assigned={i for row in locations for i in row['word_indices']}
    unused=[w for i,w in enumerate(words) if i not in assigned and w['end']>w['start']
            and match_key(w['word']) and not any(w['start']<r['end'] and w['end']>r['start'] for r in locations)]
    result=[];audit=[]
    for local,((lo,hi),location) in enumerate(zip(cuts,locations)):
        segment=segments[indices[local]]
        for side in ('before','after'):
            extra=[w for w in unused if (lo/rate<=w['start']<w['end']<=location['start'] if side=='before'
                                        else location['end']<=w['start']<w['end']<=hi/rate)]
            if not extra:continue
            recognized=''.join(match_key(w['word']) for w in extra)
            scores={}
            for field in ('original_text','translated_text'):
                target=match_key(spoken_text(segment.get(field,'')))
                if len(target)<4:continue
                hits=compact_match(target,recognized)
                coverage=len(hits)/len(target)
                precision=len(hits)/max(1,len(recognized))
                if coverage>=.7 and precision>=.5:scores[field]={'coverage':coverage,'precision':precision}
            if not scores:
                # A multilingual recognizer may corrupt/translate part of a
                # source replay. Several independent, long source fragments
                # can still prove it; do not require the entire source sentence
                # to survive ASR verbatim. Short breaths/reactions cannot pass.
                source=match_key(spoken_text(segment.get('original_text','')))
                fragments=[];covered=set()
                for word in extra:
                    token=match_key(word['word'])
                    if len(token)<4:continue
                    hits=compact_match(token,source)
                    positions={j for _,j in hits}
                    if len(hits)/len(token)>=.8 and not positions.intersection(covered):
                        covered.update(positions);fragments.append(token)
                if len(fragments)>=2 and len(covered)>=max(8,len(source)*.25):
                    scores['independent_source_fragments']={'count':len(fragments),'source_coverage':len(covered)/max(1,len(source))}
            if not scores:
                extra_speech=sum(w['end']-w['start'] for w in extra)
                target_speech=sum(w['end']-w['start'] for w in location.get('words',[]))
                if len(extra)>=2 and len(recognized)>=max(4,len(match_key(spoken_text(segment.get('translated_text',''))))*.25) and extra_speech>=max(.4,target_speech*.25):
                    audit.append({'local_index':local,'side':side,'keep_original':True,
                        'method':'substantial_unassigned_speech_not_verified',
                        'recognized_text':' '.join(w['word'] for w in extra)})
                continue
            old=[lo,hi]
            if side=='before':
                boundary=round((extra[-1]['end']+location['start'])*rate/2) if values is None else acoustic_cut(values,rate,extra[-1]['end'],location['start'])[0]
                if values is not None and extra[-1]['end']>=location['start']-.002:
                    boundary=acoustic_cut(values,rate,max(0,location['start']-.12),location['start']+.03)[0]
                    boundary=min(boundary,round(location['start']*rate))
                lo=max(lo,boundary)
            else:
                boundary=round((location['end']+extra[0]['start'])*rate/2) if values is None else acoustic_cut(values,rate,location['end'],extra[0]['start'])[0]
                hi=min(hi,boundary)
            if lo>=hi:raise RuntimeError('Unrequested speech exclusion produced an empty target')
            audit.append({'local_index':local,'side':side,'old_cut_frames':old,'new_cut_frames':[lo,hi],
                'excluded_frames':old[1]-old[0]-(hi-lo),'method':'asr_verified_source_or_target_replay',
                'evidence':scores,'recognized_text':' '.join(w['word'] for w in extra)})
        result.append((lo,hi))
    return result,audit


def acoustic_cut(values,rate,left,right):
    """Use ASR only as a search hint; verify quietness against nearby speech."""
    lo=max(0,round(left*rate));hi=min(len(values),round(right*rate))
    if hi<=lo:return lo,{'method':'touching_asr_anchors','frame':lo}
    hop=max(1,round(.005*rate))
    context_lo=max(0,lo-round(.35*rate));context_hi=min(len(values),hi+round(.35*rate))
    context=values[context_lo:context_hi];count=len(context)//hop
    if not count:return round((lo+hi)/2),{'method':'short_asr_gap'}
    energy=np.sqrt(np.mean(context[:count*hop].reshape(count,hop,-1)**2,axis=(1,2)))
    speech_level=float(np.percentile(energy,90))
    # A loud ASR gap is not silence merely because all of its frames are loud.
    threshold=max(1e-6,speech_level*.025)
    nominal=(lo+hi)/2;selection=None
    for radius in (0.,.1,.2,.35):
        a=max(context_lo,lo-round(radius*rate));b=min(context_hi,hi+round(radius*rate))
        begin=max(0,int(np.ceil((a-context_lo)/hop)));finish=min(count,(b-context_lo)//hop)
        if finish<=begin:continue
        quiet=energy[begin:finish]<=threshold
        edges=np.diff(np.r_[False,quiet,False].astype(np.int8))
        spans=[(int(x+begin),int(y+begin)) for x,y in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))
               if y-x>=max(2,round(.015*rate/hop))]
        if spans:
            x,y=max(spans,key=lambda span:(span[1]-span[0],-abs(context_lo+(span[0]+span[1])*hop/2-nominal)))
            selection=(context_lo+round((x+y)*hop/2),radius);break
    if selection is None:
        # No verified quietness: retain the lowest-energy original gap rather
        # than expanding farther into another utterance.
        a=max(0,(lo-context_lo)//hop);b=min(count,max(a+1,(hi-context_lo)//hop))
        center=context_lo+(a+int(np.argmin(energy[a:b])))*hop
        method='lowest_energy_unverified_asr_gap';radius=0.
    else:
        center,radius=selection
        method='quiet_valley_between_asr_anchors' if radius==0 else 'waveform_quiet_valley_outside_asr_gap'
    bound_lo,bound_hi=(lo,hi) if selection is None else (context_lo,context_hi)
    center=max(bound_lo,min(bound_hi-1,center))
    a=max(bound_lo,center-round(.0025*rate));b=min(bound_hi,center+round(.0025*rate)+1)
    cut=int(a+np.argmin(np.max(np.abs(values[a:b]),axis=1)))
    return cut,{'method':method,'frame':cut,'anchor_start':left,'anchor_end':right,
                'search_expansion_seconds':radius,'quiet_threshold':threshold}


def recover_single_after_reference(locations,words,values,rate,segment):
    """Handle source replay even if ASR writes target speech in the wrong script."""
    from source_alignment import locate_transcript,match_key
    if locations[0].get('matched'):return locations,[]
    original=spoken_text(segment.get('original_text',''));target=spoken_text(segment['translated_text'])
    if len(match_key(original))<4 or match_key(original)==match_key(target):return locations,[]
    source=locate_transcript([{'original_text':original}],words,len(values)/rate)[0]
    if not source.get('matched') or source.get('coverage',0)<.8:return locations,[]
    first,last=source['word_indices'][0],source['word_indices'][-1]
    if any(match_key(w['word']) for w in words[:first]):return locations,[]
    remaining=[i for i in range(last+1,len(words)) if words[i]['end']>words[i]['start'] and match_key(words[i]['word'])]
    if not remaining:return locations,[]
    start,end=words[remaining[0]]['start'],words[remaining[-1]]['end']
    if start<source['end'] or end-start<.1:return locations,[]
    return [{'matched':True,'coverage':None,'start':start,'end':end,'word_indices':remaining,
        'words':[words[i] for i in remaining],'recovery':'remaining_target_after_verified_reference_replay',
        'lexical_match_verified':False}], [{'local_index':0,'start':start,'end':end,
        'method':'remaining_target_after_verified_reference_replay','source_replay_end':source['end']}]


def verified_dialogue_boundary(values,rate,left,right):
    """A split needs measured quietness outside both aligned spoken spans."""
    cut,review=acoustic_cut(values,rate,left,right)
    lo=max(0,int(np.ceil(left*rate)));hi=min(len(values),int(np.floor(right*rate)))
    hop=max(1,round(.005*rate));count=max(0,(hi-lo)//hop)
    if count:
        block=values[lo:lo+count*hop]
        levels=np.sqrt(np.mean(block.reshape(count,hop,-1)**2,axis=(1,2)))
        # Breath can be far quieter than words. Do not inherit the much louder
        # speech gate: look for a low-floor valley in the aligned gap itself.
        threshold=min(review.get('quiet_threshold',1e-6),max(1e-7,float(np.percentile(levels,10))*2))
        quiet=levels<=threshold
        edges=np.diff(np.r_[False,quiet,False].astype(int))
        spans=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=3]
        if spans:
            a,b=max(spans,key=lambda p:(p[1]-p[0],-abs((p[0]+p[1])/2-count/2)))
            center=lo+(a+b)*hop//2
            return center,{**review,'method':'verified_quiet_inside_aligned_word_gap','frame':center,'verified':True,'boundary_quiet_threshold':threshold}
        # A stationary synthesis floor may sit above the speech-relative gate.
        # Accept only a sustained, low, stable valley inside both word anchors.
        floor=float(np.percentile(levels,20));speech=review.get('quiet_threshold',1e-6)/.025
        noise_gate=min(speech*.08,floor*1.8)
        edges=np.diff(np.r_[False,levels<=noise_gate,False].astype(int))
        stable=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))
                if (b-a)*hop/rate>=.12 and float(np.std(levels[a:b]))<=float(np.mean(levels[a:b]))*.35]
        if stable:
            a,b=max(stable,key=lambda p:p[1]-p[0]);center=lo+(a+b)*hop//2
            margin=max(1,round(.0025*rate));start=max(lo,center-margin);end=min(hi,center+margin)
            center=start+int(np.argmin(np.max(np.abs(values[start:end]),axis=1)))
            return center,{**review,'method':'stable_low_synthesis_floor_inside_word_gap','frame':center,'verified':True,'boundary_quiet_threshold':noise_gate}
    return None,{**review,'method':'no_verified_quiet_between_aligned_lines','verified':False}


def recover_boundary_with_local_alignment(values,rate,previous,current,left_text,right_text,folder,cfg):
    """Repair coarse ASR timestamps only when local phones AND quietness agree."""
    if rate!=SR:return None,{'method':'local_boundary_alignment','verified':False,'reason':'unsupported_rate'}
    from mfa_hint import alignment_hint,normalized
    left_text=spoken_text(left_text);right_text=spoken_text(right_text)
    if not normalized(left_text) or not normalized(right_text):
        return None,{'method':'local_boundary_alignment','verified':False,'reason':'empty_text'}
    start=max(0.,previous['start']-.08);end=min(len(values)/rate,current['end']+.08)
    lo=round(start*rate);hi=round(end*rate)
    hint=alignment_hint(values[lo:hi],left_text+' '+right_text,cfg['language'],Path(folder))
    review={'method':'local_boundary_alignment','verified':False,'alignment_status':hint.get('status')}
    if hint.get('status')!='aligned':return None,review
    words=hint.get('words',[]);prefix='';split=None
    for i,word in enumerate(words):
        prefix+=normalized(word['label'])
        if prefix==normalized(left_text):split=i+1;break
        if not normalized(left_text).startswith(prefix):break
    if split is None or split>=len(words) or normalized(' '.join(w['label'] for w in words[split:]))!=normalized(right_text):
        return None,{**review,'reason':'transcript_ownership_not_verified'}
    left=lo/rate+words[split-1]['end'];right=lo/rate+words[split]['start']
    if right-left<.015 or max(abs(left-previous['end']),abs(right-current['start']))>.5:
        return None,{**review,'reason':'local_alignment_disagrees_or_no_gap'}
    lexical=[p for p in hint.get('phones',[]) if p['label'].casefold() not in ('','sil','sp','<eps>')]
    if any(p['start']+lo/rate<right-.003 and p['end']+lo/rate>left+.003 for p in lexical):
        return None,{**review,'reason':'phoneme_inside_proposed_gap'}
    boundary,acoustic=verified_dialogue_boundary(values,rate,left,right)
    if boundary is None:return None,{**review,'reason':'no_acoustic_confirmation','acoustic':acoustic}
    return boundary,{**acoustic,'method':'local_phone_alignment_and_verified_quiet',
        'original_asr_anchors':[previous['end'],current['start']],
        'refined_anchors':[left,right],'alignment_status':'aligned','verified':True}


def split_generation(audio,segments,indices,folder,cfg):
    audio,folder=Path(audio),Path(folder)
    wav=folder/'generated.wav'
    decoded_identity=folder/'decoded-input.json'
    old=json.loads(decoded_identity.read_text('utf8')) if decoded_identity.exists() else {}
    source_hash=hashlib.sha256(audio.read_bytes()).hexdigest() if audio.exists() else None
    if not wav.is_file() or (source_hash is not None and old.get('source_sha256')!=source_hash):
        command(['ffmpeg','-nostdin','-v','error','-y','-i',str(audio),'-ar',str(SR),'-ac','2','-c:a','pcm_f32le',str(wav)])
        atomic_json(decoded_identity,{'source_sha256':source_hash})
    values,rate=sf.read(wav,dtype='float32',always_2d=True)
    if rate!=SR or values.shape[1]!=2 or not len(values) or not np.isfinite(values).all():
        raise RuntimeError('Invalid decoded batch TTS format.')
    texts=[spoken_text(segments[i]['translated_text']) for i in indices]
    # One ASR request per complete generated batch. It only locates generated
    # lines; source transcript/translation/character identities stay unchanged.
    from scribe import transcribe_words
    from source_alignment import locate_transcript
    evidence=transcribe_words(wav,folder/'whisper-turbo',cfg['key'],cfg['language'],attempts=1)
    alignment_words=evidence['words']
    forced=cfg.get('tts_forced_alignment',False)
    if forced:
        from tts_forced_alignment import align,locations as forced_locations
        script=align(wav,texts,cfg['language'],folder)
        alignment_words=script['words']
        locations=forced_locations(texts,alignment_words,values,SR,evidence['words'])
    else:locations=locate_transcript([{'original_text':t} for t in texts],evidence['words'],len(values)/SR)
    atomic_json(folder/'locations-initial.json',{'indices':indices,'locations':locations})
    if forced:recovery=[]
    else:
        locations,recovery=recover_between_anchors(locations,evidence['words'],values,SR)
        locations,edge_recovery=recover_edge_anchors(locations,evidence['words'],texts,values,SR)
        recovery+=edge_recovery
    if len(indices)==1 and not forced:
        locations,single_recovery=recover_single_after_reference(locations,evidence['words'],values,SR,segments[indices[0]])
        recovery+=single_recovery
    conflicts=set()
    for i,left in enumerate(locations):
        if not left.get('matched'):continue
        for j in range(i+1,len(locations)):
            right=locations[j]
            if right.get('matched') and (set(left.get('word_indices',[]))&set(right.get('word_indices',[])) or left['end']>right['start']+.002):
                conflicts.update((i,j))
    for i in conflicts:
        locations[i]={**locations[i],'matched':False,'reason':'conflicting_generated_word_ownership'}
    if cfg.get('tts_alignment_review') and audio.is_file() and any(not r.get('matched') and any(c.isalnum() for c in t) for r,t in zip(locations,texts)):
        from tts_alignment_review import review
        try:
            locations,reviewed=review(audio,texts,evidence['words'],locations,folder/'audio-script-review',cfg)
            recovery+=reviewed
        except (RuntimeError,ValueError,OSError) as error:
            atomic_json(folder/'audio-script-review-error.json',{'error':str(error)})
    atomic_json(folder/'locations.json',{'alignment_version':ALIGNMENT_VERSION,'indices':indices,'locations':locations,'neighbor_recovery':recovery})
    if len(locations)!=len(indices):raise RuntimeError('Batch TTS alignment omitted dialogue. Raw generated audio was preserved.')
    if len(indices)==1 and not forced and not cfg.get('keep_original_on_tts_failure') and any(c.isalnum() for c in texts[0]) and not locations[0].get('matched'):
        return values,[(0,len(values))],{'status':'single_turn_text_unverified','neighbor_recovery':[]}
    # Punctuation-only slots have no lexical target and must retain source audio.
    unresolved=[{'local_index':i,'source_index':indices[i],
        'reason':r.get('reason','generated_dialogue_not_aligned')} for i,r in enumerate(locations)
        if any(c.isalnum() for c in texts[i]) and not r.get('matched')]
    keep_failed=cfg.get('keep_original_on_tts_failure',False)
    active=[i for i in range(len(indices)) if any(c.isalnum() for c in texts[i]) and
        (locations[i].get('matched') or not keep_failed)]
    active_locations=[locations[i] for i in active]
    boundaries=[0];boundary_reviews=[];unsafe=set()
    for position,local in enumerate(active):
        location=locations[local]
        # locate_transcript already validates recall, compactness and short
        # utterances. A second raw-character cutoff rejects valid boundaries
        # when Korean ASR spells names or inflections differently.
        if not location['matched']:
            raise RuntimeError(f'Failed to align batch TTS dialogue {indices[local]+1} to generated audio. Raw generation and ASR responses were preserved.')
        if position:
            previous=active_locations[position-1]
            if (previous['word_indices'] and location['word_indices'] and previous['word_indices'][-1]>=location['word_indices'][0]) or previous['end']>location['start']+.002:
                raise RuntimeError('Batch TTS alignment intervals overlap. Audio was not arbitrarily trimmed.')
            boundary,review=verified_dialogue_boundary(values,SR,previous['end'],location['start'])
            if boundary is None:
                try:
                    recovered,local_review=recover_boundary_with_local_alignment(values,SR,previous,location,
                        texts[active[position-1]],texts[local],folder/f'boundary-{active[position-1]:04d}-{local:04d}-mfa.json',cfg)
                    if recovered is not None:boundary,review=recovered,local_review
                    else:review={**review,'local_alignment_review':local_review}
                except (RuntimeError,ValueError,OSError,KeyError) as error:
                    review={**review,'local_alignment_error':str(error)}
            if boundary is None:
                if not keep_failed:raise RuntimeError('Could not identify a safe split boundary between batch TTS dialogue lines.')
                unsafe.update((active[position-1],local))
                # Placeholder is never emitted for either affected line.
                boundary=round((previous['end']+location['start'])*SR/2)
            boundary_reviews.append({'after_local_index':local-1,**review})
            if not boundaries[-1]<boundary<len(values):raise RuntimeError('Invalid batch TTS boundaries.')
            boundaries.append(boundary)
    active_cuts=list(zip(boundaries,[*boundaries[1:],len(values)])) if active else []
    if unsafe:
        unresolved.extend({'local_index':local,'source_index':indices[local],'reason':'unverified_generated_dialogue_boundary'} for local in sorted(unsafe))
        accepted=[(local,location,cut) for local,location,cut in zip(active,active_locations,active_cuts) if local not in unsafe]
        active=[x[0] for x in accepted];active_locations=[x[1] for x in accepted];active_cuts=[x[2] for x in accepted]
    if unresolved and keep_failed:
        # An omitted slot cannot donate its generated speech to a neighboring
        # successful line. Keep accepted turns within their own proven span,
        # allowing only the small acoustic guard used by the normal trimmer.
        for position,local in enumerate(active):
            lo,hi=active_cuts[position];row=locations[local]
            if local and any(x['local_index']==local-1 for x in unresolved):lo=max(lo,round(max(0,row['start']-.08)*SR))
            if local+1<len(indices) and any(x['local_index']==local+1 for x in unresolved):hi=min(hi,round(min(len(values)/SR,row['end']+.1)*SR))
            active_cuts[position]=(lo,hi)
    replay_locations=active_locations
    if forced:
        replay_locations=[{**r,'word_indices':[i for i,w in enumerate(evidence['words'])
            if min(w['end'],r['end'])>max(w['start'],r['start'])]} for r in active_locations]
    active_cuts,excluded=exclude_unrequested_replay(active_cuts,replay_locations,evidence['words'],segments,[indices[i] for i in active],SR,values)
    for item in excluded:item['local_index']=active[item['local_index']]
    cuts=[None]*len(indices)
    for local,cut in zip(active,active_cuts):cuts[local]=cut
    if keep_failed:
        for item in excluded:
            if item.get('keep_original'):
                local=item['local_index'];cuts[local]=None
                unresolved.append({'local_index':local,'source_index':indices[local],'reason':item['method']})
    status='qwen_exact_script_whisper_verified' if forced else 'whisper_turbo_batch_once'
    atomic_json(folder/'cuts.json',{'version':VERSION,'indices':indices,'cuts':cuts,'alignment_status':status,'alignment_version':ALIGNMENT_VERSION,'neighbor_recovery':recovery,'excluded_replay':excluded,'boundary_reviews':boundary_reviews,'kept_original':unresolved})
    return values,cuts,{'status':status,'neighbor_recovery':recovery,'excluded_replay':excluded,'boundary_reviews':boundary_reviews,'kept_original':unresolved,
        'spoken_spans':[{'local_index':i,'start':r['start'],'end':r['end']} for i,r in enumerate(locations)
                        if r.get('matched') and r.get('word_indices') and r.get('lexical_match_verified',True)]}


def synthesize_batches(segments,clean,folder,cfg,progress,status=None,*,groups=None,
                       reference_gap_seconds=0.,tag_mode='speaker',mode='joint_tagged_reference',instructions=None,
                       reference_mode='concatenated',request_groups=None):
    import render
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if not segments:return []
    groups=plan_batches(segments) if groups is None else groups
    context_only=set(range(len(segments)))-{i for group in groups for i in group}
    if request_groups is None:
        if context_only:raise RuntimeError('The batch plan is missing dialogue.')
        request_groups=groups
    if context_only-{i for r in request_groups for i in r}:raise RuntimeError('Unused TTS context indices')
    if len(request_groups)!=len(groups) or any(not set(g)<=set(r) or len(r)!=len(set(r)) or any(type(i) is not int or not 0<=i<len(segments) for i in r) for g,r in zip(groups,request_groups)):
        raise RuntimeError('Invalid TTS context request plan')
    if sorted(i for group in groups for i in group)!=sorted(set(range(len(segments)))-context_only):
        raise RuntimeError('The batch plan contains duplicate or missing dialogue.')
    if any(not group for group in groups):raise RuntimeError('A TTS batch cannot be empty.')
    if reference_mode not in ('concatenated','continuous_source'):raise ValueError('Invalid reference mode')
    done=0;completed_batches=0
    def update(detail):
        if status:status(done/len(segments)*100,detail)
        else:progress(done,0,len(segments)-done,0,0)
    update(f'화자별 묶음 TTS 준비 · {len(groups)}묶음 · '+('원본 쉼 유지' if reference_mode=='continuous_source' else f'참조 사이 {reference_gap_seconds:g}초'))
    references={};timing_references={}
    for i,s in enumerate(segments):
        if not any(c.isalnum() for c in s['translated_text']):continue
        ref=folder/f'{i:04d}-reference.wav';render.reference_audio(clean,s,ref);references[i]=ref
        timing=folder/f'{i:04d}-timing-reference.wav'
        render.reference_audio(clean,{'lo':s['lo'],'hi':s['hi']},timing);timing_references[i]=timing

    def keep_original(i,reason='no_lexical_dubbing_text'):
        segments[i]['keep_original']=True
        sf.write(folder/f'{i:04d}-fitted.wav',np.zeros((1,2),dtype=np.float32),SR,subtype='FLOAT')
        kept={'index':i,'start':segments[i]['start'],'end':segments[i]['end'],
            'keep_original':True,'reason':reason,'output_frames':1}
        atomic_json(folder/f'{i:04d}.json',kept)
        return {'index':i,'cached':kept}

    def generate_group(group_index,indices,output_indices=None):
        output_indices=set(indices if output_indices is None else output_indices)
        if not output_indices:return []
        request_indices=[i for i in indices if any(c.isalnum() for c in segments[i]['translated_text'])]
        if not request_indices:return [keep_original(i) for i in indices if i in output_indices]
        original,translated,request_tags=tagged_text(segments,request_indices,tag_mode)
        tags=[request_tags[request_indices.index(i)] if i in request_indices else None for i in indices]
        request_instructions=None if cfg.get('tts_omit_instructions') else instructions if instructions is not None else batch_instructions(cfg['language'])
        policy=hashlib.sha256((render.tts_policy(cfg['language'],cfg.get('tts_provider_options'))+'\n'+json.dumps(request_instructions,ensure_ascii=False)).encode()).hexdigest()
        digest=hashlib.sha256(json.dumps([VERSION,policy,mode,reference_gap_seconds,tag_mode,reference_mode,indices,original,translated],ensure_ascii=False).encode())
        if cfg.get('batch_regeneration_nonce') is not None:digest.update(str(cfg['batch_regeneration_nonce']).encode())
        for i in request_indices:digest.update(references[i].read_bytes())
        continuous=None
        if reference_mode=='continuous_source':
            if indices!=list(range(indices[0],indices[-1]+1)):raise RuntimeError('An unrelated dialogue interrupts the continuous reference group.')
            stage=folder/'continuous-references';stage.mkdir(exist_ok=True)
            continuous=stage/f'{group_index:04d}.wav'
            render.reference_audio(clean,{'lo':segments[indices[0]]['lo'],'hi':segments[indices[-1]]['hi']},continuous)
            digest.update(continuous.read_bytes())
        identity=digest.hexdigest()
        group_folder=folder/'joint-batches-v1'/f'{group_index:04d}-{identity[:16]}'
        group_folder.mkdir(parents=True,exist_ok=True)
        ref=group_folder/'reference.wav';parts=[]
        for i in request_indices:
            values,rate=sf.read(references[i],dtype='float32',always_2d=True)
            if rate!=SR:raise RuntimeError('Grouped reference sample rates do not match.')
            if parts and reference_gap_seconds:
                parts.append(np.zeros((round(reference_gap_seconds*SR),values.shape[1]),dtype=np.float32))
            parts.append(values)
        if not ref.is_file():
            if continuous is not None:shutil.copy2(continuous,ref)
            else:sf.write(ref,np.concatenate(parts),SR,subtype='PCM_16')
        request_info={'version':VERSION,'identity':identity,'indices':indices,'output_indices':sorted(output_indices),'speaker_tags':tags,
            'original_text':original,'translated_text':translated,'tts_policy':policy,'instructions':request_instructions,
            'mode':mode,'reference_gap_seconds':reference_gap_seconds,'tag_mode':tag_mode,'reference_mode':reference_mode,'delivery_cue':cfg.get('tts_delivery_cue')}
        atomic_json(group_folder/'input.json',request_info)
        audio=group_folder/'tts.mp3';receipt=group_folder/'generation.json'
        if audio.is_file() and receipt.is_file():generation=json.loads(receipt.read_text(encoding='utf8'))
        else:
            try:
                generation=render.request_audio(ref,{'original_text':original,'translated_text':translated,'_joint_generation':True,'_preserve_speaker_tags':tag_mode in ('reference_slot','reference_slot_omit_first'),'_instructions':request_instructions,'_omit_instructions':request_instructions is None,'_delivery_cue':cfg.get('tts_delivery_cue'),'_provider_options':cfg.get('tts_provider_options',{})},cfg['language'],cfg['key'],audio)
            except RuntimeError as error:
                if 'HTTP 413.' not in str(error) or len(indices)==1:
                    if cfg.get('keep_original_on_tts_failure'):
                        atomic_json(group_folder/'kept-original.json',{'indices':indices,'stage':'tts_generation','error':str(error)})
                        return [keep_original(i,'tts_generation_failed') for i in indices if i in output_indices]
                    raise
                # Split only after the actual provider rejects request size,
                # never because a locally chosen duration threshold is crossed.
                atomic_json(group_folder/'size-rejection.json',{'http_status':413,'indices':indices,'reference_bytes':ref.stat().st_size})
                mid=len(indices)//2
                return generate_group(group_index,indices[:mid],output_indices.intersection(indices[:mid]))+generate_group(group_index,indices[mid:],output_indices.intersection(indices[mid:]))
            generation.update(tts_policy=policy,instructions=request_instructions)
            atomic_json(receipt,generation)
        update(f'묶음 {group_index+1}/{len(groups)} · '+('생성 음성 전사·대사별 정렬' if len(indices)>1 else '단일 대사 준비'))
        try:values,cuts,alignment=split_generation(audio,segments,indices,group_folder,cfg)
        except RuntimeError as error:
            if not cfg.get('keep_original_on_tts_failure'):raise
            atomic_json(group_folder/'kept-original.json',{'indices':indices,'stage':'generated_audio_alignment','error':str(error)})
            return [keep_original(i,'generated_audio_alignment_failed') for i in indices if i in output_indices]
        prepared=[]
        for local,(i,cut) in enumerate(zip(indices,cuts)):
            if i not in output_indices:continue
            if cut is None:
                reason=next((r['reason'] for r in alignment.get('kept_original',[]) if r['local_index']==local),'no_lexical_dubbing_text')
                prepared.append(keep_original(i,reason))
                continue
            lo,hi=cut
            s=segments[i];generated=folder/f'{i:04d}-joint-tts.wav'
            sf.write(generated,values[lo:hi],SR,subtype='FLOAT')
            meta={**generation,'generation_input':identity,'reference_source':'separated_dialogue',
                'tts_group_generated_seconds':generation.get('generated_seconds'),
                'generated_seconds':(hi-lo)/SR,
                'tts_generation_mode':mode,'tts_batch_index':group_index,'tts_batch_turns':len(indices),
                'tts_reference_gap_seconds':reference_gap_seconds,'tts_tag_mode':tag_mode,
                'tts_source_speaker_id':s.get('tts_speaker_id'),'tts_source_speaker_status':s.get('tts_speaker_status'),
                'tts_batch_version':VERSION,'tts_speaker_tag':tags[local],'tts_batch_alignment':alignment['status'],
                'tts_batch_excluded_replay':[r for r in alignment.get('excluded_replay',[]) if r['local_index']==local],
                'tts_batch_alignment_version':ALIGNMENT_VERSION,
                'tts_batch_boundary_reviews':[r for r in alignment.get('boundary_reviews',[]) if r['after_local_index'] in (local-1,local)],
                'tts_batch_neighbor_recovery':[r for r in alignment.get('neighbor_recovery',[]) if r['local_index']==local],
                'tts_batch_cut_frames':[lo,hi],'tts_batch_folder':str(group_folder),'tts_reference_mode':reference_mode}
            span=next((r for r in alignment.get('spoken_spans',[]) if r['local_index']==local),None)
            if span:meta['tts_protected_spoken_end_frames']=max(1,min(hi-lo,round(span['end']*SR)-lo))
            cache=folder/f'{i:04d}-joint-generation.json';atomic_json(cache,meta)
            next_start=s.get('next_dialogue_start_frame',segments[i+1]['lo'] if i+1<len(segments) else None)
            prepared.append(dict(index=i,segment=s,folder=folder,reference=references[i],timing_reference=timing_references[i],
                generated=generated,fitted=folder/f'{i:04d}-fitted.wav',metadata=folder/f'{i:04d}.json',generation_cache=cache,
                generation=meta,policy=policy,frames=s['hi']-s['lo'],
                window=[s.get('speech_lo',s['lo'])-s['lo'],s.get('speech_hi',s['hi'])-s['lo']],
                collision_budget=None if next_start is None else next_start-s['lo']))
        return prepared
    results={}
    def fit_or_keep(item):
        try:return render.fit_speech(item,cfg)
        except RuntimeError as error:
            if not cfg.get('keep_original_on_tts_failure'):raise
            atomic_json(folder/f'{item["index"]:04d}-fit-failure.json',{'error':str(error)})
            return keep_original(item['index'],'tts_timing_failed')['cached']
    # Independent batch requests and per-turn fitting use separate bounded pools.
    with ThreadPoolExecutor(max_workers=TTS_CONCURRENCY) as requests,ThreadPoolExecutor(max_workers=FIT_CONCURRENCY) as fits:
        generating={requests.submit(generate_group,n,request_groups[n],indices):n for n,indices in enumerate(groups)}
        fitting={}
        while generating or fitting:
            finished,_=wait([*generating,*fitting],return_when=FIRST_COMPLETED)
            for future in finished:
                if future in generating:
                    generating.pop(future);items=future.result();completed_batches+=1
                    for item in items:fitting[fits.submit(fit_or_keep,item)]=item['index']
                else:
                    i=fitting.pop(future);results[i]=future.result();done+=1
            update(f'묶음 TTS {completed_batches}/{len(groups)} · 대사 완료 {done}/{len(segments)} · 보정 {len(fitting)}')
    atomic_json(folder/'joint-batches.json',{'version':VERSION,'target_reference_seconds':BATCH_REFERENCE_POLICY['target_seconds'],'max_parts':None,'groups':groups,'request_groups':request_groups,'mode':mode,
        'reference_gap_seconds':reference_gap_seconds,'tag_mode':tag_mode,'reference_mode':reference_mode})
    return [results[i] if i in results else {'index':i,'context_only':True} for i in range(len(segments))]
