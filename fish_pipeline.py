"""Whole-recording Fish rich transcription supplies both voices and text."""
import re
import json
from source_asr import transcribe_words,MODEL,CACHE_DIRECTORY,INPUT_POLICY
from rich_transcript import enrich,join_words
from mai_pipeline import normalize,PROMPT as BASE_PROMPT

AUTHORITY='fish_transcribe_1_pro'
PROMPT=BASE_PROMPT.replace('MAI','Fish')+' Preserve inline square-bracket emotion/delivery cues at their corresponding positions in the translation. Treat these as TTS directions, not spoken dialogue. Exclude background sound-event labels from spoken text. Remove speaker markup from output text; identity is already supplied separately.'

def source(clean,folder,key,progress,language=None):
    hint={'fil':'tl'}.get(language,language)
    result=transcribe_words(clean,folder,key,language=hint,progress=lambda p,d:progress(p*.3,d))
    return result,{**result['signature'],'rich_transcript_policy':result.get('rich_transcript_policy')}

def measured_word_tails(raw,clean):
    """Do not count a long silent ASR word tail as continuous speech."""
    import numpy as np
    import soundfile as sf
    words=[]
    with sf.SoundFile(clean) as audio:
        sr=audio.samplerate;hop=max(1,round(.01*sr))
        for original in raw.get('words',[]):
            word=dict(original);start,end=word['start'],word['end']
            if end-start>.8:
                lo=max(0,round(start*sr));hi=min(len(audio),round(end*sr));audio.seek(lo)
                values=audio.read(max(0,hi-lo),dtype='float32',always_2d=True)
                count=len(values)//hop
                if count:
                    energy=np.sqrt(np.mean(values[:count*hop].reshape(count,hop,audio.channels)**2,axis=1)).max(axis=1)
                    threshold=max(1e-6,float(np.percentile(energy,90))*.025,float(energy.max())*.0025)
                    active=np.flatnonzero(energy>threshold)
                    if len(active):
                        measured=min(end,(lo+(int(active[-1])+1)*hop)/sr+.04)
                        if end-measured>.8 and measured>start:
                            word.update(end=measured,asr_end=end,timing_repair='measured_long_silent_word_tail')
            words.append(word)
    return {**raw,'words':words}


def word_speaker(word):
    voice=word.get('fish_speaker') or word.get('speaker',word.get('speaker_id'))
    match=re.fullmatch(r'<\|speaker:(\d+)\|>',str(voice))
    return match.group(1) if match else str(voice) if voice is not None else 'unknown'


def restore_collapsed_continuations(rich,clean):
    """Recover phrase membership, not invented word durations, at voice collisions."""
    import numpy as np
    import soundfile as sf
    from rich_transcript import spoken
    words=[dict(w) for w in sorted(rich.get('words',[]),key=lambda w:(w['start'],w['end']))]
    with sf.SoundFile(clean) as audio:
        sr=audio.samplerate;hop=max(1,round(.01*sr))
        def energy(start,end):
            lo=max(0,round(start*sr));hi=min(len(audio),round(end*sr));audio.seek(lo)
            values=audio.read(max(0,hi-lo),dtype='float32',always_2d=True)
            count=len(values)//hop
            return (np.sqrt(np.mean(values[:count*hop].reshape(count,hop,audio.channels)**2,axis=1)).max(axis=1)
                    if count else np.array([]))
        for i,word in enumerate(words):
            if word['start']!=word['end'] or not spoken(word.get('word','')).strip() or i+1>=len(words):continue
            following=words[i+1];voice=word_speaker(word)
            collision=(voice!='unknown' and word_speaker(following) not in (voice,'unknown')
                       and following['end']>following['start'] and abs(following['start']-word['start'])<=.02)
            if not collision:continue
            word['point_anchor_conflicts_with_next_speaker']=True
            if i==0:continue
            previous=words[i-1]
            # Text alone must not merge completed replies, other voices or
            # independently audible speech across a real pause.
            if (word_speaker(previous)!=voice or previous['end']<=previous['start']
                or not re.search(r'[,，、:：;；]$',previous.get('rich_surface',previous.get('word','')).strip())
                or word['start']-previous['end']<=.8):continue
            begin=i-1
            while begin>0:
                prior=words[begin-1]
                if (word_speaker(prior)!=voice or words[begin]['start']-prior['end']>.8
                    or re.search(r'[。！？!?]$',prior.get('rich_surface',prior.get('word','')).strip())):break
                begin-=1
            reference=energy(words[begin]['start'],previous['end'])
            gap=energy(previous['end']+.04,word['start']-.35)
            if not len(reference) or len(gap)<30 or reference.max()<1e-5:continue
            threshold=max(1e-6,float(np.percentile(reference,90))*.025)
            active=gap>threshold
            edges=np.diff(np.r_[False,active,False].astype(int))
            longest=max((b-a for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))),default=0)
            if np.mean(active)>.05 or longest>=8:continue
            word.update(asr_start=word['start'],asr_end=word['end'],start=previous['end'],end=previous['end'],
                timing_repair='unfinished_same_voice_phrase_before_silent_gap',
                point_anchor_conflicts_with_next_speaker=False)
    return {**rich,'words':words}


def correct_fish_edges(clean,rows,progress=lambda *a:None):
    import soundfile as sf
    from mai_pipeline import correct_edges
    from timing import envelope,measured_energy_gate,runs,HOP
    rows=[dict(row) for row in rows]
    # Some point tokens are attached to the preceding OTHER voice's endpoint.
    # Recover only a single isolated utterance in the otherwise unclaimed gap;
    # a decay touching the previous voice is not a new reply.
    with sf.SoundFile(clean) as audio:
        for i,row in enumerate(rows):
            a=row['start']
            if row['end']!=a or not 0<i<len(rows)-1:continue
            previous,following=rows[i-1],rows[i+1]
            # A collapsed reply may be stamped immediately before the NEXT
            # voice although its isolated waveform lies in the preceding gap.
            if (row['speaker']!='unknown' and previous['speaker'] not in (row['speaker'],'unknown')
                and following['speaker'] not in (row['speaker'],'unknown')
                and 0<=following['start']-a<=.2):
                lo=max(previous['end']+.08,a-2.);hi=a-.15
                if hi-lo>=.2:
                    audio.seek(round(lo*audio.samplerate))
                    values=audio.read(round((min(audio.frames/audio.samplerate,following['start']+.4)-lo)*audio.samplerate),dtype='float32',always_2d=True)
                    levels=envelope(values);threshold,noise=measured_energy_gate(levels,.015,values)
                    levels=levels[:int((hi-lo)*audio.samplerate/HOP)]
                    spans=[]
                    for x,y in runs(levels>threshold):
                        if spans and (x-spans[-1][1])*HOP/audio.samplerate<=.25:spans[-1]=(spans[-1][0],y)
                        else:spans.append((x,y))
                    islands=[(lo+x*HOP/audio.samplerate,lo+y*HOP/audio.samplerate) for x,y in spans
                        if (y-x)*HOP/audio.samplerate>=.06 and x*HOP/audio.samplerate>=.04
                        and (hi-lo)-y*HOP/audio.samplerate>=.08]
                    if len(islands)==1:
                        start,end=islands[0]
                        row.update(start=max(lo,start-.02),end=min(hi,end+.04),
                            point_recovery={'asr_start':a,'asr_end':a,
                                'method':'unique_isolated_reply_before_next_voice_anchor','noise':noise})
                        continue
            if (row['speaker']=='unknown' or previous['speaker'] in (row['speaker'],'unknown')
                or abs(previous['end']-a)>.02 or following['start']-a<.4):continue
            lo=a;hi=following['start']-.1
            audio.seek(round(lo*audio.samplerate));values=audio.read(round((hi-lo)*audio.samplerate),dtype='float32',always_2d=True)
            levels=envelope(values);threshold,noise=measured_energy_gate(levels,.02,values)
            spans=[]
            for start,end in runs(levels>threshold):
                if spans and start-spans[-1][1]<=3:spans[-1]=(spans[-1][0],end)
                else:spans.append((start,end))
            islands=[(lo+x*HOP/audio.samplerate,lo+y*HOP/audio.samplerate) for x,y in spans
                     if y-x>=3 and x*HOP/audio.samplerate>=.12 and (len(levels)-y)*HOP/audio.samplerate>=.12]
            if len(islands)!=1:continue
            start,end=islands[0]
            row.update(start=max(lo,start-.03),end=min(hi,end+.05),
                point_recovery={'asr_start':a,'asr_end':a,'method':'unique_isolated_reply_after_other_voice_tail','noise':noise})
    corrected,audit=correct_edges(clean,rows,progress)
    for i,row in enumerate(rows):
        if row.get('point_recovery'):
            corrected[i]={**row,'id':row.get('id',i)}
            audit[i].update(original_start=row['point_recovery']['asr_start'],original_end=row['point_recovery']['asr_end'],
                            start=row['start'],end=row['end'],method=row['point_recovery']['method'])
        repairs=[{k:w[k] for k in ('word','asr_start','asr_end','timing_repair') if k in w}
                 for w in row.get('words',[]) if w.get('timing_repair')]
        if repairs:audit[i]['word_timing_repairs']=repairs
        if row['start']==row['end'] and any(w.get('point_anchor_conflicts_with_next_speaker') for w in row.get('words',[])):
            corrected[i]={**row,'id':row.get('id',i),'keep_original':True}
            audit[i].update(start=row['start'],end=row['end'],method='unresolved_point_at_other_voice_boundary_keep_original')
    return corrected,audit


def regions(raw,duration,*,clean=None):
    if clean is not None:raw=measured_word_tails(raw,clean)
    rich=enrich(raw)
    words=[]
    for word in rich.get('words',[]):
        if (words and word.get('numeric_span') and word.get('numeric_span')==words[-1].get('numeric_span')
            and word_speaker(word)==word_speaker(words[-1])):
            previous=words[-1]
            previous.update(end=max(previous['end'],word['end']),word=previous['word']+word['word'],
                rich_surface=previous.get('rich_surface',previous['word']).rstrip()+word.get('rich_surface',word['word']))
        else:words.append(dict(word))
    rich={**rich,'words':words}
    if clean is not None:rich=restore_collapsed_continuations(rich,clean)
    groups=[]
    for index,word in enumerate(sorted(rich.get('words',[]),key=lambda w:(w['start'],w['end']))):
        if word['end']<word['start']:continue
        word={**word,'word_id':index}
        speaker=word_speaker(word)
        # Punctuation has no audible duration. Its late point timestamp must
        # not extend a voice's source window into a following reply.
        if not any(c.isalnum() for c in word.get('word','')):
            if groups and speaker==groups[-1]['speaker']:groups[-1]['words'].append(word)
            continue
        if groups and speaker==groups[-1]['speaker'] and word['start']-groups[-1]['end']<=.8+1e-7:
            groups[-1]['end']=max(groups[-1]['end'],word['end']);groups[-1]['words'].append(word)
        else:groups.append({'start':word['start'],'end':word['end'],'speaker':speaker,'words':[word]})
    if not groups:raise RuntimeError('Fish transcription contains no valid word timestamps. The raw response was preserved.')
    rows=normalize({'segments':[{**g,'text':join_words(g['words'])} for g in groups]},duration,allow_point_anchors=True)
    for row in rows:
        row['words']=[w for g in groups for w in g['words'] if
            min(g['end'],row['end'])>max(g['start'],row['start']) or
            (g['start']==g['end'] and g['speaker']==row['speaker'] and
                (row['start']<=g['start']<row['end'] or row['start']==row['end']==g['start']))]
    from source_word_boundaries import repair_japanese_boundaries
    rows,_=repair_japanese_boundaries(rows,raw.get('text',''))
    return rows

TEXT_PROMPT='''Translate each numbered source dialogue faithfully into natural spoken {language}. Preserve meaning, expressive punctuation, hesitations and unfinished speech. The source transcript, speaker IDs and boundaries are fixed by Fish ASR and local acoustic correction. Never change speakers, move words between IDs, merge regions or rewrite the source. No audio is provided. Preserve EVERY supplied square-bracket cue VERBATIM in its ORIGINAL language, including its exact spelling and bracket contents, at the corresponding position. Never translate, paraphrase, replace, omit or add cues. For example, keep [足音] exactly as [足音]; never turn it into [footsteps]. Translate only spoken dialogue outside the brackets. Do not invent or intensify cues.
Return only a JSON object mapping every supplied ID to a translated string, or null for nonverbal/background-only sounds needing no dubbing. Preserve meaningful short replies. Do not output speakers, source text, timestamps or explanations.'''


def fixed_translations(content,rows):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate translation ID')
            result[key]=value
        return result
    values=json.loads(content,object_pairs_hook=unique)
    if not isinstance(values,dict) or set(values)!={str(r['id']) for r in rows}:
        raise ValueError('Every fixed Fish region ID is required exactly once')
    if any(v is not None and (not isinstance(v,str) or not v.strip() or len(v)>4000) for v in values.values()):
        raise ValueError('Only translated strings or null are allowed; speakers are fixed')
    for row in rows:
        text=values[str(row['id'])]
        if text is not None and re.findall(r'\[[^\]\n]*\]',text)!=re.findall(r'\[[^\]\n]*\]',row['text']):
            raise ValueError('Performance cues must remain verbatim in their original language')
    return {'values':[values[str(r['id'])] for r in rows],'speaker_corrections':{}}


def translate(clean,work,output,cfg,progress):
    from mai_pipeline import translate as translate_regions
    from fish_qwen_pipeline import schema
    from mai_native_pipeline import record_auto_language
    from core import LANGUAGES,GEMINI,atomic_json
    from pathlib import Path
    def load(clean,folder,key,progress):
        raw,identity=source(clean,folder,key,progress,language=cfg.get('source_language'))
        record_auto_language(raw,cfg)
        return raw,{**identity,'translation_policy':'fixed_fish_speakers_original_cues_corrected_gap_groups_v7'}
    def boundaries(clean,rows,progress):
        from corrected_source_groups import merge_corrected
        corrected,audit=correct_fish_edges(clean,rows,progress)
        merged,merges=merge_corrected(corrected,.8)
        return merged,audit+merges
    result=translate_regions(clean,work,output,cfg,progress,source_loader=load,region_builder=lambda raw,duration:regions(raw,duration,clean=clean),
        source_model=MODEL,authority=AUTHORITY,prompt=TEXT_PROMPT.format(language=LANGUAGES[cfg['language']]),boundary_builder=boundaries,
        cache_directory=CACHE_DIRECTORY,source_input_policy=INPUT_POLICY,prefix='fish',
        response_parser=fixed_translations,format_builder=lambda:schema('translations'))
    result.update(preserve_source_speaker_ids=True,speaker_authority=MODEL,source_transcription_model=MODEL,
        transcription_mode='fish_native_transcript_speakers_local_edges_gemini_text_translation',
        translation_output='region_id_to_translation_or_null_only',
        models={'source_asr':MODEL,'speakers':MODEL,'translation':GEMINI},fish_asr_requests=1)
    atomic_json(Path(output),result)
    return result
