"""One whole-recording MAI request, bounded acoustic edges, text-only translation."""
from pathlib import Path
from local_resources import local_stage
import base64, hashlib, json, math, urllib.request, urllib.error
import numpy as np
import soundfile as sf
from core import SR, GEMINI, LANGUAGES, atomic_json, command,MAI_LANGUAGES

MODEL='microsoft/mai-transcribe-2'
VERSION=1
AUTHORITY='mai_transcribe_2'
WORD_GROUPING={'version':1,'same_speaker_max_gap_seconds':.8,'speaker_change':'split_immediately'}
OPTIONS={'azure':{'diarization':{'enabled':True},'enhancedMode':{'modelOptions':{'transcribeStyle':'verbatim'}}}}
PROMPT='''Translate each region into natural dubbing dialogue in target_language, preserving meaning, character, intent and expressive punctuation. Use surrounding dialogue as context. Translate ordinary expressions, emotional reactions and mimetic words by their intended meaning and function, rather than merely rendering their source pronunciation in the target script. Preserve proper names, established loanwords, character-specific speech and intentionally foreign wording when context warrants; do not force cultural localization or flatten the original style. Speaker IDs are MAI voice assignments, not guaranteed character identities. Keep every region separate and in order; do not invent, merge or omit dialogue.
Keep MAI speaker IDs by default. Only when explicit dialogue context makes a particular assignment clearly contradictory and clearly identifies an existing alternative speaker, correct that region's ID. If uncertain, keep MAI. You receive text, not audio: do not infer vocal identity or gender from wording, speaking style, names, or outside knowledge alone. Do not merge narration, inner monologue or distinct voices merely because they belong to the same character. Never change unknown/mixed-speaker regions to a single speaker. Do not invent new IDs or change timing.
Return only an object mapping every input id to its translated string, or null for nonverbal sounds or utterances needing no dubbing. Only for an unmistakable speaker correction return [translated_string, corrected_existing_speaker_id] instead of a string. Do not repeat unchanged IDs, source text, timestamps or explanations.'''

def response_format():
    value={'anyOf':[{'type':'null'},{'type':'string'},
        {'type':'array','minItems':2,'maxItems':2,'items':{'type':'string'}}]}
    return {'type':'json_schema','json_schema':{'name':'mai_translations','strict':True,
        'schema':{'type':'object','additionalProperties':value}}}

def parse_translations(content,rows,*,allow_unknown_correction=False):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError(f'Duplicate region ID {key}')
            result[key]=value
        return result
    data=json.loads(content,object_pairs_hook=unique)
    if not isinstance(data,dict) or set(data)!={str(r['id']) for r in rows}:raise ValueError('Expected every source region ID exactly once')
    speakers={r['speaker'] for r in rows if r['speaker']!='unknown'}
    values=[];corrections={}
    for row in rows:
        value=data[str(row['id'])]
        if isinstance(value,list):
            if len(value)!=2 or not all(isinstance(v,str) for v in value) or value[1] not in speakers:
                raise ValueError('Speaker correction must name an existing source speaker')
            if row['speaker']=='unknown' and not allow_unknown_correction:raise ValueError('Mixed/unknown region cannot become a single speaker')
            text,speaker=value
            if speaker!=row['speaker']:corrections[str(row['id'])]=speaker
        else:text=value
        if text is not None and (not isinstance(text,str) or not text.strip() or len(text)>4000):raise ValueError('Expected a translated string or null')
        values.append(text.strip() if isinstance(text,str) else None)
    return {'values':values,'speaker_corrections':corrections}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def normalize(raw,duration,*,allow_point_anchors=False):
    rows=[]
    for item in raw.get('segments',[]):
        a,b=item.get('start'),item.get('end');text=item.get('text','').strip()
        if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not all(map(math.isfinite,[a,b])):
            raise RuntimeError('Invalid MAI segment timestamps. The raw response was preserved.')
        a,b=max(0.,float(a)),min(duration,float(b))
        if not text or b<a or (b==a and not allow_point_anchors):continue
        speaker=item.get('speaker',item.get('speaker_id',item.get('speaker_label')))
        row={'start':a,'end':b,'text':text,'speaker':str(speaker) if speaker is not None else 'unknown'}
        rows.append(row)
    rows.sort(key=lambda r:(r['start'],r['end']))
    result=[]
    for row in rows:
        # Simultaneous speech has one replacement window, never an invented cut.
        if result and row['start']<result[-1]['end']:
            old=result[-1];old['end']=max(old['end'],row['end']);old['text']+=' '+row['text']
            if old['speaker']!=row['speaker']:old['speaker']='unknown'
        else:result.append(dict(row))
    if not result and str(raw.get('text','')).strip():
        raise RuntimeError('MAI transcription has no segment timestamps. The raw response was preserved.')
    if not result and not isinstance(raw.get('segments'),list):
        raise RuntimeError('MAI transcription response has no segments. The raw response was preserved.')
    return result

def word_regions(raw,duration,*,merge_breaks=()):
    """Rebuild regions from MAI words while retaining punctuated source text."""
    segments=normalize(raw,duration)
    words=[]
    for w in raw.get('words',[]):
        a,b=w.get('start'),w.get('end');text=w.get('word',w.get('text',''))
        if not isinstance(text,str) or not text.strip():continue
        if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not all(map(math.isfinite,[a,b])) or b<=a:continue
        a,b=max(0.,float(a)),min(duration,float(b))
        if b<=a:continue
        owner=max(segments,key=lambda s:max(0.,min(b,s['end'])-max(a,s['start'])),default=None)
        if owner is None or min(b,owner['end'])<=max(a,owner['start']):continue
        speaker=w.get('speaker',w.get('speaker_id',owner['speaker']))
        words.append({'start':a,'end':b,'text':text,'speaker':str(speaker) if speaker is not None else 'unknown','owner':id(owner)})
    if not words:return segments
    words.sort(key=lambda w:(w['start'],w['end']))
    for segment in segments:
        members=[w for w in words if w['owner']==id(segment)]
        if not members:
            # Never drop text just because MAI omitted word annotations.
            words.append({**segment,'owner':id(segment)});continue
        text=segment['text'];positions=[];cursor=0
        for w in members:
            pos=text.find(w['text'],cursor)
            if pos<0:positions=[];break
            positions.append(pos);cursor=pos+len(w['text'])
        if positions:
            for i,w in enumerate(members):
                lo=0 if i==0 else positions[i];hi=positions[i+1] if i+1<len(members) else len(text)
                w['text']=text[lo:hi]
        else:
            # Provider tokenization may normalize spelling. Keep every word,
            # with spaces only where both neighboring tokens use Latin letters.
            for a,b in zip(members,members[1:]):
                if a['text'][-1:].isascii() and b['text'][:1].isascii():a['text']+=' '
    words.sort(key=lambda w:(w['start'],w['end']))
    rows=[]
    for w in words:
        blocked=bool(rows) and any(rows[-1]['start']<cut<=w['start'] for cut in merge_breaks)
        if rows and not blocked and w['speaker']==rows[-1]['speaker'] and w['start']-rows[-1]['end']<=.8+1e-7:
            rows[-1]['end']=max(rows[-1]['end'],w['end']);rows[-1]['text']+=w['text']
        else:rows.append({k:w[k] for k in ('start','end','speaker','text')})
    # Preserve true overlap as a mixed replacement window, as for segment input.
    return normalize({'segments':rows},duration)

def transcribe_source(clean,folder,key,progress,*,language=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    signature={'model':MODEL,'version':VERSION,'source_sha256':digest(clean),'options':OPTIONS}
    if language is not None:
        if language not in MAI_LANGUAGES:raise ValueError('Invalid source language')
        signature['language']=language
    receipt=folder/'source.json';raw_path=folder/'raw.json'
    if receipt.exists() and raw_path.exists() and json.loads(receipt.read_text('utf8')).get('signature')==signature:
        raw=json.loads(raw_path.read_text('utf8'));raw['_cache_reused']=True;progress(30,'MAI 전사·화자 캐시 재사용')
    else:
        if not key:raise RuntimeError('OPENROUTER_API_KEY is not configured.')
        progress(3,'MAI 전체 음성 입력 준비')
        audio=folder/'input.flac'
        command(['ffmpeg','-nostdin','-v','error','-y','-i',str(clean),'-ac','1','-ar','44100','-sample_fmt','s16','-c:a','flac',str(audio)])
        fmt='flac'
        # Keep the full recording clock. Long audio is compressed, never compacted.
        if audio.stat().st_size>24*1024**2:
            audio=folder/'input.mp3';fmt='mp3'
            command(['ffmpeg','-nostdin','-v','error','-y','-i',str(clean),'-ac','1','-ar','44100','-c:a','libmp3lame','-b:a','96k',str(audio)])
        if audio.stat().st_size>24*1024**2:raise RuntimeError('MAI input exceeds 24 MB. Audio was not automatically split.')
        payload={'model':MODEL,'input_audio':{'data':base64.b64encode(audio.read_bytes()).decode(),'format':fmt},
            'response_format':'verbose_json','timestamp_granularities':['segment','word'],'provider':{'options':OPTIONS}}
        if language is not None:payload['language']=language
        request=urllib.request.Request('https://openrouter.ai/api/v1/audio/transcriptions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Title':'Fast Dubbing'})
        progress(8,'MAI 전체 음성 전사·화자 분석 요청 중 · 한 번 요청')
        try:
            with urllib.request.urlopen(request,timeout=600) as response:raw=json.load(response)
        except urllib.error.HTTPError as error:
            atomic_json(folder/'error.json',{'status':error.code,'response':error.read().decode('utf8',errors='replace')})
            raise RuntimeError(f'MAI transcription HTTP {error.code}. The raw error response was preserved.') from None
        atomic_json(raw_path,raw)
        normalize(raw,sf.info(clean).duration)
        atomic_json(receipt,{'signature':signature,'input_format':fmt,'input_bytes':audio.stat().st_size})
    return raw,signature

@local_stage
def correct_edges(clean,rows,progress=lambda *a:None):
    from timing import envelope,measured_energy_gate,runs,HOP
    info=sf.info(clean)
    if info.samplerate!=SR:raise RuntimeError('MAI boundary correction input must be sampled at 44.1 kHz.')
    corrected=[];audit=[]
    with sf.SoundFile(clean) as audio:
        for i,row in enumerate(rows):
            a,b=row['start'],row['end']
            # Neighbor midpoints forbid borrowing another annotated utterance.
            lower=(rows[i-1]['end']+a)/2 if i else 0.
            upper=(b+rows[i+1]['start'])/2 if i+1<len(rows) else info.duration
            lo=max(lower,a-.6);hi=min(upper,b+.3)
            left=round(lo*SR);right=round(hi*SR)
            audio.seek(left);v=audio.read(right-left,dtype='float32',always_2d=True)
            levels=envelope(v);threshold,noise=measured_energy_gate(levels,.015,v)
            quiet=[(lo+x*HOP/SR,lo+y*HOP/SR) for x,y in runs(levels<=threshold) if y-x>=3]
            if a==b:
                # A zero-duration ASR token is still recognized speech. Use its
                # point as an acoustic anchor, never discard its transcript.
                speech=[(lo+x*HOP/SR,min(hi,lo+y*HOP/SR)) for x,y in runs(levels>threshold) if y>x]
                if speech:
                    start,end=min(speech,key=lambda span:max(span[0]-a,a-span[1],0.))
                    corrected.append({**row,'id':row.get('id',i),'start':start,'end':end})
                    audit.append({'source_index':i,'original_start':a,'original_end':b,'start':start,'end':end,'noise':noise,'method':'zero_duration_token_nearest_bounded_acoustic_run'})
                else:
                    corrected.append({**row,'id':row.get('id',i),'keep_original':True})
                    audit.append({'source_index':i,'original_start':a,'original_end':b,'start':a,'end':b,'method':'zero_duration_token_without_measurable_audio_keep_original'})
                progress(30+20*(i+1)/max(1,len(rows)),f'MAI 시작·끝 파형 보정 {i+1}/{len(rows)}')
                continue
            # Snap to the nearest quiet gap, retaining weak consonants and tails.
            starts=[(x+y)/2 for x,y in quiet if abs((x+y)/2-a)<=.3 and (x+y)/2<b]
            ends=[(x+y)/2 for x,y in quiet if abs((x+y)/2-b)<=.3 and (x+y)/2>a]
            start=min(starts,key=lambda t:abs(t-a)) if starts else a
            end=min(ends,key=lambda t:abs(t-b)) if ends else b
            # If a boundary falls in continuous speech, walk to the nearest outer
            # quiet gap rather than cutting inward through an active waveform.
            def active(t):
                j=min(len(levels)-1,max(0,int((t-lo)*SR/HOP)))
                return bool(len(levels) and levels[j]>threshold)
            if active(a):
                before=[y for x,y in quiet if y<=a]
                start=max(before) if before else lo
            # A late ASR onset can sit after an intraword consonant closure.
            # Recover the connected leading speech island, never crossing the
            # neighbor midpoint or a genuine long pause.
            if active(a):
                speech=[(lo+x*HOP/SR,lo+y*HOP/SR) for x,y in runs(levels>threshold) if y-x>=3]
                anchor=next((j for j,(x,y) in enumerate(speech) if x<=a<y),None)
                if anchor is not None:
                    first=anchor
                    while first and speech[first][0]-speech[first-1][1]<=.12 and a-speech[first-1][0]<=.6:
                        first-=1
                    if first<anchor:start=min(start,max(lower,speech[first][0]-.02))
            if active(b):
                after=[x for x,y in quiet if x>=b]
                end=min(after) if after else hi
            start=max(lower,min(start,a+.3));end=min(upper,max(end,b-.3))
            if end<=start:start,end=a,b
            start,end=round(start*SR)/SR,round(end*SR)/SR
            corrected.append({**row,'id':row.get('id',i),'start':start,'end':end})
            audit.append({'source_index':i,'original_start':a,'original_end':b,'start':start,'end':end,'noise':noise,'method':'asr_anchored_nearest_quiet_gap_neighbor_bounded'})
            progress(30+20*(i+1)/max(1,len(rows)),f'MAI 시작·끝 파형 보정 {i+1}/{len(rows)}')
    return corrected,audit

def translate(clean,work,output,cfg,progress,*,source_loader=None,region_builder=None,
        source_model=MODEL,authority=AUTHORITY,prompt=PROMPT,cache_directory='source-asr-mai-v1',
        source_input_policy='whole_recording_original_clock_one_request',prefix='mai',
        input_builder=None,response_parser=None,format_builder=None,region_reviser=None,audio_builder=None,boundary_builder=None):
    import speech
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    raw,source_identity=(source_loader or transcribe_source)(clean,Path(clean).parent/cache_directory,cfg['key'],progress)
    label={'fish':'Fish','gemini':'Gemini','nemotron':'Nemotron + Fish'}.get(prefix,'MAI')
    rows,audit=(boundary_builder or correct_edges)(clean,(region_builder or word_regions)(raw,sf.info(clean).duration),
        lambda p,d:progress(p,d.replace('MAI',label)))
    atomic_json(work/f'{prefix}-boundaries.json',{'rows':rows,'measurements':audit})
    inputs=(input_builder(rows) if input_builder else
        [{'id':r['id'],'speaker':r['speaker'],'text':r['text']} for r in rows])
    parser=lambda text:(response_parser(text,rows) if response_parser else parse_translations(text,rows))
    context={'target_language':LANGUAGES[cfg['language']],'regions':inputs}
    signature={'source':source_identity,'version':VERSION,'word_grouping':WORD_GROUPING,'model':GEMINI,'context':context,'prompt':prompt}
    audio_manifest=audio_builder(clean,work,source_identity) if audio_builder and rows else None
    if audio_manifest:
        signature['audio']={k:v for k,v in audio_manifest.items() if k!='path'}
        atomic_json(work/'gemini-audio-input.json',audio_manifest)
    cache=work/f'{prefix}-translation.json';saved=json.loads(cache.read_text('utf8')) if cache.exists() else {}
    ids=[r['id'] for r in rows]
    translation_requests=0
    if saved.get('signature')==signature:
        parsed=parser(json.dumps(saved['translations'],ensure_ascii=False));usage=saved.get('usage',{})
    elif rows:
        atomic_json(work/'gemini-text-input.json',context)
        payload={'model':GEMINI,'max_tokens':speech.MAX_OUTPUT_TOKENS,'reasoning':{'effort':'low'},'provider':{'require_parameters':True},
            'messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps(context,ensure_ascii=False,separators=(',',':'))}],
            'response_format':(format_builder or response_format)()}
        if audio_manifest:
            payload['messages'][1]['content']=[{'type':'text','text':payload['messages'][1]['content']},
                {'type':'input_audio','input_audio':{'format':audio_manifest['format'],
                    'data':base64.b64encode(Path(audio_manifest['path']).read_bytes()).decode()}}]
        def activity(event):
            n=min(len(rows),event.get('received_segments',0));elapsed=round(event.get('elapsed_seconds',0))
            progress(50+45*n/max(1,len(rows)),f'구간 번역 {n}/{len(rows)}개 수신 · {elapsed}초')
        translation_requests=1
        parsed,usage=speech.http_json(payload,cfg['key'],response_directory=work/f'{prefix}-responses',parser=parser,on_activity=activity,expected_region_ids=ids)
        encoded={str(r['id']):([text,parsed['speaker_corrections'][str(r['id'])]] if str(r['id']) in parsed['speaker_corrections'] else text) for r,text in zip(rows,parsed['values'])}
        atomic_json(cache,{'signature':signature,'translations':parsed.get('response_entries',encoded),'usage':usage})
    else:parsed={'values':[],'speaker_corrections':{}};usage={}
    values=parsed['values'];corrections=[]
    rows=[dict(r) for r in rows]
    for row in rows:
        before=row['speaker'];after=parsed['speaker_corrections'].get(str(row['id']),before)
        row[f'{prefix}_speaker']=before;row['speaker']=after
        if before!=after:corrections.append({'source_region_id':row['id'],f'{prefix}_speaker':before,'corrected_speaker':after,'source':'gemini_explicit_context'})
    reassignments=[]
    if region_reviser:
        rows,reassignments=region_reviser(rows,parsed)
        if reassignments:
            rows,audit=correct_edges(clean,rows,lambda p,d:progress(96,d.replace('MAI',label)))
            atomic_json(work/f'{prefix}-boundaries.json',{'rows':rows,'measurements':audit,'word_reassignments':reassignments})
        atomic_json(work/f'{prefix}-word-reassignments.json',{'changes':reassignments})
        values=parsed.get('revised_values',values)
    corrections=[{'source_region_id':row['id'],f'{prefix}_speaker':row.get(f'{prefix}_speaker',row['speaker']),
        'corrected_speaker':row['speaker'],'source':'gemini_audio_and_context' if audio_manifest else 'gemini_explicit_context',
        **({'parent_region_id':row['parent_region_id']} if 'parent_region_id' in row else {})}
        for row in rows if row.get(f'{prefix}_speaker',row['speaker'])!=row['speaker']]
    atomic_json(work/f'{prefix}-speaker-corrections.json',{'corrections':corrections,
        'policy':f'audio_and_context_keep_{prefix}_when_uncertain' if audio_manifest else f'explicit_context_only_keep_{prefix}_when_uncertain'})
    segments=[];excluded=[];windows=[]
    for row,text in zip(rows,values):
        if row.get('keep_original') or text is None or not any(c.isalnum() for c in text):
            excluded.append({'start':row['start'],'end':row['end'],'reason':'unmeasurable_point_anchor' if row.get('keep_original') else 'not_needing_dubbing'});continue
        segments.append({'start':row['start'],'end':row['end'],'original_text':row['text'],'translated_text':text,'speaker_id':row['speaker'],'source_region_id':row['id']})
        windows.append({'start':row['start'],'end':row['end'],'start_frame':round(row['start']*SR),'end_frame':round(row['end']*SR),'fade_in_frames':0,'fade_out_frames':0,'boundary_policy':authority})
    timeline={'identity':source_identity,'duration':sf.info(clean).duration,'boundary_authority':authority,'turns':[{'start':r['start'],'end':r['end'],'speaker':r['speaker']} for r in rows]}
    result={'segments':segments,'excluded_intervals':excluded,'source_regions':rows,'speaker_timeline':timeline,'reference_windows':windows,'boundary_measurements':audit,
        'boundary_authority':authority,'word_grouping':WORD_GROUPING,'speaker_authority':prefix+('_with_gemini_audio_and_context_corrections' if audio_manifest else '_with_gemini_explicit_context_corrections'),'speaker_corrections':corrections,'source_asr_model':source_model,'source_asr_input_policy':source_input_policy,
        'source_languages':[raw.get('language') or 'unknown'],'original_text_source':source_model,'target_language':cfg['language'],'duration_seconds':timeline['duration'],'time_unit':'seconds',
        'asr_used':True,'llm_timestamps':False,'word_reassignments':reassignments,'request_count':translation_requests,'translation_output':'region_id_to_translation_with_optional_speaker_word_span_or_speaker_parts','usage':usage,'unmatched_segments':[],
        'llm_audio_input':({k:v for k,v in audio_manifest.items() if k!='path'} if audio_manifest else None),
        'transition_policy':'standard_external_fade','reference_tail_policy':{'enabled':False,'method':prefix+'_waveform_edges_before_translation'}}
    atomic_json(output,result);progress(100,f'{label} {len(rows)}개 전사·경계 보정 · {len(segments)}개 번역 완료')
    return result
