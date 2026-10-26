"""One audio request adds English performance cues before text-only translation."""
from pathlib import Path
import base64,hashlib,json,re,time,urllib.request,urllib.error
from core import QWEN,atomic_json
from rich_transcript import spoken
MODEL=QWEN

def lexical(text):return re.sub(r'\s+','',spoken(text))

def validate(pair,original,translated):
    if not isinstance(pair,list) or len(pair)!=2 or not all(isinstance(s,str) for s in pair):return 'invalid_text_pair'
    if lexical(pair[0])!=lexical(original) or lexical(pair[1])!=lexical(translated):return 'spoken_text_or_punctuation_changed'
    tags=[re.findall(r'\[([^\]\n]*)\]',s) for s in pair]
    if tags[0]!=tags[1]:return 'paired_cues_differ'
    for text in pair:
        if text.count('[')!=len(tags[0]) or text.count(']')!=len(tags[0]):return 'malformed_cue_brackets'
        if re.search(r'\[[^\]]*\]\s*$',text):return 'cue_after_final_spoken_word'
    if any(not tag.strip() or not tag.isascii() or not any(c.isalpha() for c in tag) for tag in tags[0]):return 'invalid_or_non_english_cue'
    return None



def normalize_response(value):
    # Both forms carry the same typed entries; never infer missing text or IDs.
    if isinstance(value,list):value={'segments':value}
    if not isinstance(value,dict) or set(value)!={'segments'} or not isinstance(value['segments'],list):raise ValueError('invalid_annotated_response')
    seen=set()
    for item in value['segments']:
        if not isinstance(item,dict) or set(item)!={'id','text'} or type(item['id']) is not int or not isinstance(item['text'],str) or item['id'] in seen:raise ValueError('invalid_or_duplicate_annotated_segment')
        seen.add(item['id'])
    return value


def read_stream(response,folder,validator=None):
    """Consume SSE while the response context closes immediately on validation failure."""
    chunks=[];metadata={};finish=None;event=[];error=None;last_content=time.monotonic()
    def accept(lines):
        nonlocal finish,last_content
        if not lines:return
        text='\n'.join(lines)
        if text=='[DONE]':return
        data=json.loads(text)
        if data.get('error'):
            metadata['provider_error']=data['error']
            detail=data['error']
            if isinstance(detail,dict):detail=detail.get('message') or detail.get('code') or 'unknown'
            raise RuntimeError('Qwen 스트리밍 제공자 오류: '+str(detail)[:500])
        for key in ('id','model','provider','usage'):
            if data.get(key) is not None:metadata[key]=data[key]
        for choice in data.get('choices',[]):
            if choice.get('index',0)!=0:continue
            if choice.get('finish_reason') is not None:finish=choice['finish_reason']
            content=choice.get('delta',{}).get('content')
            if content is None:continue
            if not isinstance(content,str):raise ValueError('review_stream_invalid_content')
            if content.strip():last_content=time.monotonic()
            chunks.append(content);partial.write(content);partial.flush()
            if validator:validator.feed(content)
    try:
        with (folder/'stream-content.txt').open('w',encoding='utf8') as partial:
            for line in response:
                if time.monotonic()-last_content>120:raise RuntimeError('Qwen returned no response content for 120 seconds. Retrying.')
                line=line.decode('utf8') if isinstance(line,bytes) else line
                line=line.rstrip('\r\n')
                if not line:accept(event);event=[]
                elif line.startswith('data:'):event.append(line[5:].lstrip())
            accept(event)
        if finish!='stop':raise RuntimeError('Qwen streaming response is incomplete.')
        if validator:validator.finish()
    except Exception as exc:
        error=str(exc);raise
    finally:
        data={**metadata,'choices':[{'index':0,'finish_reason':finish,'message':{'role':'assistant','content':''.join(chunks)}}]}
        if error:data['stream_error']=error;data['stream_aborted']=True
        atomic_json(folder/'raw.json',data)
    return data


def request(payload,key,folder,ids,progress,*,operation='단서',normalizer=normalize_response,content_parser=None,stream_validator=None):
    started=time.monotonic()
    req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Title':'Fast Dubbing delivery cues'})
    progress(0,f'원문 {operation} · {len(ids)}개 분석 · 응답 대기')
    try:
        with urllib.request.urlopen(req,timeout=120 if payload.get('stream') else 900) as response:
            data=read_stream(response,folder,stream_validator) if payload.get('stream') else json.load(response)
    except urllib.error.HTTPError as error:
        atomic_json(folder/'http-error.json',{'status':error.code,'body':error.read().decode('utf8','replace').replace(key,'[REDACTED]')})
        raise RuntimeError(f'Qwen {operation} HTTP {error.code}') from None
    except (urllib.error.URLError,TimeoutError):raise RuntimeError(f'Qwen {operation} connection interrupted or timed out.') from None
    atomic_json(folder/'raw.json',data)
    if data.get('error'):raise RuntimeError(f'Qwen {operation} provider error.')
    choices=data.get('choices') or []
    if not choices or choices[0].get('finish_reason')!='stop':raise RuntimeError(f'Qwen {operation} response is incomplete.')
    text=choices[0].get('message',{}).get('content')
    if not isinstance(text,str):raise RuntimeError(f'Qwen {operation} response contains no text.')
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Duplicate region ID')
            result[k]=v
        return result
    result=content_parser(text) if content_parser else json.loads(text,object_pairs_hook=unique)
    result=normalizer(result)
    return result,{'requests':1,'elapsed_seconds':time.monotonic()-started,'usage':data.get('usage',{}),
        **{k:data[k] for k in ('id','model','provider') if k in data}}


# Alibaba rejected the valid 9.67MB Base64 recording as empty; the same
# complete audio at 7.73MB succeeded. Keep headroom below the nominal 10MB.
SOURCE_POLICY={'version':26,'model':MODEL,'reasoning':{'enabled':False},'max_tokens':65536,
    'requests':1,'audio':'whole_separated_dialogue_original_clock','base64_limit':8_000_000,
    'lexical_changes':'reject_per_region_keep_uncued_text','speaker_and_time_changes':False,'output':'annotation_only_cues_and_unit_positions_no_source_echo','stage':'before_gemini_translation'}
CUE_ACOUSTIC_RULE='MANDATORY CUE COVERAGE: EVERY output dialogue line and EVERY split part MUST begin with at least one sufficiently detailed English [acting/delivery cue]. No exceptions for short replies, interjections, neutral speech or unchanged parts. A cue on a preceding line or part does not count; each part must carry its own cue. If the emotion or intent is uncertain, describe only the audible manner of speaking rather than omitting the cue or inventing an emotion. For neutral or matter-of-fact speech, describe its actual audible delivery as well; a bare neutral label is insufficient. Check every part for its own leading English cue before returning. '+ 'Describe BOTH communicative intent and audible delivery together in the same adequately detailed cue when both are identifiable. Keep the intent (such as seeking an answer, teasing, reassuring or warning), and independently describe HOW the voice actually sounds: relative loudness, register, softness or firmness, breathiness or strain, pace and pauses as needed. Describe register relative to this speaker, without changing voice identity. A question, accusation or demand can be quiet, gentle and low-voiced; do not turn it sharp, forceful or loud merely because of its meaning or punctuation. Conversely, do not erase its intent just because the voice is soft. Preserve mixed qualities when audible. Do not simplify, summarize or compress the performance into generic labels such as angry shout or harsh tone. Preserve every distinguishable audible feature that matters to reproducing this performance, including simultaneous or changing qualities. Do not omit trembling, tears, broken voice, breath control, strained onset, emphasized words, pacing or intensity changes when they are actually audible. Do not add features merely to fill a checklist, repeat synonyms, explain the plot or speculate. Audio-grounded delivery constrains the intent label; never invent either intent or vocal traits from the script alone.'
CUE_ACOUSTIC_RULE+=' '+"""Do not stop after identifying two or three representative descriptors. There is no fixed descriptor count: preserve the distinct audible qualities needed to reproduce this particular delivery. Listen for qualities that coexist or seem contradictory; a firm request for confirmation can also sound hesitant, restrained or unsure. Firmness describes neither fluency nor certainty, and communicative intent does not replace how the words are delivered. If hesitation is audible, retain its actual form (such as a tentative onset, faltering rhythm, a restart or a meaningful pause) alongside firmness or pressure, rather than letting the dominant intent erase it. For example, [firm, pressing for confirmation] is incomplete IF the audio also contains hesitation; [firm but hesitant, pressing for confirmation, brief faltering pauses] is appropriate ONLY if those pauses are actually heard. The example is not a default cue and must not be copied onto unrelated lines. Before finalizing each cue, check whether the dominant labels have hidden another audible quality that changes the acting; include that quality without padding, synonym lists or invented traits. Use inline cues when the delivery changes within the line."""
CUE_ACOUSTIC_RULE+=' '+"""REQUIRED CONTENT OF EVERY CUE: Describe the AUDIBLE VOCAL EXECUTION FIRST. Each English bracket cue MUST say how the voice actually sounds or how the words are physically delivered, not merely what the speaker wants or feels. An intention-only or emotion-only cue is INVALID, even if its interpretation is correct. [panicked demand for information], [firm, accusatory statement] and [pressing for confirmation] are INCOMPLETE because they omit concrete audible delivery. Keep the intent when identifiable, but include the actual speaking manner and every other distinguishable delivery feature that matters; do not stop after the first feature. A valid cue might describe a soft low voice and measured pacing alongside pressure, or a strained raised voice and rushed uneven rhythm alongside panic, ONLY when heard. Do not equate panic with shouting, accusation with harshness, or firmness with fluent confidence. If an intent is uncertain, omit that intent and describe the audible voice/delivery instead. Stable neutral speech still requires its actual delivery, not just [neutral] or [matter-of-fact]. Before returning each part, silently check whether a voice actor could tell HOW to say these words from its cue; if it only says WHY or with WHAT EMOTION, replace it with an audio-grounded delivery direction before emitting. Keep one combined bracket cue with optional later inline cues; do not output extra analysis fields or explanations. This is a semantic requirement, not a minimum word count or a vocabulary checklist."""
CUE_ACOUSTIC_RULE+=' '+"""Be greedy and uncapped in capturing distinguishable audible acting and delivery details: do not impose an arbitrary descriptor-count or cue-length cap. Retain all relevant details actually supported by the audio, including subtle or coexisting qualities, without padding, repetition or invented traits."""
# Refreshing existing dialogue uses the same annotations-only contract. Unit IDs
# locate cue insertions; the model never needs to echo the locked source text.
SOURCE_PROMPT='''Listen to the recording and annotate EVERY supplied fixed line. Return ONLY English audible-delivery cues and their positions, NEVER the source dialogue, translation or explanations. Each line supplies ordered units; their zero-based IDs are insertion anchors, not timestamps. Preserve line IDs; do not split, merge or change source words, speakers or times. Return compact JSON: {"segments":[{"id":0,"c":"soft low voice, measured pace","m":[{"i":3,"c":"hesitant onset"}]}]}. c is the mandatory leading delivery cue, WITHOUT brackets. Optional m adds mid-line cues immediately BEFORE the indicated unit; omit m when unnecessary. Include all distinguishable audible acting details, with concrete vocal execution first, and identifiable intent as well. Do not return text or repeat any supplied units. '''+CUE_ACOUSTIC_RULE


def cue_units(text):
    # Whole Latin words/numbers remain indivisible; Japanese ASR also uses
    # character-sized word units. Offsets refer to the original immutable string.
    return [(m.start(),m.group()) for m in re.finditer(r"[A-Za-z0-9]+(?:['’.,][A-Za-z0-9]+)*|[^\s]",text)]


def expand_cue_response(value,rows):
    if isinstance(value,list):value={'segments':value}
    if not isinstance(value,dict) or not isinstance(value.get('segments'),list):raise ValueError('invalid_cue_response')
    by_id={r['id']:r for r in rows};seen=set();output=[]
    def cue(raw):
        if isinstance(raw,list) and raw and all(isinstance(x,str) for x in raw):raw=''.join(raw)
        if not isinstance(raw,str):raise ValueError('invalid_delivery_cue')
        raw=raw.strip()
        if raw.startswith('[') and raw.endswith(']'):raw=raw[1:-1].strip()
        if not raw or not raw.isascii() or not any(c.isalpha() for c in raw) or any(c in '[]\r\n' for c in raw):raise ValueError('invalid_english_cue')
        return '['+raw+']'
    for entry in value['segments']:
        if not isinstance(entry,dict):raise ValueError('invalid_cue_entry')
        identifier=entry.get('id')
        if isinstance(identifier,str) and identifier.isdigit():identifier=int(identifier)
        if type(identifier) is not int or identifier not in by_id or identifier in seen:raise ValueError('unknown_or_duplicate_cue_id')
        seen.add(identifier)
        if 'text' in entry:  # Backward compatibility for old cached responses.
            output.append({'id':identifier,'text':entry['text']});continue
        text=by_id[identifier]['text'];units=cue_units(text);insertions={0:cue(entry.get('c'))+' '}
        inline=entry.get('m',[])
        if not isinstance(inline,list):raise ValueError('invalid_inline_cues')
        for item in inline:
            if not isinstance(item,dict):raise ValueError('invalid_inline_cue')
            i=item.get('i')
            if isinstance(i,str) and i.isdigit():i=int(i)
            if type(i) is not int or not 0<=i<len(units):raise ValueError('unknown_cue_unit')
            at=units[i][0]
            if at in insertions:raise ValueError('duplicate_cue_position')
            insertions[at]=cue(item.get('c'))+' '
        for at,insert in sorted(insertions.items(),reverse=True):text=text[:at]+insert+text[at:]
        output.append({'id':identifier,'text':text})
    if seen!=set(by_id):raise ValueError('missing_cue_regions')
    return {'segments':output}


def annotate_source(clean,rows,work,cfg,progress=lambda *a:None,*,prompt=None):
    from qwen_audio_pipeline import encode_audio,digest
    prompt=SOURCE_PROMPT if prompt is None else prompt
    if not rows:return rows,{'requests':0,'policy':SOURCE_POLICY,'reviews':[]}
    folder=Path(work)/'qwen-source-cues';folder.mkdir(parents=True,exist_ok=True)
    (folder/'audio').mkdir(exist_ok=True)
    report={'requests':0,'policy':SOURCE_POLICY,'reviews':[]}
    try:
        audio,manifest=encode_audio(Path(clean),folder/'audio',digest(clean),lambda *a:None,base64_limit=SOURCE_POLICY['base64_limit'])
        inputs=[{'id':r['id'],'start':r['start'],'end':r['end'],'units':[[i,t] for i,(_,t) in enumerate(cue_units(r['text']))],
            **({'source_text':r['source_text']} if 'source_text' in r else {})} for r in rows]
        identity={'policy':SOURCE_POLICY,'prompt':prompt,'regions':inputs,'audio':manifest,'locked_text_hashes':{str(r['id']):hashlib.sha256(r['text'].encode()).hexdigest() for r in rows}}
        run=folder/hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest();run.mkdir(exist_ok=True)
        atomic_json(run/'input.json',identity);cache=run/'result.json'
        saved=json.loads(cache.read_text('utf8')) if cache.exists() else {}
        report['audio']=manifest
        if saved.get('signature')==identity:
            values=saved['values'];report.update(saved['receipt'],requests=0);progress(100,'Qwen 원문 단서 캐시 재사용')
        else:
            schema={'type':'object','properties':{'segments':{'type':'array','items':{'type':'object','properties':{'id':{'type':'integer'},'c':{'type':'string'},'m':{'type':'array','items':{'type':'object','properties':{'i':{'type':'integer'},'c':{'type':'string'}},'required':['i','c'],'additionalProperties':False}}},'required':['id','c'],'additionalProperties':False}}},'required':['segments'],'additionalProperties':False}
            payload={'model':MODEL,'max_tokens':SOURCE_POLICY['max_tokens'],'reasoning':{'enabled':False},'stream':False,
                'response_format':{'type':'json_schema','json_schema':{'name':'original_delivery_cues','strict':True,'schema':schema}},
                'messages':[{'role':'system','content':prompt},{'role':'user','content':[
                    {'type':'text','text':'Add bracketed English delivery cues to these fixed lines:\n'+json.dumps(inputs,ensure_ascii=False)},
                    {'type':'input_audio','input_audio':{'format':'mp3','data':base64.b64encode(audio.read_bytes()).decode()}},
                    {'type':'text','text':'Return every line ID with c (audible delivery first) and optional m unit positions. Do not return any source dialogue or repeat supplied units.'}]}]}
            report['requests']=1
            from qwen_performance_turns import decode
            values,response=request(payload,cfg['key'],run,[str(r['id']) for r in rows],progress,
                content_parser=decode,normalizer=lambda value:expand_cue_response(value,rows))
            entries=values.get('segments')
            if not isinstance(entries,list):raise ValueError('missing_annotated_segments')
            converted={}
            for item in entries:
                if not isinstance(item,dict) or set(item)!={'id','text'} or type(item['id']) is not int or str(item['id']) in converted:raise ValueError('invalid_annotated_segment')
                converted[str(item['id'])]=item['text']
            values=converted
            report.update(response);atomic_json(cache,{'signature':identity,'values':values,'receipt':response})
        result=[]
        for row in rows:
            value=values.get(str(row['id']));error=validate([value,value],row['text'],row['text'])
            if error:
                report['reviews'].append({'id':row['id'],'action':'keep_uncued_text','reason':error});result.append(dict(row))
            else:result.append({**row,'text':value,'uncued_text':row['text']})
    except (RuntimeError,ValueError,KeyError,TypeError) as exc:
        report['error']=str(exc);report['reviews']=[{'id':r['id'],'action':'keep_uncued_text','reason':'cue_stage_failed'} for r in rows];result=rows
    report['annotated_regions']=sum(r['text']!=old['text'] for r,old in zip(result,rows))
    atomic_json(folder/'audit.json',report)
    progress(100,f'Qwen 원문 단서 완료 · {report["annotated_regions"]}개 반영 · 기존 전사 유지 {len(report["reviews"])}개')
    return result,report
