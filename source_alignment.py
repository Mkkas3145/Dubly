"""Match Gemini dialogue to Scribe's whole-recording word sequence.

There is no local ASR, forced alignment, or second-model fallback here.
Gemini timestamps are deliberately absent from the matching decision.
"""
from __future__ import annotations
import difflib,hashlib,json,unicodedata
from core import atomic_json,config
from scribe import transcribe_words,MODEL

VERSION=9


def measure_collapsed_words(source,words):
    """Restore zero-duration lexical ASR anchors from local separated audio.

    This does not add guessed text or copy an LLM time. Punctuation remains
    zero-duration; adjacent timed words bound the search for a measured island.
    """
    import numpy as np
    import soundfile as sf
    from scipy.ndimage import binary_closing
    from reference_edges import spans
    from translation_alignment import speaker
    result=[dict(w) for w in words]
    with sf.SoundFile(source) as audio:
        sr=audio.samplerate;duration=len(audio)/sr;hop=max(1,round(sr*.01))
        for i,w in enumerate(words):
            if w['end']!=w['start'] or not match_key(w['word']) or speaker(w) is None:continue
            if w.get('probability',1.)<.15:continue
            anchor=w['start']
            lower=max(0.,anchor-.5,next((p['end'] for p in reversed(words[:i]) if p['end']>p['start'] and match_key(p['word'])),0.))
            upper=min(duration,anchor+.5,next((p['start'] for p in words[i+1:] if p['end']>p['start'] and match_key(p['word'])),duration))
            if upper-lower<.04:continue
            audio.seek(round(lower*sr));x=audio.read(round((upper-lower)*sr),dtype='float32',always_2d=True)
            x=np.pad(x,((0,(-len(x))%hop),(0,0)))
            if not len(x):continue
            energy=np.sqrt(np.mean(x.reshape(-1,hop,x.shape[1])**2,axis=1)).max(axis=1)
            floor=float(np.percentile(energy,20));peak=float(energy.max())
            if peak<max(1e-5,floor*4):continue
            # Local floor is used to verify contrast, not as a loud cutoff
            # inside a quiet consonant or between the syllables of one word.
            mask=binary_closing(energy>max(1e-6,peak*.025,floor*.5),structure=np.ones(7),border_value=0)
            islands=[(lower+a*hop/sr,min(upper,lower+b*hop/sr)) for a,b in spans(mask) if b-a>=3]
            if not islands:continue
            start,end=min(islands,key=lambda p:max(p[0]-anchor,anchor-p[1],0.))
            if max(start-anchor,anchor-end,0.)>.2:continue
            result[i].update(start=start,end=end,zero_duration_anchor=anchor,
                             anchor_recovery='local_voice_at_zero_duration_asr_word')
    return result


def restore_sentence_openings(locations,words,segments=None,encoder=None):
    """Keep unclaimed openings of an already matched ASR sentence.

    Commas and hesitation gaps are retrieval units, not sentence boundaries.
    Never cross another owner's words, a sentence ending, or a known voice
    change. Unknown voice identity is insufficient to annex earlier speech.
    """
    import re
    from translation_alignment import speaker
    parts=[p for location in locations if location.get('matched')
           for p in location.get('parts',[location])]
    owner={id(p):i for i,location in enumerate(locations) if location.get('matched')
           for p in location.get('parts',[location])}
    proposals=[]
    claimed={j for p in parts for j in p['word_indices']}
    for part in parts:
        indices=part['word_indices'];first,last=indices[0],indices[-1]
        label=speaker(words[first])
        if label is None:continue
        start=first
        for j in range(first-1,-1,-1):
            word=words[j]
            if j in claimed or re.search(r'[。！？!?;；\n]|\.(?=\s|$)',word['word']):break
            if match_key(word['word']):
                if speaker(word)!=label or words[first]['start']-word['start']>3.5:break
                # A long separate utterance is not an opening hesitation.
                if words[start]['start']-word['end']>1.2:break
            start=j
        while start<first and not match_key(words[start]['word']):start+=1
        positive=[w for w in words[start:last+1] if w['end']>w['start'] and match_key(w['word'])]
        if start==first or not positive:continue
        proposals.append((part,start,first,last,positive))
    if segments is not None and proposals:
        from translation_alignment import embeddings
        encode=encoder or embeddings
        texts=list(dict.fromkeys([s['translated_text'] for s in segments]+[
            ''.join(w['word'] for w in words[a:b+1])
            for part,start,first,last,positive in proposals
            for a,b in [(start,first-1),(first,last),(start,last)]]))
        vectors=dict(zip(texts,encode(texts)))
    for part,start,first,last,positive in proposals:
        if segments is not None:
            index=owner[id(part)];target=vectors[segments[index]['translated_text']]
            prefix=''.join(w['word'] for w in words[start:first])
            if len(match_key(prefix))>3:
                full=''.join(w['word'] for w in words[start:last+1]);body=''.join(w['word'] for w in words[first:last+1])
                if float(target@vectors[full])<float(target@vectors[body])-.025:continue
                others=[vectors[segments[j]['translated_text']] for j in [index-1,index+1] if 0<=j<len(segments)]
                if any(float(v@vectors[prefix])>float(target@vectors[prefix])+.05 for v in others):continue
        part.update(start=positive[0]['start'],words=words[start:last+1],
                    word_indices=list(range(start,last+1)),
                    recognized_text=''.join(w['word'] for w in words[start:last+1]),
                    opening_recovery='unclaimed_same_speaker_sentence_opening')
        claimed.update(range(start,first))
    return locations

def match_key(text):
    text=unicodedata.normalize('NFKC',text)
    text=''.join(chr(ord(c)-0x60) if '\u30a1'<=c<='\u30f6' else c for c in text)
    return ''.join(c.casefold() for c in text if c.isalnum())

def compact_match(target,recognized):
    """Align one complete target locally, penalizing unrelated intervening text.

    Recording-wide matches supply order/context, but cannot supply endpoints:
    repeated lines may otherwise donate different characters to the same target.
    Time gaps are not penalized, so genuine long pauses remain possible.
    """
    n,m=len(target),len(recognized)
    scores=[[0]*(m+1) for _ in range(n+1)]
    moves=[bytearray(m+1) for _ in range(n+1)]
    for i in range(1,n+1):scores[i][0]=-i;moves[i][0]=1
    for i in range(1,n+1):
        for j in range(1,m+1):
            options=(scores[i-1][j-1]+(2 if target[i-1]==recognized[j-1] else -1),
                     scores[i-1][j]-1,scores[i][j-1]-1)
            move=max(range(3),key=options.__getitem__)
            scores[i][j]=options[move];moves[i][j]=move
    best=max(scores[n]);candidates=[]
    for end,score in enumerate(scores[n]):
        if score!=best:continue
        i,j=n,end;hits=[]
        while i:
            move=moves[i][j]
            if move==0:
                if target[i-1]==recognized[j-1]:hits.append((i-1,j-1))
                i-=1;j-=1
            elif move==1:i-=1
            else:j-=1
        if hits:
            hits.reverse()
            candidates.append((len(hits),-(hits[-1][1]-hits[0][1]+1),-hits[0][1],hits))
    return max(candidates,key=lambda c:c[:3])[-1] if candidates else []

def reclaim_isolated_prefixes(targets,asr,locations):
    """Return an isolated opening word to an omitted earlier complete line.

    Repeated opening text can attach a preceding short utterance to a later
    long sentence. Reassign only an exact whole line separated from the
    sentence body by a timestamp gap; never split continuous speech.
    """
    for index,location in enumerate(locations):
        if not location['matched']:continue
        previous=next((i for i in range(index-1,-1,-1) if locations[i]['matched']),-1)
        missing=[i for i in range(previous+1,index) if not locations[i]['matched']]
        if not missing:continue
        indices=location['word_indices']
        content=[i for i in indices if match_key(asr[i]['word']) and asr[i]['end']>asr[i]['start']]
        if len(content)<2:continue
        for k,(left,right) in enumerate(zip(content,content[1:])):
            if asr[right]['start']-asr[left]['end']<.6:continue
            head=content[:k+1];tail=content[k+1:]
            prefix=''.join(match_key(asr[i]['word']) for i in head)
            candidate=next((i for i in missing if targets[i]==prefix),None)
            if candidate is None or not targets[index].startswith(prefix):continue
            remainder=targets[index][len(prefix):]
            recognized=''.join(match_key(asr[i]['word']) for i in tail)
            hits=compact_match(remainder,recognized)
            if not remainder or len(hits)/max(1,len(targets[index]))<.6:continue
            if previous>=0 and asr[head[0]]['start']<locations[previous]['end']:continue
            first,last=head[0],head[-1]
            locations[candidate]={'start':asr[first]['start'],'end':asr[last]['end'],
                'coverage':1.,'matched':True,'words':asr[first:last+1],
                'word_indices':list(range(first,last+1)),
                'alignment_repair':'isolated_prefix_reassigned_from_following_line'}
            first,last=tail[0],indices[-1]
            location.update(start=asr[first]['start'],words=asr[first:last+1],
                word_indices=list(range(first,last+1)),coverage=len(hits)/len(targets[index]),
                alignment_repair='isolated_prefix_returned_to_preceding_line')
            break
    return locations


def locate_transcript(segments,asr,duration):
    targets=[match_key(s['original_text']) for s in segments]
    source=''.join(targets);recognized='';owners=[]
    for index,word in enumerate(asr):
        part=match_key(word['word']);recognized+=part;owners.extend([index]*len(part))
    positions={}
    for block in difflib.SequenceMatcher(None,source,recognized,autojunk=False).get_matching_blocks():
        for offset in range(block.size):positions[block.a+offset]=block.b+offset
    coarse=[];cursor=0
    for target in targets:
        coarse.append([positions[cursor+i] for i in range(len(target)) if cursor+i in positions])
        cursor+=len(target)
    result=[]
    for index,target in enumerate(targets):
        anchors=coarse[index];hits=[]
        if anchors:
            previous=next((max(c)+1 for c in reversed(coarse[:index]) if c),0)
            following=next((min(c) for c in coarse[index+1:] if c),len(recognized))
            padding=max(8,len(target))
            lo=max(previous,min(anchors)-padding)
            hi=min(following,max(anchors)+1+padding)
            hits=[(i,j+lo) for i,j in compact_match(target,recognized[lo:hi])]
        coverage=len(hits)/max(1,len(target))
        valid=bool(hits) and coverage>=.6 and (len(target)>=4 or len(hits)==len(target))
        # A high character recall alone must not bridge unrelated ASR sentences.
        if valid:
            span=hits[-1][1]-hits[0][1]+1
            valid=span-len(hits)<=max(3,len(target)//2)
        if valid:
            first,last=owners[hits[0][1]],owners[hits[-1][1]]
            words=asr[first:last+1];positive=[w for w in words if w['end']>w['start']]
            if positive:
                result.append({'start':positive[0]['start'],'end':min(duration,positive[-1]['end']),
                    'coverage':coverage,'matched':True,'words':words,'word_indices':list(range(first,last+1))})
            else:result.append({'coverage':coverage,'matched':False,'words':[],'word_indices':[]})
        else:result.append({'coverage':coverage,'matched':False,'words':[],'word_indices':[]})
    return reclaim_isolated_prefixes(targets,asr,result)

def source_evidence(source,segments,folder,language,progress=lambda *args:None,key=None,transcription=None):
    folder.mkdir(parents=True,exist_ok=True)
    source_cache=source.parent/'scribe-source-v1'
    if transcription is None:
        transcription=transcribe_words(source,source_cache,key or config()['key'],language,lambda p,d:progress(p*.65,d))
    from semantic_alignment import locate,MODEL as SIMILARITY_MODEL,VERSION as SEMANTIC_VERSION,REVISION
    translation_only=all('original_text' not in s for s in segments)
    if translation_only:
        from translation_alignment import locate,MODEL as SIMILARITY_MODEL,VERSION as SEMANTIC_VERSION,REVISION
    identity=hashlib.sha256(json.dumps([VERSION,SEMANTIC_VERSION,REVISION,transcription['signature'],segments],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    cache=folder/'scribe-evidence.json'
    if cache.exists():
        old=json.loads(cache.read_text(encoding='utf-8'))
        if old.get('identity')==identity:return old
    if translation_only:
        transcription=dict(transcription,words=measure_collapsed_words(source,transcription['words']))
    located=locate(segments,transcription['words'],transcription['duration'],progress,
        **({'folder':folder} if translation_only else {}))
    if translation_only:restore_sentence_openings(located,transcription['words'],segments)
    rows=[];unmatched=[];expanded=[]
    for i,(segment,location) in enumerate(zip(segments,located)):
        for part in location.get('parts',[location]):expanded.append((i,segment,part))
    for i,segment,location in expanded:
        if not location['matched']:
            unmatched.append({'source_index':i,'original_text':segment.get('original_text',''),
                'translated_text':segment['translated_text'],'reason':location.get('reason','translation_not_matched_to_scribe' if translation_only else 'source_text_not_matched_to_scribe')})
            continue
        rows.append({'index':len(rows),'source_index':i,'start':location['start'],'end':location['end'],
            'located_start':location['start'],'located_end':location['end'],
            'words':[{**w,'timing_source':'scribe_translation_match'} for w in location['words']] if translation_only else location['words'],
            'word_indices':location['word_indices'],'confidence':location['coverage'],
            'text_match_coverage':location['coverage'],'backend':MODEL,
            'semantic_similarity':location.get('semantic_similarity'),
            'recognized_text':location.get('recognized_text',''.join(w['word'] for w in location['words'])),
            **({'translated_text':location.get('translated_text',segment['translated_text']),
                'asr_speaker_label':location.get('asr_speaker_label'),
                'speaker_boundary_split':location.get('speaker_boundary_split',False)} if translation_only else {}),
            **({'source_word_floor':min(location['start'],max((w['end'] for w in transcription['words'][:location['word_indices'][0]]
                    if w['end']>w['start'] and match_key(w['word'])),default=0.)),
                'source_word_ceiling':max(location['end'],min((w['start'] for w in transcription['words'][location['word_indices'][-1]+1:]
                    if w['end']>w['start'] and match_key(w['word'])),default=transcription['duration']))}
                if translation_only else {}),
            **({'alignment_repair':location['alignment_repair']} if 'alignment_repair' in location else {})})
        if location.get('opening_recovery'):rows[-1]['opening_recovery']=location['opening_recovery']
    result={'identity':identity,'model':MODEL,'source_sha256':transcription['signature']['source_sha256'],
        'similarity_model':SIMILARITY_MODEL,'alignment_policy':'translation_to_asr_monotonic' if translation_only else 'monotonic_unique_asr_words',
        'segments':rows,'asr':transcription['words'],'audio_events':transcription['audio_events'],
        'unmatched_segments':unmatched,'usage':transcription['usage']}
    atomic_json(cache,result);progress(100,'Scribe 단어 시간과 원문 연결 완료')
    return result
