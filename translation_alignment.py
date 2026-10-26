"""Match translated dialogue to original-language ASR by multilingual semantics.

The model sees translations and ASR text, never Gemini originals/timestamps.
The recording-wide monotonic path prevents source-word reuse and reordering.
"""
from pathlib import Path
import hashlib,json,re
import numpy as np
from core import ROOT,atomic_json

VERSION=8
MODEL='sentence-transformers/LaBSE'
REVISION='836121a0533e5664b21c7aacc5d22951f2b8b25b'
MAX_UNITS=12
MIN_SIMILARITY=.55


def clauses(text):
    """Retain the actual translated wording while proposing phrase cuts."""
    return [m.group().strip() for m in re.finditer(
        r'.+?(?:[.!?。！？]+(?=\s|$)|…+(?=\s|$)|\n+|$)',text,re.S) if m.group().strip()]


def partition_clauses(parts,scores,turns,group_scores=None):
    """Assign contiguous translated phrases to contiguous ASR speaker turns.

    Scores are learned cross-language similarities. No voice names, phrases,
    video identities, or absolute times participate in choosing the cuts.
    """
    from source_alignment import match_key
    n=len(parts);k=len(turns)
    if n<k:return None
    weights=np.asarray([max(1,len(match_key(p))) for p in parts])
    values=scores*weights[:,None]
    states={0:(0.,[])}
    for turn in range(k):
        updated={}
        for start,(score,path) in states.items():
            for end in range(start+1,n-(k-turn-1)+1):
                value=score+(float(weights[start:end].sum())*group_scores[start,end,turn]
                    if group_scores is not None else float(values[start:end,turn].sum()))
                if end not in updated or value>updated[end][0]:
                    updated[end]=(value,path+[(start,end)])
        states=updated
    return states[n][1] if n in states else None


def embeddings(texts,progress=lambda *args:None):
    from embedding_client import encode
    return encode(texts,progress)


def speaker(word):
    return word.get('speaker_id',word.get('speaker_label',word.get('speaker')))


def source_units(words,ignore_gaps=False):
    from source_alignment import match_key
    indices=[i for i,w in enumerate(words) if match_key(w['word'])]
    if not indices:return []
    result=[];first=indices[0];previous=None
    for index in indices:
        if previous is not None:
            between=''.join(w['word'] for w in words[previous:index])
            changed=speaker(words[index]) is not None and speaker(words[previous]) is not None and speaker(words[index])!=speaker(words[previous])
            if changed or (not ignore_gaps and words[index]['start']-words[previous]['end']>.4) or re.search(r'[。！？!?、,;；\n]',between):
                result.append((first,previous));first=index
        previous=index
    result.append((first,indices[-1]))
    return result


def ordered(candidates,beam=256,min_similarity=MIN_SIMILARITY):
    states={-1:(0.,[])}
    for options in candidates:
        # Missing/uncertain dialogue may remain unmatched; do not force a line
        # onto unrelated source speech just to obtain a zero-warning counter.
        updated={end:(score,path+[None]) for end,(score,path) in states.items()}
        for option in options:
            preceding=[(score,path) for end,(score,path) in states.items() if end<option['first']]
            if not preceding:continue
            score,path=max(preceding,key=lambda item:item[0])
            value=score+option['score']-min_similarity
            end=option['last']
            if end not in updated or value>updated[end][0]:updated[end]=(value,path+[option])
        states=dict(sorted(updated.items(),key=lambda item:item[1][0],reverse=True)[:beam])
    return max(states.values(),key=lambda item:item[0])[1] if states else []


def locate(segments,words,duration,progress=lambda *args:None,folder=None,encoder=embeddings,min_similarity=MIN_SIMILARITY,allow_gap_rescue=True,sentence_boundaries=False,min_source_text_ratio=0.):
    texts=[s['translated_text'] for s in segments]
    missing=lambda:{'matched':False,'coverage':0.,'words':[],'word_indices':[]}
    if not texts:return []
    units=[(a,b) for a,b in source_units(words,ignore_gaps=sentence_boundaries) if
        any(w['end']>w['start'] for w in words[a:b+1])];spans=[]
    for left,(first,_) in enumerate(units):
        for right in range(left,min(len(units),left+MAX_UNITS)):
            a,b=units[right]
            # Diarization labels form candidate boundaries, not a veto: one
            # translated line can cover an exchange, and ASR can split a voice.
            text=''.join(w['word'] for w in words[first:b+1])
            if len(text)>1000:break
            positive=[w for w in words[first:b+1] if w['end']>w['start']]
            if not positive:continue
            spans.append({'first':first,'last':b,'text':text,'unit_start':left,'unit_end':right,
                'start':positive[0]['start'],'end':min(duration,positive[-1]['end'])})
    if not spans:return [missing() for _ in segments]
    source_texts=list(dict.fromkeys(span['text'] for span in spans));lookup={t:i for i,t in enumerate(source_texts)}
    phrases=[clauses(text) or [text] for text in texts]
    extra=list(dict.fromkeys(' '.join(group[a:b]) for group in phrases
        for a in range(len(group)) for b in range(a+1,len(group)+1)))
    queries=texts+extra;query_lookup={text:len(texts)+j for j,text in enumerate(extra)}
    all_texts=queries+source_texts
    identity=hashlib.sha256(json.dumps([VERSION,MODEL,REVISION,all_texts],ensure_ascii=False).encode()).hexdigest()
    cache=Path(folder)/'translation-embeddings.npz' if folder is not None else None
    if cache is not None and cache.exists():
        with np.load(cache,allow_pickle=False) as data:
            vectors=data['vectors'] if str(data['identity'])==identity else None
    else:vectors=None
    if vectors is None:
        vectors=encoder(all_texts,progress)
        if cache is not None:
            cache.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(cache,identity=np.array(identity),vectors=vectors)
    query_scores=vectors[:len(queries)]@vectors[len(queries):].T
    similarities=query_scores[:len(texts)]
    unit_texts=[''.join(w['word'] for w in words[a:b+1]) for a,b in units]
    unit_scores=similarities[:,[lookup[text] for text in unit_texts]]
    candidates=[];candidate_reasons=[]
    contained={}
    from source_alignment import match_key
    for i in range(len(texts)):
        row=[];speaker_cut_rejected=False
        # Retrieve a broad pool; global order, not a timestamp hint, resolves
        # repeated short lines and competing translations.
        top=np.argsort(similarities[i])[-64:]
        valid={source_texts[j]:float(similarities[i,j]) for j in top if similarities[i,j]>=min_similarity}
        for span in spans:
            if span['text'] in valid:
                # Reject incomplete candidates before choosing the best path.
                # A short but high-similarity fragment must not mask a slightly
                # lower-scoring complete utterance which passes verification.
                if len(match_key(span['text']))/max(1,len(match_key(texts[i])))<min_source_text_ratio:continue
                a,b=span['unit_start'],span['unit_end'];score=valid[span['text']]
                labels=[speaker(words[ua]) for ua,ub in units[a:b+1]]
                turns=1+sum(x is not None and y is not None and x!=y for x,y in zip(labels,labels[1:]))
                if turns>len(phrases[i]):speaker_cut_rejected=True;continue
                # A whole-sentence embedding may score a partial translation
                # highly. Require the substantial translated clauses to be
                # supported somewhere inside the proposed ASR span as well.
                substantial=[p for p in phrases[i] if len(match_key(p))>=4]
                if len(substantial)>1:
                    bounds=(a,b)
                    if bounds not in contained:
                        contained[bounds]=list({lookup[s['text']] for s in spans
                            if s['unit_start']>=a and s['unit_end']<=b})
                    cols=contained[bounds]
                    coverage=float(np.mean([np.max(query_scores[query_lookup[p],cols]) for p in substantial]))
                    score=.6*score+.4*coverage
                # Discourage borrowing the previous/next translation's clause
                # for a tiny whole-sentence similarity gain. All comparisons
                # remain learned cross-language similarities, not phrase rules.
                if i:score-=.5*max(0,float(unit_scores[i-1,a]-unit_scores[i,a]))
                if i+1<len(texts):score-=.5*max(0,float(unit_scores[i+1,b]-unit_scores[i,b]))
                score-=.02*(b-a)
                row.append({**span,'score':score,'semantic_similarity':valid[span['text']]})
        candidates.append(row)
        candidate_reasons.append('speaker_boundary_without_safe_translation_cut' if speaker_cut_rejected and not row else 'translation_not_matched_to_scribe')
    path=ordered(candidates,min_similarity=min_similarity);result=[]
    for i,choice in enumerate(path):
        if choice is None:result.append({**missing(),'reason':candidate_reasons[i]});continue
        a,b=choice['first'],choice['last']
        location={'matched':True,'start':choice['start'],'end':choice['end'],
            'coverage':choice['semantic_similarity'],'semantic_similarity':choice['semantic_similarity'],
            'alignment_model':MODEL,'recognized_text':choice['text'],
            'words':words[a:b+1],'word_indices':list(range(a,b+1))}
        cols=[lookup[s['text']] for s in spans if s['first']>=a and s['last']<=b]
        location['clause_support']=[float(np.max(query_scores[query_lookup[p],cols]))
            for p in phrases[i] if len(match_key(p))>=4]
        turns=[]
        for ua,ub in units[choice['unit_start']:choice['unit_end']+1]:
            label=speaker(words[ua])
            if turns and (label is None or turns[-1]['speaker']==label):turns[-1]['last']=ub
            else:turns.append({'first':ua,'last':ub,'speaker':label})
        if len(turns)>1:
            parts=phrases[i]
            columns=[lookup[''.join(w['word'] for w in words[t['first']:t['last']+1])] for t in turns]
            scores=query_scores[[query_lookup[p] for p in parts]][:,columns]
            group_scores={(a,b,t):float(query_scores[query_lookup[' '.join(parts[a:b])],column])
                for a in range(len(parts)) for b in range(a+1,len(parts)+1)
                for t,column in enumerate(columns)}
            cuts=partition_clauses(parts,scores,turns,group_scores=group_scores)
            if cuts is None:
                result.append({**missing(),'reason':'speaker_boundary_without_safe_translation_cut'});continue
            separated=[]
            for t,(turn,(left,right)) in enumerate(zip(turns,cuts)):
                first,last=turn['first'],turn['last'];ws=words[first:last+1]
                positive=[w for w in ws if w['end']>w['start']]
                separated.append({**location,'translated_text':' '.join(parts[left:right]),
                    'start':positive[0]['start'],'end':min(duration,positive[-1]['end']),
                    'recognized_text':''.join(w['word'] for w in ws),'words':ws,
                    'word_indices':list(range(first,last+1)),'asr_speaker_label':turn['speaker'],
                    'coverage':group_scores[left,right,t],
                    'semantic_similarity':group_scores[left,right,t],
                    'speaker_boundary_split':True})
            if any(p['coverage']<min_similarity for p in separated):
                result.append({**missing(),'reason':'speaker_partition_not_confident'});continue
            location['parts']=separated
        else:location['asr_speaker_label']=turns[0]['speaker'] if turns else None
        result.append(location)
    if allow_gap_rescue and min_similarity==MIN_SIMILARITY:
        known={t:vectors[j] for j,t in enumerate(all_texts)}
        def reuse_encoder(wanted,callback):
            absent=list(dict.fromkeys(t for t in wanted if t not in known))
            if absent:known.update(zip(absent,encoder(absent,callback)))
            return np.asarray([known[t] for t in wanted])
        cursor=0
        while cursor<len(result):
            if result[cursor]['matched']:cursor+=1;continue
            stop=cursor+1
            while stop<len(result) and not result[stop]['matched']:stop+=1
            lo=result[cursor-1]['word_indices'][-1]+1 if cursor else 0
            hi=result[stop]['word_indices'][0] if stop<len(result) else len(words)
            if hi>lo:
                retry=locate(segments[cursor:stop],words[lo:hi],duration,progress,
                    encoder=reuse_encoder,min_similarity=.30,allow_gap_rescue=False)
                for offset,location in enumerate(retry):
                    if not location['matched']:continue
                    # A single weak match still needs an independent ASR check.
                    if stop-cursor==1 and location['coverage']<MIN_SIMILARITY:continue
                    for part in [location]+location.get('parts',[]):
                        part['word_indices']=[j+lo for j in part['word_indices']]
                        part['alignment_repair']='ordered_unclaimed_asr_gap'
                    result[cursor+offset]=location
            cursor=stop
    if folder is not None:
        atomic_json(Path(folder)/'translation-matches.json',{'model':MODEL,'version':VERSION,
            'method':'cross_language_translation_to_asr_monotonic','segments':result,
            'gemini_originals_used':False,'gemini_times_used':False})
    return result
