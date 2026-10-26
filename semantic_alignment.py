"""Multilingual semantic retrieval followed by monotonic, non-reusing alignment.

Only ASR supplies timestamps. Gemini supplies authoritative dialogue text.
No video identities, dialogue dictionaries or LLM timing hints are used.
"""
from pathlib import Path
import difflib
import re
import numpy as np

MODEL='intfloat/multilingual-e5-small'
REVISION='614241f622f53c4eeff9890bdc4f31cfecc418b3'
VERSION=2


def embeddings(texts,progress=lambda *args:None):
    import torch
    from transformers import AutoModel,AutoTokenizer
    path=Path(__file__).parent/'models'/'multilingual-e5-small'
    location=str(path) if (path/'model.safetensors').is_file() else MODEL
    options={} if location==str(path) else {'revision':REVISION}
    tokenizer=AutoTokenizer.from_pretrained(location,**options)
    model=AutoModel.from_pretrained(location,**options).eval()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device);result=[]
    try:
        with torch.inference_mode():
            for start in range(0,len(texts),32):
                # E5's documented symmetric similarity task uses query: on both sides.
                batch=tokenizer(['query: '+s for s in texts[start:start+32]],
                    padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
                hidden=model(**batch).last_hidden_state
                mask=batch['attention_mask'][...,None]
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                result.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy())
                progress(20+60*min(len(texts),start+32)/max(1,len(texts)),'문장 유사도 분석')
    finally:
        del model
        if device=='cuda':torch.cuda.empty_cache()
    return np.concatenate(result) if result else np.empty((0,384),dtype=np.float32)


def ordered_path(candidates,beam_size=128):
    """Beam DP: a source word can belong to at most one ordered dialogue."""
    states={-1:(0.,[])}
    for options in candidates:
        updated={end:(score-.12,path+[None]) for end,(score,path) in states.items()}
        for option in options:
            predecessors=[(score,path) for end,(score,path) in states.items() if end<option['first']]
            if not predecessors:continue
            score,path=max(predecessors,key=lambda pair:pair[0])
            score+=option['score']-.65
            end=option['last']
            if end not in updated or score>updated[end][0]:updated[end]=(score,path+[option])
        states=dict(sorted(updated.items(),key=lambda item:item[1][0],reverse=True)[:beam_size])
    return max(states.values(),key=lambda pair:pair[0])[1] if states else []


def locate(segments,words,duration,progress=lambda *args:None,encoder=embeddings):
    from source_alignment import match_key,compact_match,locate_transcript
    if not segments:return []
    content=[i for i,w in enumerate(words) if match_key(w['word']) and w['end']>w['start']]
    missing=lambda:{'matched':False,'coverage':0.,'words':[],'word_indices':[]}
    if not content:return [missing() for _ in segments]
    units=[];start=content[0];previous=None
    for index in content:
        if previous is not None:
            between=''.join(w['word'] for w in words[previous:index])
            boundary=(words[index]['start']-words[previous]['end']>.55 or
                bool(re.search(r'[。！？!?\n]',between)) or
                (words[index].get('speaker_id') is not None and
                 words[previous].get('speaker_id') is not None and
                 words[index]['speaker_id']!=words[previous]['speaker_id']))
            if boundary:units.append((start,previous));start=index
        previous=index
    units.append((start,content[-1]))
    texts=[s['original_text'] for s in segments]
    unit_texts=[''.join(w['word'] for w in words[a:b+1]) for a,b in units]
    vectors=encoder(texts+unit_texts,lambda p,d:progress(p*.5,d))
    similarities=vectors[:len(texts)]@vectors[len(texts):].T
    coarse=locate_transcript(segments,words,duration)
    proposals=[];all_texts=[]
    for i,text in enumerate(texts):
        target=match_key(text);spans=set()
        lexical=[difflib.SequenceMatcher(None,target,match_key(t),autojunk=False).ratio() for t in unit_texts]
        seeds=set(np.argsort(similarities[i])[-8:].tolist())
        seeds.update(j for j,value in enumerate(lexical) if value>=.8)
        if coarse[i]['matched']:
            spans.add((coarse[i]['word_indices'][0],coarse[i]['word_indices'][-1]))
            seeds.update(j for j,(a,b) in enumerate(units) if a<=coarse[i]['word_indices'][0]<=b)
        for seed in seeds:
            # Expand clauses until their text covers the complete model dialogue.
            # Bounds depend on text size, never Gemini's estimated time.
            for left in range(seed,-1,-1):
                if len(match_key(''.join(unit_texts[left:seed+1])))>max(8,len(target)*1.8):break
                for right in range(seed,len(units)):
                    recognized=''.join(unit_texts[left:right+1]);key=match_key(recognized)
                    if len(key)>max(8,len(target)*1.8):break
                    if len(key)<len(target)*.45:continue
                    a,b=units[left][0],units[right][1]
                    # Character alignment refines endpoints when spelling agrees.
                    owners=[];source=''
                    for wi in range(a,b+1):
                        part=match_key(words[wi]['word']);source+=part;owners.extend([wi]*len(part))
                    hits=compact_match(target,source)
                    if hits and len(hits)/max(1,len(target))>=.6:
                        first,last=owners[hits[0][1]],owners[hits[-1][1]]
                        spans.add((first,last))
                    else:spans.add((a,b))
        row=[]
        for first,last in sorted(spans):
            recognized=''.join(w['word'] for w in words[first:last+1])
            key=match_key(recognized)
            lexical_score=difflib.SequenceMatcher(None,target,key,autojunk=False).ratio()
            row.append({'first':first,'last':last,'lexical':lexical_score,'text':recognized})
            all_texts.append(recognized)
        proposals.append(row)
    # Deduplicate candidates across repeated dialogue before model inference.
    unique=list(dict.fromkeys(all_texts));lookup={text:i for i,text in enumerate(unique)}
    candidate_vectors=encoder(unique,lambda p,d:progress(40+p*.5,d)) if unique else np.empty((0,vectors.shape[1]))
    for i,row in enumerate(proposals):
        for candidate in row:
            cosine=float(vectors[i]@candidate_vectors[lookup[candidate['text']]])
            candidate['semantic_similarity']=cosine
            candidate['score']=.85*cosine+.15*candidate['lexical']
        # Semantic agreement plus lexical evidence avoids mapping an unrelated
        # but emotionally similar short reaction to a complete sentence.
        row[:]=[c for c in row if c['semantic_similarity']>=
            (.92 if c['lexical']<.25 else .86 if c['lexical']<.6 else .78)
            and c['score']>=.72]
    path=ordered_path(proposals);result=[]
    for candidate in path:
        if candidate is None:result.append(missing());continue
        a,b=candidate['first'],candidate['last']
        result.append({'matched':True,'start':words[a]['start'],'end':min(duration,words[b]['end']),
            'coverage':candidate['lexical'],'semantic_similarity':candidate['semantic_similarity'],
            'alignment_model':MODEL,'words':words[a:b+1],'word_indices':list(range(a,b+1))})
    return result
