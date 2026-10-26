"""Sentence meaning tolerates ASR spelling variation; deletion still needs extra-text evidence."""
from difflib import SequenceMatcher
import math,unicodedata
import numpy as np
from rich_transcript import spoken,normalize

MODEL='sentence-transformers/LaBSE'
POLICY={'version':1,'model':MODEL,'minimum_semantic_similarity':.84,
    'short_semantic_similarity':.90,'minimum_lexical_similarity':.55,
    'maximum_candidates':24,'cross_script':'phonetic_guard_only',
    'extra_dialogue':'must_be_explained_by_reference_before_any_crop'}
def family(c):
    name=unicodedata.name(c,'')
    if name.startswith(('HIRAGANA','KATAKANA','CJK')):return 'japanese-chinese'
    return name.split(' ')[0]
def comparable(expected,heard):
    wanted={family(c) for c in expected if c.isalpha()}
    heard=[c for c in heard if c.isalpha()]
    return bool(heard) and sum(family(c) in wanted for c in heard)/len(heard)>=.75
def locate(words,segment,explained_extras,encoder=None):
    """Return a whole-output keep or a uniquely justified target crop span."""
    expected=spoken(segment['translated_text']);target=normalize(expected)
    diagnostic={'model':MODEL,'policy':POLICY,'accepted':False}
    if len(target)<4 or not words:return [],{**diagnostic,'reason':'too_short_or_unrecognized'}
    if encoder is None:
        from embedding_client import encode
        encoder=encode
    tokens=[w.get('word',w.get('text','')) for w in words];proposals=[]
    for first in range(len(words)):
        value=''
        for last in range(first,len(words)):
            value+=(' ' if value else '')+tokens[last]
            compact=normalize(value);ratio=len(compact)/len(target)
            if ratio>1.8:break
            if ratio<.6 or not comparable(expected,value):continue
            lexical=SequenceMatcher(None,target,compact,autojunk=False).ratio()
            if lexical<POLICY['minimum_lexical_similarity']:continue
            proposals.append({'first':first,'last':last,'text':value,'lexical':lexical,'length_ratio':ratio})
    proposals.sort(key=lambda p:p['lexical']-.03*abs(math.log(p['length_ratio'])),reverse=True)
    whole=next((p for p in proposals if p['first']==0 and p['last']==len(words)-1),None)
    proposals=proposals[:POLICY['maximum_candidates']]
    if whole is not None and whole not in proposals:proposals[-1:]=[whole]
    if not proposals:return [],{**diagnostic,'reason':'no_same_language_sentence_candidate'}
    texts=list(dict.fromkeys([expected]+[p['text'] for p in proposals]));vectors=np.asarray(encoder(texts),dtype=np.float32)
    known=dict(zip(texts,vectors));threshold=POLICY['short_semantic_similarity'] if len(target)<10 else POLICY['minimum_semantic_similarity']
    accepted=[]
    for p in proposals:
        semantic=float(known[expected]@known[p['text']])
        p['semantic']=semantic;p['score']=.8*semantic+.2*p['lexical']-.04*abs(math.log(p['length_ratio']))
        if semantic>=threshold:accepted.append(p)
    if whole is not None:
        diagnostic['whole_sentence_similarity']=whole.get('semantic')
        diagnostic['whole_lexical_similarity']=whole['lexical']
    whole_ok=next((p for p in accepted if p['first']==0 and p['last']==len(words)-1),None)
    crops=[]
    for p in accepted:
        a,b=p['first'],p['last']
        if a==0 and b==len(words)-1:continue
        outside=words[:a]+words[b+1:]
        if any(normalize(w.get('word','')) and normalize(w.get('word','')) in target for w in outside):continue
        # A high semantic score may ignore polite endings. Never drop those merely
        # because a shorter candidate embeds better: every outside word needs proof.
        if not explained_extras(words[:a],segment) or not explained_extras(words[b+1:],segment):continue
        if whole_ok and p['score']<whole_ok['score']-.03:continue
        crops.append(p)
    crops.sort(key=lambda p:p['score'],reverse=True)
    if crops:
        best=crops[0]
        if any((p['last']<best['first'] or p['first']>best['last']) and p['score']>=best['score']-.04 for p in crops[1:]):
            return [],{**diagnostic,'reason':'ambiguous_sentence_matches'}
        return [(best['first'],best['last'])],{**diagnostic,'accepted':True,'selected':best,'reason':'semantic_target_with_verified_extras'}
    if whole_ok:
        return [(0,len(words)-1)],{**diagnostic,'accepted':True,'selected':whole_ok,'reason':'whole_sentence_semantic_match'}
    return [],{**diagnostic,'reason':'sentence_not_sufficiently_similar_or_unexplained_extras'}
