"""Keep Japanese lexical units intact when ASR returns character-sized words."""
import re
from functools import lru_cache
from rich_transcript import join_words,ANNOTATION

@lru_cache(maxsize=1)
def japanese_tokenizer():
    from sudachipy import dictionary
    return dictionary.Dictionary().create()

def repair_japanese_boundaries(rows,text):
    text=ANNOTATION.sub('',text)
    if not re.search(r'[ぁ-んァ-ヶ一-龯]',text):return rows,[]
    from sudachipy import tokenizer
    units=[]
    for token in japanese_tokenizer().tokenize(text,tokenizer.Tokenizer.SplitMode.C):
        start,end=token.begin(),token.end();pos=token.part_of_speech()[0]
        if units and pos in ('助詞','助動詞','接尾辞') and units[-1][1]==start:
            units[-1]=(units[-1][0],end)
        elif pos not in ('補助記号','空白'):units.append((start,end))
    result=[{**r,'words':[dict(w) for w in r.get('words',[])]} for r in rows];audit=[]
    for left,right in zip(result,result[1:]):
        if left['speaker']=='unknown' or left['speaker']!=right['speaker'] or not right['words']:continue
        boundary=right['words'][0].get('source_char_start')
        if boundary is None:continue
        unit=next(((a,b) for a,b in units if a<boundary<b),None)
        if unit is None:continue
        moved=[w for w in left['words'] if unit[0]<=w.get('source_char_start',-1)<boundary]
        kept=[w for w in left['words'] if w not in moved]
        if not moved or left['words'][-len(moved):]!=moved:continue
        if not kept:
            # A pause can split a single lexical unit into two entire rows.
            # Move its source window too: otherwise its first sound is lost.
            right['words']=moved+right['words']
            right['text']=join_words(right['words'])
            right['start']=min(left['start'],right['start'])
            right['end']=max(left['end'],right['end'])
            left['_merged_into_next']=True
            audit.append({'left_id':left['id'],'right_id':right['id'],'source_span':list(unit),
                'moved_word_ids':[w['word_id'] for w in moved],
                'method':'merge_rows_inside_japanese_lexical_unit','timestamps_changed':True})
            continue
        # Correct text membership, not ASR timestamps. The measured pause still
        # controls the two source windows; the ASR word times remain available.
        left['words']=kept;right['words']=moved+right['words']
        left['text']=join_words(left['words']);right['text']=join_words(right['words'])
        audit.append({'left_id':left['id'],'right_id':right['id'],'source_span':list(unit),
            'moved_word_ids':[w['word_id'] for w in moved],'method':'japanese_morphological_unit_crossed_pause_boundary',
            'timestamps_changed':False})
    result=[r for r in result if not r.pop('_merged_into_next',False)]
    for i,row in enumerate(result):row['id']=i
    return result,audit
