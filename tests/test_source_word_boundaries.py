from rich_transcript import enrich,join_words
from fish_pipeline import regions
from source_word_boundaries import repair_japanese_boundaries

def test_character_gap_does_not_split_a_japanese_lexical_unit():
    text='何にも何にも変わってねえなお前は。'
    chars=list(text);times=[(i*.1,i*.1+.1) if i<4 else (2+i*.1,2+i*.1+.1) for i in range(len(chars))]
    raw={'text':text,'words':[{'word':c,'start':a,'end':b,'speaker':'0'} for c,(a,b) in zip(chars,times)]}
    rows=[{**r,'id':i} for i,r in enumerate(regions(raw,6.))]
    old_times=[(r['start'],r['end']) for r in rows]
    repaired,audit=repair_japanese_boundaries(rows,text)
    assert repaired[0]['text']=='何にも'
    assert repaired[1]['text']=='何にも変わってねえなお前は。'
    assert [(r['start'],r['end']) for r in repaired]==old_times
    assert audit and audit[0]['timestamps_changed'] is False

def test_valid_japanese_pause_and_different_speakers_are_not_reassigned():
    raw={'text':'何にも何にも','words':[{'word':c,'start':i*.1+(2 if i>=3 else 0),'end':i*.1+.1+(2 if i>=3 else 0),'speaker':'0'} for i,c in enumerate('何にも何にも')]}
    rows=[{**r,'id':i} for i,r in enumerate(regions(raw,5.))]
    assert repair_japanese_boundaries(rows,raw['text'])[1]==[]

def test_raw_repeated_punctuation_is_preserved_without_emitting_owned_tokens_twice():
    for punctuation in ('。','。。','...','?!','！！'):
        raw={'text':'嘘だ'+punctuation,'words':[{'word':c} for c in '嘘だ'+punctuation]}
        assert join_words(enrich(raw)['words'])=='嘘だ'+punctuation


def test_entire_rows_inside_one_word_merge_with_both_source_windows():
    raw={'text':'残念。','words':[
        {'word':'残','start':127.4,'end':127.42,'speaker':'0'},
        {'word':'念','start':128.479,'end':128.62,'speaker':'0'},
        {'word':'。','start':128.62,'end':128.62,'speaker':'0'}]}
    rows=[{**r,'id':i} for i,r in enumerate(regions(raw,130.))]
    assert len(rows)==2
    repaired,audit=repair_japanese_boundaries(rows,raw['text'])
    assert len(repaired)==1 and repaired[0]['text']=='残念。'
    assert repaired[0]['start']==127.4 and repaired[0]['end']==128.62
    assert [w['start'] for w in repaired[0]['words']]==[127.4,128.479,128.62]
    assert audit[0]['method']=='merge_rows_inside_japanese_lexical_unit'


def test_entire_lexical_fragments_in_different_voices_are_not_merged():
    raw={'text':'残念。','words':[
        {'word':'残','start':1.,'end':1.1,'speaker':'0'},
        {'word':'念。','start':2.,'end':2.1,'speaker':'1'}]}
    rows=[{**r,'id':i} for i,r in enumerate(regions(raw,3.))]
    repaired,audit=repair_japanese_boundaries(rows,raw['text'])
    assert len(repaired)==2 and not audit
