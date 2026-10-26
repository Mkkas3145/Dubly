from rich_transcript import enrich,join_words
from native_timeline import text_regions

def test_tags_punctuation_and_speaker_changes_follow_existing_words():
    result=enrich({'text':'<|speaker:2|>[嬉しい]スパイ、ミッション！ <|speaker:0|>よし。',
        'words':[{'word':v,'start':i,'end':i+.4} for i,v in enumerate(['スパイ','ミッション','よし'])]})
    words=result['words']
    assert [(w['start'],w['end']) for w in words]==[(0,.4),(1,1.4),(2,2.4)]
    assert words[0]['rich_surface']=='スパイ、' and words[1]['rich_surface']=='ミッション！'
    assert words[1]['vocal_tags']==['[嬉しい]'] and not words[2]['vocal_tags']
    assert words[0]['fish_speaker']!=words[2]['fish_speaker']
    assert join_words(words)=='[嬉しい]スパイ、ミッション！よし。'
    assert '<|speaker:' not in join_words(words)

def test_each_native_region_inherits_active_emotion_without_extra_llm_request():
    source=enrich({'text':'<|speaker:2|>[嬉しい]スパイ、ミッション、わくわく！',
        'words':[{'word':v,'start':i,'end':i+.5} for i,v in enumerate(['スパイ','ミッション','わくわく'])]})
    rows=[{'id':i,'start':i,'end':i+.8} for i in range(3)]
    timeline={'turns':[{**r,'speaker':'A'} for r in rows]}
    result=text_regions(rows,timeline,source)
    assert [r['text'] for r in result]==['[嬉しい]スパイ、','[嬉しい]ミッション、','[嬉しい]わくわく！']

def test_partial_mismatch_does_not_invent_times_or_words():
    r=enrich({'text':'[驚き]Hello!', 'words':[{'word':'different','start':1,'end':2}]})
    assert r['words'][0]['word']=='different' and r['words'][0]['start']==1

def test_timestamp_drift_does_not_steal_following_different_fish_speaker():
    source=enrich({'text':'<|speaker:2|>[嬉しい]わくわく！ <|speaker:3|>少女',
        'words':[{'word':'わくわく','start':.1,'end':.8},{'word':'少女','start':1.04,'end':1.52}]})
    rows=[{'id':0,'start':0.,'end':1.}];timeline={'turns':[{**rows[0],'speaker':'E0'}]}
    result=text_regions(rows,timeline,source)
    assert result[0]['text']=='[嬉しい]わくわく！'
