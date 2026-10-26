from corrected_source_groups import merge_corrected


def row(i,a,b,voice='0',**extra):
    return {'id':i,'start':a,'end':b,'speaker':voice,'text':str(i),**extra}


def test_corrected_times_control_merging_instead_of_raw_word_times():
    rows=[row(10,121.44,122.70,words=[{'word':'First.','start':107.46,'end':107.7}]),
          row(11,122.72,126.74,words=[{'word':'Second.','start':123.2,'end':126.44}])]
    merged,audit=merge_corrected(rows)
    assert len(merged)==1 and merged[0]['text']=='First.Second.'
    assert merged[0]['start']==121.44 and merged[0]['end']==126.74
    assert merged[0]['merged_source_ids']==[10,11]
    assert abs(audit[0]['corrected_gap_seconds']-.02)<1e-7
    assert len(rows[0]['words'])==1


def test_threshold_and_chain_keep_full_text_and_order():
    merged,_=merge_corrected([row(0,0,1),row(1,1.8,2),row(2,2.8,3),row(3,3.801,4)])
    assert len(merged)==2 and merged[0]['text']=='0 1 2'
    assert [r['id'] for r in merged]==[0,1]


def test_other_voices_unknown_overlap_and_keep_original_are_barriers():
    for second in [row(1,1.2,2,'1'),row(1,1.2,2,keep_original=True),row(1,.9,2)]:
        assert len(merge_corrected([row(0,0,1),second])[0])==2
    assert len(merge_corrected([row(0,0,1,'unknown'),row(1,1.1,2,'unknown')])[0])==2
