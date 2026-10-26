import numpy as np
import soundfile as sf
from fish_speaker_guard import split_regions
from shared_reference import plan

def test_fish_label_flips_do_not_fragment_same_nemotron_reference_group():
    segments=[{'start':i,'end':i+.5,'original_text':'原文','translated_text':'대사',
        'fish_speakers':[speaker]} for i,speaker in enumerate(['A','B','A'])]
    timeline={'turns':[{**s,'speaker':'E0'} for s in segments]}
    groups=plan(segments,timeline)
    assert [g['indices'] for g in groups]==[[0,1,2]]

def test_fish_never_merges_different_nemotron_speakers():
    segments=[{'start':i,'end':i+.5,'original_text':'原文','translated_text':'대사','fish_speakers':['A']} for i in range(2)]
    timeline={'turns':[{**s,'speaker':f'E{i}'} for i,s in enumerate(segments)]}
    assert len(plan(segments,timeline))==2

def test_fish_change_splits_region_at_quiet_gap(tmp_path):
    path=tmp_path/'voice.wav';sf.write(path,np.zeros(3*44100),44100)
    source={'words':[{'word':'父','start':.1,'end':.8,'fish_speaker':'A'},
        {'word':'よし','start':1.2,'end':1.8,'fish_speaker':'B'}]}
    result=split_regions([{'id':0,'start':0.,'end':2.}],source,path)
    assert len(result)==2 and .8<result[0]['end']<1.2 and result[0]['end']==result[1]['start']

def test_overlapping_different_fish_voices_keep_original_instead_of_arbitrary_cut():
    source={'words':[{'word':'a','start':.1,'end':1.2,'fish_speaker':'A'},
        {'word':'b','start':1.,'end':1.8,'fish_speaker':'B'}]}
    result=split_regions([{'id':0,'start':0.,'end':2.}],source,None)
    assert len(result)==1 and result[0]['fish_conflict_unresolved']

def test_unchanged_regions_keep_exact_identity_for_translation_cache():
    rows=[{'id':0,'start':0.,'end':2.}]
    assert split_regions(rows,{'words':[]},None)==rows

def test_legacy_fish_isolation_flag_does_not_fragment_native_voice_bank():
    segments=[{'start':i,'end':i+.5,'original_text':'source','translated_text':'target',
        'fish_speakers':['A'],'reference_isolated':i==1} for i in range(3)]
    timeline={'turns':[{**s,'speaker':'E0'} for s in segments]}
    groups=plan(segments,timeline)
    assert [g['indices'] for g in groups]==[[0,1,2]]
    assert groups[0]['reference_indices']==[0,1,2]
