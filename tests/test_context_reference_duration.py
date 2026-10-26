import json
from unittest.mock import patch
import numpy as np
import soundfile as sf
from core import SR
from batch_tts import prepare_regeneration_reference

def segment(i,duration,voice='0'):
    return {'lo':i*10*SR,'hi':round((i*10+duration)*SR),'start':i*10.,'end':i*10+duration,
        'speaker_id':voice,'original_text':f'line {i}','translated_text':f'target {i}'}

def crop(clean,row,path):
    sf.write(path,np.full(row['hi']-row['lo'],.1,np.float32),SR,subtype='PCM_16')

def test_current_line_longer_than_two_seconds_is_used_alone_without_truncation(tmp_path):
    rows=[segment(i,3.) for i in range(20)]
    with patch('render.reference_audio',side_effect=crop):
        indices=prepare_regeneration_reference(rows,10,None,tmp_path)
    assert indices==[10]
    path=rows[10]['_shared_reference_path'];assert sf.info(path).duration==3.
    meta=json.load(open(path.replace('.wav','.json'),encoding='utf8'))
    assert meta['minimum_seconds']==2. and meta['minimum_met']
    assert meta['reference_text']=='line 10'


def test_one_sufficient_donor_is_preferred_to_many_short_lines_and_other_voices(tmp_path):
    rows=[segment(0,.3),segment(1,.4),segment(2,4.),segment(3,9.,'other')]
    with patch('render.reference_audio',side_effect=crop):
        indices=prepare_regeneration_reference(rows,0,None,tmp_path)
    assert indices==[0,2]
    assert sf.info(rows[0]['_shared_reference_path']).duration==4.3


def test_added_silence_does_not_satisfy_minimum_and_short_speaker_is_not_repeated(tmp_path):
    rows=[segment(i,.5) for i in range(5)]
    with patch('render.reference_audio',side_effect=crop):
        indices=prepare_regeneration_reference(rows,0,None,tmp_path)
    assert len(indices)==4 and len(set(indices))==4
    path=rows[0]['_shared_reference_path'];meta=json.load(open(path.replace('.wav','.json'),encoding='utf8'))
    assert meta['content_seconds']==2. and meta['reference_seconds']==2.
    rows=[segment(0,.3),segment(1,.4)]
    with patch('render.reference_audio',side_effect=crop):
        assert prepare_regeneration_reference(rows,0,None,tmp_path)==[0,1]
    path=rows[0]['_shared_reference_path'];meta=json.load(open(path.replace('.wav','.json'),encoding='utf8'))
    assert not meta['minimum_met'] and meta['content_seconds']==.7


def test_silent_or_overlapping_donors_are_skipped_without_changing_speaker_assignment(tmp_path):
    rows=[segment(0,.4),segment(1,3.),segment(2,4.),segment(3,3.)]
    other={**segment(1,3.,'other')};rows.append(other)
    def guarded_crop(clean,row,path):
        values=np.zeros(row['hi']-row['lo'],np.float32) if row is rows[2] else np.full(row['hi']-row['lo'],.1,np.float32)
        sf.write(path,values,SR,subtype='PCM_16')
    with patch('render.reference_audio',side_effect=guarded_crop):
        assert prepare_regeneration_reference(rows,0,None,tmp_path)==[0,3]
    assert [r['speaker_id'] for r in rows]==['0','0','0','0','other']
