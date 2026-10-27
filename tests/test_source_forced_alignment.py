import json
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
from source_forced_alignment import measured_rows,align_rows
from mms_text import text_units


def test_mapped_words_own_real_gaps_and_preserve_speakers_and_text():
    rows=[{'id':0,'text':'Wait!','start':0.,'end':1.,'speaker':'4','translation':'잠ㄲ…'},
          {'id':1,'text':'Eh?','start':1.,'end':2.,'speaker':'9','translation':'어?'}]
    words=[{'word':'Wait','start':.9,'end':1.2},{'word':'Eh','start':2.,'end':2.08}]
    aligned=measured_rows(rows,words,3.,offset=10.)
    assert [(r['start'],r['end']) for r in aligned]==[(10.9,11.2),(12.,12.08)]
    assert [r['speaker'] for r in aligned]==['4','9']
    assert aligned[0]['text']=='Wait!' and aligned[0]['translation']=='잠ㄲ…'
    assert rows[0]['start']==0.


@pytest.mark.parametrize('words',[
    [{'word':'No','start':1.,'end':2.}],
    [{'word':'Wait','start':1.,'end':1.}],
    [{'word':'Wait','start':float('nan'),'end':2.}],
])
def test_invalid_alignment_is_not_replaced_with_gemini_time(words):
    with pytest.raises(ValueError):
        measured_rows([{'id':0,'text':'Wait','start':0.,'end':1.}],words,3.)


def test_source_stage_cache_and_original_clock(tmp_path):
    audio=tmp_path/'audio.wav';sf.write(audio,np.ones(3000,dtype='float32')*.1,1000)
    rows=[{'id':0,'text':'Wait','start':0.,'end':.5,'speaker':'2'}]
    result={'results':[{'words':[{'word':'Wait','start':1.1,'end':1.6}]}],
            'model_loads':1,'model_load_seconds':1.}
    with patch('source_forced_alignment.align_batch',return_value=result) as worker:
        measured,audit=align_rows(audio,rows,'en',tmp_path/'stage')
        again=align_rows(audio,rows,'en',tmp_path/'stage')
    assert worker.call_count==1
    assert measured[0]['start']==1.1 and measured[0]['end']==1.6
    assert again==(measured,audit)
    assert audit[0]['method']=='Meta/MMS-FA'


def test_japanese_readings_and_target_agnostic_text_normalization():
    ja=text_units('変わってねえな。','ja')
    assert ''.join(t['word'] for t in ja)=='変わってねえな'
    assert ja[0]['roman'].startswith('kawa')
    assert ''.join(t['word'] for t in text_units('정말? 네!','ko'))=='정말네'
    assert all(t['roman'] for t in text_units('مرحبا بالعالم','ar'))
    assert text_units('12','en')[0]['word']=='12'


def test_dialogue_edges_preserve_every_sample_without_fades_or_padding():
    from audio_edges import fade_edges
    for n in (80,1000):
        values=np.random.default_rng(1).uniform(-.5,.5,(n,2)).astype(np.float32)
        result=fade_edges(values,1000)
        np.testing.assert_array_equal(result,values)
        assert result.shape==values.shape and result is not values



def test_acoustic_chunks_cover_original_clock_without_gemini_time_windows():
    from mms_align_worker import emission_windows
    for samples in [80,400,16000,16000*3600]:
        windows=list(emission_windows(samples))
        frames=max(1,(samples-400)//320+1)
        assert windows[0][0]==0 and windows[-1][1]==frames
        assert all(a[1]==b[0] for a,b in zip(windows,windows[1:]))
        assert all(0<=left<=start<end<=right<=frames for start,end,left,right in windows)
        assert all(right-left<=850 for start,end,left,right in windows)


def test_gemini_time_drift_cannot_remove_source_audio_from_alignment(tmp_path):
    audio=tmp_path/'audio.wav';sf.write(audio,np.ones(100000,dtype='float32')*.1,1000)
    rows=[{'id':0,'text':'Wait','start':70.,'end':72.,'speaker':'2'},
          {'id':1,'text':'Eh','start':92.,'end':94.,'speaker':'3'}]
    result={'results':[{'words':[{'word':'Wait','start':61.5,'end':63.2},
                               {'word':'Eh','start':65.,'end':65.2}]}]}
    def worker(jobs,language,folder,progress):
        assert len(jobs)==1 and jobs[0]['texts']==['Wait','Eh']
        assert jobs[0]['audio']==str(audio.resolve())
        return result
    with patch('source_forced_alignment.align_batch',side_effect=worker):
        fixed,_=align_rows(audio,rows,'en',tmp_path/'stage')
    assert fixed[0]['start']==61.5 and fixed[1]['start']==65.

def test_japanese_standalone_prolonged_marks_preserve_source_and_extend_vowel():
    import torchaudio
    for text in ('やったー！','えーー？','うわー、待ってー！','ー','ーー'):
        units=text_units(text,'ja')
        assert ''.join(u['word'] for u in units)==''.join(c for c in text if c.isalnum())
        assert all(u['roman'] for u in units)
        assert all(torchaudio.pipelines.MMS_FA.get_tokenizer()([u['roman']])[0] for u in units)
    units=text_units('あ ー','ja')
    marker=next(i for i,u in enumerate(units) if u['word']=='ー')
    assert units[marker]['roman']=='a'
    assert text_units('ー','ja')==[{'word':'ー','roman':'*'}]

def test_untimed_source_rows_receive_only_acoustically_measured_times():
    rows=[{'id':0,'text':'Wait','speaker':'2','translation':'잠깐'}]
    words=[{'word':'Wait','start':6.1,'end':6.8}]
    result=measured_rows(rows,words,10)
    assert result[0]['start']==6.1 and result[0]['end']==6.8
    assert result[0]['coarse_start'] is None and result[0]['coarse_end'] is None
