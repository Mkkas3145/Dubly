from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from core import SR
from studio_workflow import use_raw_generation,restore_generation,clip_file,digest,listening_audio
from mix_levels import TTS_GAIN
from test_studio_workflow import fixture


def test_raw_replacement_preserves_samples_position_and_undo(tmp_path):
    video,_,doc,ctx,_=fixture(tmp_path)
    base=clip_file(video,0,tmp_path);previous=digest(base)
    raw=base.with_name('0000-tts.mp3')
    values=np.column_stack([np.linspace(-.2,.2,SR,dtype=np.float32)]*2)
    sf.write(raw,values,SR,format='WAV',subtype='FLOAT')
    doc['transcript']['segments'][0]['audio_edit']={'position':.2,'trim_start':.3,'speed':2,'fade_in':.1}
    doc=use_raw_generation(video,0,doc,tmp_path)
    actual,rate=sf.read(base,dtype='float32',always_2d=True)
    np.testing.assert_array_equal(actual,values)
    assert rate==SR and doc['transcript']['segments'][0]['audio_edit']=={'position':.2}
    played=listening_audio(video,0,doc,ctx,tmp_path)
    corrected,_=sf.read(played,dtype='float32',always_2d=True)
    np.testing.assert_allclose(corrected,values*TTS_GAIN,atol=1e-7)
    restored,_=restore_generation(video,0,previous,tmp_path)
    assert digest(base)==previous
    assert restored['transcript']['segments'][0]['audio_edit']['speed']==2


def test_missing_or_split_raw_does_not_change_audio(tmp_path):
    video,_,doc,ctx,_=fixture(tmp_path);base=clip_file(video,0,tmp_path);before=digest(base)
    with pytest.raises(ValueError):use_raw_generation(video,0,doc,tmp_path)
    base.with_name('0000-tts.mp3').write_bytes(b'not an audio file')
    doc['generated']['0']['split_from']=0
    with pytest.raises(ValueError):use_raw_generation(video,0,doc,tmp_path)
    assert digest(base)==before
