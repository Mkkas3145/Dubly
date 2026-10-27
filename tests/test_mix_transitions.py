from pathlib import Path
import numpy as np
import soundfile as sf
from core import SR
from render import transition_weights,mix
from mix_levels import TTS_GAIN


def test_full_plateau_and_both_fades_are_outside_the_interval():
    w=transition_weights(0,4*SR,[{'lo':SR,'hi':3*SR}])
    np.testing.assert_array_equal(w[SR:3*SR],1)
    assert w[round(.9*SR)]==0 and w[round(3.1*SR)]==0
    assert abs(w[round(.95*SR)]-.5)<1e-6
    assert abs(w[round(3.05*SR)]-.5)<1e-6
    # Rendering in tiles must not restart or shrink the crossfade.
    np.testing.assert_array_equal(w,np.concatenate([transition_weights(0,2*SR,[{'lo':SR,'hi':3*SR}]),transition_weights(2*SR,2*SR,[{'lo':SR,'hi':3*SR}])]))


def test_mix_keeps_tts_at_full_gain_while_only_original_and_bed_crossfade(tmp_path):
    source=np.full((4*SR,2),.2,np.float32);clean=np.full_like(source,.1);bed=np.full_like(source,.02)
    for name,values in [('original',source),('clean',clean),('background',bed)]:sf.write(tmp_path/(name+'.wav'),values,SR,subtype='FLOAT')
    tts=np.full((SR,2),.05,np.float32);sf.write(tmp_path/'0000-fitted.wav',tts,SR,subtype='FLOAT')
    mix(tmp_path/'original.wav',tmp_path/'clean.wav',[{'lo':SR,'hi':2*SR}],tmp_path,tmp_path/'mixed.wav',lambda *a:None,background=tmp_path/'background.wav')
    result,_=sf.read(tmp_path/'mixed.wav',dtype='float32',always_2d=True)
    np.testing.assert_allclose(result[SR:2*SR],bed[SR:2*SR]+tts*TTS_GAIN,atol=1e-7)
    np.testing.assert_array_equal(result[:round(.9*SR)],source[:round(.9*SR)])
    np.testing.assert_array_equal(result[round(2.1*SR):],source[round(2.1*SR):])
