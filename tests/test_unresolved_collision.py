from unittest.mock import patch
import json
import numpy as np
import soundfile as sf
from core import SR
import render


def test_unresolved_long_tts_is_kept_original_without_stopping_other_segments(tmp_path):
    segments=[{'start':0.,'end':.5,'lo':0,'hi':SR//2,'translated_text':'First'},
        {'start':1.,'end':1.5,'lo':SR,'hi':SR+SR//2,'translated_text':'Second'}]
    def generate(index,segment,*args):
        return {'index':index,'cached':{'index':index,'start':segment['start'],'end':segment['end'],
            'output_frames':42*SR if index==0 else SR//2,'collision_budget_frames':SR if index==0 else None}}
    with patch.object(render,'generate_speech',side_effect=generate):
        reports=render.synthesize_individual_all(segments,None,tmp_path,
            {'tts_mode':'individual','tts_output_guard':None,'keep_original_on_tts_failure':True},lambda *a:None)
    assert reports[0]['keep_original'] and reports[0]['output_frames']==1
    assert segments[0]['keep_original'] and not segments[1].get('keep_original')
    assert reports[1]['output_frames']==SR//2
    assert sf.info(tmp_path/'0000-fitted.wav').frames==1


def test_deadline_check_allows_rounding_but_not_failed_speech_overrun():
    assert render.fits_next_dialogue({'output_frames':SR+100,'collision_budget_frames':SR})
    assert not render.fits_next_dialogue({'output_frames':42*SR,'collision_budget_frames':SR})
    assert render.fits_next_dialogue({'output_frames':42*SR,'collision_budget_frames':None})
