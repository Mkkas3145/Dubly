from unittest.mock import patch
import render
from core import config


def test_production_uses_grouped_generation_with_sixteen_paired_tag_slots():
    cfg=config()
    assert cfg['tts_mode']=='same_speaker_batches'
    assert cfg['tts_reference_mode']=='same_speaker_batch'
    assert cfg['tts_output_guard'] is None
    from batch_tts import BATCH_REFERENCE_POLICY,plan_speaker_batches
    assert BATCH_REFERENCE_POLICY['target_seconds'] is None
    assert BATCH_REFERENCE_POLICY['max_parts']==16
    rows=[{'speaker_id':'0','start':i,'end':i+.5} for i in range(65)]
    assert list(map(len,plan_speaker_batches(rows)))==[16,16,16,16,1]


def test_individual_mode_ignores_legacy_group_metadata(tmp_path):
    segments=[{'start':0.,'end':1.,'lo':0,'hi':44100,'translated_text':'안녕','_tts_group':{}}]
    with patch('render.generate_speech',return_value={'index':0,'cached':{'index':0}}) as generate, \
         patch('same_speaker_tts.synthesize_aligned',side_effect=AssertionError('grouped generation')):
        result=render.synthesize_all(segments,None,tmp_path,{'tts_mode':'individual'},lambda *a:None)
    generate.assert_called_once()
    assert result==[{'index':0}]


def test_individual_failures_keep_only_affected_lines_and_continue(tmp_path):
    segments=[{'start':float(i),'end':i+1.,'lo':i*44100,'hi':(i+1)*44100,
               'translated_text':'대사'} for i in range(3)]
    def generate(index,*args):
        if index==0:raise RuntimeError('generation failed')
        return {'index':index}
    original_fit=render.fit_speech
    def fit(prepared,cfg):
        if 'cached' in prepared:return original_fit(prepared,cfg)
        if prepared['index']==1:raise RuntimeError('timing failed')
        return {'index':2}
    with patch('render.generate_speech',side_effect=generate),patch('render.fit_speech',side_effect=fit):
        results=render.synthesize_all(segments,None,tmp_path,
            {'tts_mode':'individual','keep_original_on_tts_failure':True},lambda *a:None)
    assert [r['index'] for r in results]==[0,1,2]
    assert results[0]['reason']=='tts_generation_failed'
    assert results[1]['reason']=='tts_timing_failed'
    assert segments[0]['keep_original'] and segments[1]['keep_original']
    assert not segments[2].get('keep_original')
