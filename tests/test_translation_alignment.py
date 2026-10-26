import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from translation_alignment import locate,source_units,ordered,clauses,partition_clauses
from reference_edges import acoustic_boundaries


class TranslationAlignmentTests(unittest.TestCase):
    def test_complete_candidate_is_considered_before_short_fragment_is_rejected(self):
        def encoder(texts,progress):
            lookup={'eightchr':[1.,0.],'ab':[1.,0.],'cdef':[0.,1.],'abcdef':[.8,.6]}
            return np.asarray([lookup[t] for t in texts],dtype=np.float32)
        words=[dict(word='ab',start=1.,end=1.2),dict(word='cdef',start=1.8,end=2.)]
        kwargs=dict(encoder=encoder,min_similarity=.3,allow_gap_rescue=False)
        partial=locate([{'translated_text':'eightchr'}],words,3.,**kwargs)[0]
        complete=locate([{'translated_text':'eightchr'}],words,3.,min_source_text_ratio=.5,**kwargs)[0]
        self.assertEqual(partial['recognized_text'],'ab')
        self.assertEqual(complete['recognized_text'],'abcdef')
        self.assertEqual((complete['start'],complete['end']),(1.,2.))

    def test_cross_language_match_uses_translations_and_asr_not_llm_originals(self):
        observed=[]
        def encoder(texts,progress):
            observed.extend(texts)
            a={'안녕하세요','こんにちは'}
            return np.asarray([[1.,0.] if t in a else [0.,1.] for t in texts],dtype=np.float32)
        words=[dict(word='こんにちは',start=1.,end=2.,speaker=0),
               dict(word='さようなら',start=3.,end=4.,speaker=1)]
        segments=[dict(translated_text='안녕하세요',original_text='POISON_ORIGINAL'),
                  dict(translated_text='잘 가요',original_text='POISON_ORIGINAL')]
        result=locate(segments,words,5.,encoder=encoder)
        self.assertEqual([r['start'] for r in result],[1.,3.])
        self.assertEqual([r['recognized_text'] for r in result],['こんにちは','さようなら'])
        self.assertNotIn('POISON_ORIGINAL',observed)

    def test_source_units_keep_zero_duration_characters_and_respect_real_speaker_field(self):
        words=[dict(word='ご',start=1.,end=1.2,speaker_label='speaker_0'),
            dict(word='み',start=1.2,end=1.2,speaker_label='speaker_0'),
            dict(word='の',start=1.2,end=1.3,speaker_label='speaker_0'),
            dict(word='別',start=1.3,end=1.5,speaker_label='speaker_1')]
        self.assertEqual(source_units(words),[(0,2),(3,3)])

    def test_order_and_unique_word_ownership_with_an_unmatchable_line(self):
        options=[[dict(first=9,last=10,score=.95),dict(first=1,last=2,score=.9)],[],
            [dict(first=5,last=7,score=.95)]]
        path=ordered(options)
        self.assertEqual([r['first'] if r else None for r in path],[1,None,5])

    def test_ordered_unclaimed_gap_recovers_short_labels_without_lowering_global_threshold(self):
        values={'translation one':[.42,0,0,.9075],'translation two':[0,.49,0,.8717],
                'one':[1,0,0,0],'two':[0,1,0,0],'anchor':[0,0,1,0]}
        def encoder(texts,progress):
            return np.asarray([values.get(t,[0,0,0,0]) for t in texts],dtype=np.float32)
        words=[dict(word='one',start=1.,end=2.),dict(word='two',start=3.,end=4.),
               dict(word='anchor',start=5.,end=6.)]
        result=locate([dict(translated_text=t) for t in ['translation one','translation two','anchor']],
                      words,7.,encoder=encoder)
        self.assertEqual([r['start'] for r in result],[1.,3.,5.])
        self.assertEqual(result[0]['alignment_repair'],'ordered_unclaimed_asr_gap')

    def test_unpartitioned_translation_cannot_create_a_mixed_speaker_reference(self):
        def encoder(texts,progress):
            return np.asarray([[1.,0.] if t in {'whole exchange','firstsecond'} else
                               [0.,1.] for t in texts],dtype=np.float32)
        words=[dict(word='first',start=1.,end=2.,speaker_label='speaker_0'),
               dict(word='second',start=2.5,end=3.,speaker_label='speaker_1')]
        result=locate([dict(translated_text='whole exchange')],words,4.,encoder=encoder)
        self.assertFalse(result[0]['matched'])
        self.assertEqual(result[0]['reason'],'speaker_boundary_without_safe_translation_cut')

    def test_speaker_partition_keeps_contiguous_phrases_and_all_words(self):
        parts=clauses('First line. Another sentence!\nSecond voice…')
        self.assertEqual(parts,['First line.','Another sentence!','Second voice…'])
        scores=np.asarray([[.95,.1],[.9,.2],[.1,.94]])
        cuts=partition_clauses(parts,scores,[{},{}])
        self.assertEqual(cuts,[(0,2),(2,3)])
        self.assertEqual([p for a,b in cuts for p in parts[a:b]],parts)

    def test_two_speaker_parts_survive_reference_measurement_without_text_loss(self):
        import soundfile as sf
        from reference_timing import refine_references
        sr=16000;x=np.zeros((4*sr,2),dtype=np.float32);rows=[]
        for i,start in enumerate([1.,2.]):
            lo,hi=round(start*sr),round((start+.3)*sr)
            x[lo:hi]=.2*np.sin(np.arange(hi-lo)[:,None]*2*np.pi*220/sr)
            rows.append(dict(source_index=0,start=start,end=start+.3,
                recognized_text=f'source {i}',translated_text=f'translation {i}',
                asr_speaker_label=f'speaker_{i}',speaker_boundary_split=True,
                words=[dict(word='word',start=start,end=start+.3,timing_source='scribe_translation_match')]))
        evidence=dict(segments=rows,unmatched_segments=[],model='test-asr',usage={})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sf.write(root/'source.wav',x,sr)
            with patch('reference_timing.source_evidence',return_value=evidence),patch('reference_timing.recover_prefixes') as prefix:
                result,measured=refine_references(root/'source.wav',dict(
                    translation_output='translated_strings_only',segments=[dict(translated_text='translation 0 translation 1')]),root/'result')
            prefix.assert_not_called()
        self.assertEqual([s['translated_text'] for s in result['segments']],['translation 0','translation 1'])
        self.assertEqual([s['asr_speaker_label'] for s in result['segments']],['speaker_0','speaker_1'])
        self.assertLessEqual(measured[0]['end'],measured[1]['start'])

    def test_zero_duration_only_unit_is_not_encoded_as_an_independent_span(self):
        def encoder(texts,progress):
            return np.ones((len(texts),2),dtype=np.float32)/np.sqrt(2)
        words=[dict(word='silent',start=0.,end=0.,speaker_label='speaker_0'),
               dict(word='spoken',start=1.,end=2.,speaker_label='speaker_1')]
        result=locate([dict(translated_text='translation')],words,3.,encoder=encoder)
        self.assertTrue(result[0]['matched']);self.assertEqual(result[0]['word_indices'],[1])

    def test_positive_asr_anchor_does_not_grab_an_earlier_grunt_inside_search_margin(self):
        for sr in [16000,44100,48000]:
            with self.subTest(sr=sr):
                x=np.zeros((3*sr,2),dtype=np.float32)
                for a,b in [(1.,1.25),(1.4,1.7),(2.,2.5)]:
                    lo,hi=round(a*sr),round(b*sr)
                    x[lo:hi]=.2*np.sin(np.arange(hi-lo)[:,None]*2*np.pi*230/sr)
                words=[dict(word='first',start=1.5,end=1.7,timing_source='scribe_translation_match'),
                       dict(word='last',start=2.,end=2.5,timing_source='scribe_translation_match')]
                r=acoustic_boundaries(x,sr,[dict(words=words)])[0]
                self.assertGreater(r['start'],1.3);self.assertLessEqual(r['start'],1.4)

    def test_unselected_asr_voice_bounds_prevent_absorbing_a_previous_speaker(self):
        sr=16000;x=np.zeros((3*sr,2),dtype=np.float32)
        for a,b in [(1.,1.3),(1.45,1.8)]:
            lo,hi=round(a*sr),round(b*sr);x[lo:hi]=.2*np.sin(np.arange(hi-lo)[:,None]*2*np.pi*220/sr)
        r=acoustic_boundaries(x,sr,[dict(words=[dict(word='target',start=1.5,end=1.8,
            timing_source='scribe_translation_match')],source_word_floor=1.3,source_word_ceiling=2.)])[0]
        self.assertGreaterEqual(r['start'],1.3);self.assertLessEqual(r['start'],1.45)


if __name__=='__main__':unittest.main()
