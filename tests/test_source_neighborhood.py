import unittest
from unittest.mock import patch
from source_neighborhood import match_neighborhood,verified_native_words,assign_speakers
from translation_alignment import source_units


class NeighborhoodTests(unittest.TestCase):
    def test_single_collapsed_api_timestamp_is_not_trusted(self):
        native={'text':'Several words here','words':[dict(word='Several words here',start=9.,end=9.1)]}
        self.assertIsNone(verified_native_words(native,0.,10.,[]))

    def test_native_times_require_distributed_independent_anchors(self):
        words=[dict(word=t,start=float(i*2),end=float(i*2+1)) for i,t in enumerate(['Alpha','Beta','Gamma','Delta'])]
        source=[dict(w,start=w['start']+.1,end=w['end']+.1) for w in words]
        source.insert(2,dict(word='。',start=3.1,end=3.1))
        result=verified_native_words({'text':'Alpha Beta Gamma Delta','words':words},0.,8.,source)
        self.assertIsNotNone(result);self.assertAlmostEqual(result[0]['start'],.1)
        self.assertEqual(result[1]['word'],'Beta。')
        self.assertIsNone(verified_native_words({'text':'Alpha Beta Gamma Delta','words':words},0.,8.,source[:2]))

    def test_missing_voice_labels_propagate_only_between_matching_anchors(self):
        words=[dict(word=str(i),start=float(i),end=float(i+1)) for i in range(5)]
        source=[dict(words[0],speaker_label='a'),dict(words[2],speaker_label='a'),dict(words[4],speaker_label='b')]
        assign_speakers(words,source)
        self.assertEqual(words[1]['speaker_label'],'a');self.assertNotIn('speaker_label',words[3])

    def test_pause_inside_sentence_does_not_split_off_its_final_noun(self):
        words=[dict(word='noun of',start=1.,end=1.3,speaker=0),
               dict(word='egg?',start=2.1,end=2.3,speaker=0),
               dict(word='next.',start=2.4,end=2.8,speaker=0)]
        self.assertEqual(source_units(words),[(0,0),(1,1),(2,2)])
        self.assertEqual(source_units(words,ignore_gaps=True),[(0,1),(2,2)])

    def test_every_phrase_must_match_before_assigning_a_whole_translation(self):
        with patch('source_neighborhood.locate',return_value=[dict(matched=True),dict(matched=False)]):
            self.assertIsNone(match_neighborhood([(0,'First sentence. Second sentence!')],[],10))

    def test_same_voice_phrases_keep_the_intervening_source_audio(self):
        words=[dict(word='first.',start=1.,end=1.3),dict(word='unrecognized bridge',start=1.5,end=2.4),
               dict(word='last.',start=2.5,end=2.8)]
        locations=[dict(matched=True,word_indices=[0],recognized_text='first.',coverage=.8,asr_speaker_label='a'),
                   dict(matched=True,word_indices=[2],recognized_text='last.',coverage=.8,asr_speaker_label='a')]
        with patch('source_neighborhood.locate',return_value=locations):
            rows=match_neighborhood([(6,'First. Last!')],words,10)
        self.assertEqual(len(rows),1);self.assertEqual((rows[0]['start'],rows[0]['end']),(1.,2.8))
        self.assertEqual(rows[0]['words'],words);self.assertEqual(rows[0]['translated_text'],'First. Last!')

    def test_different_voices_keep_separate_references_and_all_translated_words(self):
        words=[dict(word='first.',start=1.,end=1.3),dict(word='last.',start=1.5,end=1.8)]
        locations=[dict(matched=True,word_indices=[0],recognized_text='first.',coverage=.8,asr_speaker_label='a'),
                   dict(matched=True,word_indices=[1],recognized_text='last.',coverage=.8,asr_speaker_label='b')]
        with patch('source_neighborhood.locate',return_value=locations):
            rows=match_neighborhood([(6,'First. Last!')],words,10)
        self.assertEqual(len(rows),2);self.assertTrue(all(r['speaker_boundary_split'] for r in rows))
        self.assertEqual([r['translated_text'] for r in rows],['First.','Last!'])
        self.assertLessEqual(rows[0]['end'],rows[1]['start'])

if __name__=='__main__':unittest.main()
