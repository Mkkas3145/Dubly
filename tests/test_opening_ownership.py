import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from source_alignment import restore_sentence_openings,measure_collapsed_words
from source_recovery import accept_missing


class OpeningOwnershipTests(unittest.TestCase):
    def location(self,first,last,words):
        return dict(matched=True,word_indices=list(range(first,last+1)),words=words[first:last+1],
                    start=words[first]['start'],end=words[last]['end'],recognized_text=''.join(w['word'] for w in words[first:last+1]))

    def test_same_sentence_opening_keeps_commas_and_hesitation_regardless_of_translation_punctuation(self):
        words=[dict(word=t,start=a,end=b,speaker_label=s) for t,a,b,s in
               [('earlier.',0.,.2,'a'),('I,',1.,1.2,'b'),('well,',1.6,1.8,'b'),('agree',2.,2.5,'b')]]
        row=self.location(3,3,words)
        restore_sentence_openings([row],words)
        self.assertEqual(row['word_indices'],[1,2,3]);self.assertEqual(row['start'],1.)

    def test_other_voice_claimed_words_and_previous_sentence_are_not_annexed(self):
        for speaker,ending,claimed in [('other','grunt',False),('same','previous!',False),('same','claimed,',True)]:
            with self.subTest(speaker=speaker,ending=ending):
                words=[dict(word=ending,start=1.,end=1.2,speaker_label=speaker),
                       dict(word='body',start=1.5,end=2.,speaker_label='same')]
                row=self.location(1,1,words);locations=[row]
                if claimed:locations.insert(0,self.location(0,0,words))
                restore_sentence_openings(locations,words)
                self.assertEqual(row['word_indices'],[1])

    def test_longer_opening_cannot_borrow_the_previous_translations_clause(self):
        words=[dict(word=t,start=a,end=b,speaker_label='same') for t,a,b in
               [('previous,',1.,1.3),('body',1.6,2.)]]
        row=self.location(1,1,words)
        # A missing earlier translation owns the meaning of this unclaimed
        # clause. Same voice/sentence alone must not annex it to the next line.
        locations=[dict(matched=False),row]
        vectors={'earlier translation':[1.,0.],'current translation':[0.,1.],
                 'previous,':[1.,0.],'body':[0.,1.],'previous,body':[0.,1.]}
        restore_sentence_openings(locations,words,
            [dict(translated_text='earlier translation'),dict(translated_text='current translation')],
            encoder=lambda texts:np.asarray([vectors[t] for t in texts]))
        self.assertEqual(row['word_indices'],[1])

    def test_collapsed_lexical_anchor_is_measured_but_punctuation_and_silence_are_not(self):
        sr=16000;x=np.zeros((3*sr,2),np.float32)
        for lo,hi in [(1.05,1.2),(1.23,1.5),(2.,2.4)]:
            a,b=round(lo*sr),round(hi*sr);x[a:b]=.15*np.sin(np.arange(b-a)[:,None]*2*np.pi*220/sr)
        words=[dict(word=t,start=a,end=b,speaker_label='voice') for t,a,b in
               [('prefix',1.15,1.15),(',',1.15,1.15),('silent',1.8,1.8),('body',2.,2.4)]]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.wav';sf.write(path,x,sr)
            result=measure_collapsed_words(path,words)
        self.assertLessEqual(result[0]['start'],1.06);self.assertGreaterEqual(result[0]['end'],1.49)
        self.assertEqual(result[1],words[1]);self.assertEqual(result[2],words[2])
        self.assertEqual(words[0]['start'],words[0]['end'])

    def test_single_ordered_gap_needs_full_independent_asr_agreement_and_one_voice(self):
        words=[dict(word=t,start=a,end=b,speaker_label='a') for t,a,b in
               [('source',1.,1.3),('sentence.',1.3,1.8)]]
        job=dict(translated_text='idiomatic translation',neighbor_floor=.9,neighbor_ceiling=1.9,
                 source_index=2,asr_text='source sentence.',isolated_ordered_gap=True,minimum_text_ratio=.25)
        x=np.ones((3*16000,2),np.float32)*.1
        with patch('translation_alignment.locate',return_value=[dict(matched=False)]),patch('source_prefix.speech_support',return_value=1.):
            rows,reason=accept_missing(job,{'words':[]},3.,words,x,16000)
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['start'],1.)
            self.assertIn('independent_asr_agreement',reason)
            for modified in [dict(job,isolated_ordered_gap=False),dict(job,asr_text='unrelated words')]:
                self.assertEqual(accept_missing(modified,{'words':[]},3.,words,x,16000)[0],[])
            wrong=copy.deepcopy(words);wrong[-1]['speaker_label']='b'
            self.assertEqual(accept_missing(job,{'words':[]},3.,wrong,x,16000)[0],[])
            multiple=copy.deepcopy(words)
            multiple.insert(1,dict(word='。',start=1.3,end=1.3,speaker_label='a'))
            self.assertEqual(accept_missing(job,{'words':[]},3.,multiple,x,16000)[0],[])


if __name__=='__main__':unittest.main()
