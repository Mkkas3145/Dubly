"""Resident MFA/Kalpy aligner. Run only with .mfa-env/python.exe."""
from __future__ import annotations
from contextlib import redirect_stdout
import json,os,sys,tempfile,time,traceback
from pathlib import Path


# Windows extension modules require explicit DLL search directories after relocation.
_DLL_HANDLES=[]
if os.name=='nt':
    root=Path(__file__).resolve().parent
    for directory in (root/'.mfa-env/Library/bin',root.parent/'runtime/python/Lib/site-packages/torch/lib'):
        if directory.is_dir():_DLL_HANDLES.append(os.add_dll_directory(str(directory)))


class ResidentAligner:
    def __init__(self,folder):
        from kalpy.feat.cmvn import CmvnComputer
        from kalpy.fstext.lexicon import LexiconCompiler,Pronunciation
        from montreal_forced_aligner.models import AcousticModel,G2PModel
        from montreal_forced_aligner.tokenization.simple import SimpleTokenizer
        from montreal_forced_aligner.dictionary.mixins import (
            DEFAULT_BRACKETS,DEFAULT_CLITIC_MARKERS,DEFAULT_COMPOUND_MARKERS,DEFAULT_PUNCTUATION,DEFAULT_WORD_BREAK_MARKERS)
        root=Path(__file__).resolve().parent/'models/mfa/pretrained_models'
        self.acoustic=AcousticModel(root/'acoustic/korean_mfa.zip',folder/'acoustic')
        self.g2p=G2PModel(root/'g2p/korean_mfa.zip',folder/'g2p')
        p=self.acoustic.parameters
        self.lexicon=LexiconCompiler(disambiguation=False,silence_probability=p['silence_probability'],
            initial_silence_probability=p['initial_silence_probability'],final_silence_correction=p['final_silence_correction'],
            final_non_silence_correction=p['final_non_silence_correction'],silence_phone=p['optional_silence_phone'],
            oov_phone=p['oov_phone'],position_dependent_phones=p['position_dependent_phones'],phones=p['non_silence_phones'],ignore_case=True)
        self.lexicon.load_pronunciations(root/'dictionary/korean_mfa.dict')
        self.lexicon.create_fsts();self.lexicon.clear()
        self.tokenizer=SimpleTokenizer(word_table=self.lexicon.word_table,word_break_markers=DEFAULT_WORD_BREAK_MARKERS,
            punctuation=DEFAULT_PUNCTUATION,clitic_markers=DEFAULT_CLITIC_MARKERS,compound_markers=DEFAULT_COMPOUND_MARKERS,brackets=DEFAULT_BRACKETS)
        self.pronunciation=Pronunciation;self.g2p_cache={}
        self.aligner=None;self.vocabulary_size=-1
        self.mfcc=self.acoustic.mfcc_computer;self.cmvn=CmvnComputer()

    def tokenize(self,text):
        tokenized=self.tokenizer(text)
        pronunciation_text=tokenized.pronunciation_text or tokenized.normalized_text
        for norm,word in zip(tokenized.normalized_text.split(),pronunciation_text.split()):
            # The CLI computes G2P even for dictionary words, then discards it.
            # Skip that discarded work; unknown words keep the identical G2P model.
            if self.lexicon.word_table.member(norm):continue
            if word not in self.g2p_cache:self.g2p_cache[word]=self.g2p.rewriter(word)
            pronunciations=self.g2p_cache[word]
            if pronunciations:self.lexicon.add_pronunciation(self.pronunciation(norm,pronunciations[0].pronunciation))
        return tokenized.normalized_text

    def align(self,audio,text,output):
        from kalpy.aligner import KalpyAligner
        from kalpy.decoder.training_graphs import TrainingGraphCompiler
        from kalpy.utterance import Segment,Utterance
        import soundfile as sf
        transcript=self.tokenize(text)
        size=self.lexicon.word_table.num_symbols()
        if self.aligner is None:self.aligner=KalpyAligner(self.acoustic,self.lexicon)
        elif size!=self.vocabulary_size:
            self.aligner.graph_compiler=TrainingGraphCompiler(self.acoustic.model_path,self.acoustic.tree_path,self.lexicon)
        self.vocabulary_size=size
        utterance=Utterance(Segment(Path(audio)),transcript)
        utterance.generate_mfccs(self.mfcc)
        utterance.apply_cmvn(self.cmvn.compute_cmvn_from_features([utterance.mfccs]))
        ctm=self.aligner.align_utterance(utterance)
        ctm.export_textgrid(Path(output),file_duration=sf.info(audio).duration,output_format='json')


def main():
    root=Path(__file__).resolve().parent/'work/mfa-runtime';root.mkdir(parents=True,exist_ok=True)
    engine=None;completed=0
    with tempfile.TemporaryDirectory(prefix='resident-',dir=root) as temporary:
        for line in sys.stdin:
            request={};started=time.monotonic()
            try:
                request=json.loads(line)
                with redirect_stdout(sys.stderr):
                    if engine is None:engine=ResidentAligner(Path(temporary))
                    engine.align(request['audio'],request['text'],request['output'])
                completed+=1
                result={'ok':True,'id':request['id'],'pid':os.getpid(),'model_loads':1,
                        'completed':completed,'seconds':round(time.monotonic()-started,6)}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                result={'ok':False,'id':request.get('id'),'fatal':engine is None,'error':f'{type(error).__name__}: {error}'}
            print('MFA_RESULT '+json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
