"""Scribe output checking: keep intended dialogue, crop only verified copied extras."""
from __future__ import annotations
import hashlib,json,re,threading,unicodedata,shutil
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import numpy as np
import soundfile as sf
from core import SR,atomic_json
from rich_transcript import normalize as _normalize,spoken
from scribe import transcribe_words

MODEL='elevenlabs/scribe-v2'
POLICY={'version':3,'model':MODEL,'batch_size':16,'language':'automatic',
    'target':'sentence_semantics_with_lexical_coverage_and_cross_script_phonetic_protection',
    'crop':'only_reference_explained_extras_outside_unique_target',
    'cut':'quiet_gap_between_words','uncertain':'preserve_generated_audio_with_review_warning',
    'reject':'unusable_audio_or_positive_reference_replay_evidence',
    'cognates':'never_reject_on_script_alone','regeneration':False,
    'semantic_model':'sentence-transformers/LaBSE',
    'missing_short_asr':'both_verified_neighbors_and_one_bounded_vocalization'}
_ROMAN_LOCK=threading.Lock()
def normalize(text):
    # Kana spelling is not a different utterance (ワクワク / わくわく).
    return ''.join(chr(ord(c)-0x60) if '\u30a1'<=c<='\u30f6' else c for c in _normalize(text))
@lru_cache(maxsize=8192)
def phonetic(text):
    import uroman
    with _ROMAN_LOCK:
        if not hasattr(phonetic,'engine'):phonetic.engine=uroman.Uroman()
        value=phonetic.engine.romanize_string(spoken(text)).lower()
    value=''.join(c for c in unicodedata.normalize('NFKD',value) if 'a'<=c<='z' or c.isdigit())
    # General cross-script romanization differences, not dialogue-specific aliases.
    for a,b in [('yeo','o'),('eu','u'),('sh','s'),('sy','s')]:value=value.replace(a,b)
    return re.sub(r'(.)\1+',r'\1',value)
def spans(words,text,converter):
    target=converter(text)
    if not target:return []
    stream='';owners=[]
    for i,w in enumerate(words):
        value=converter(w.get('word',w.get('text','')));stream+=value;owners.extend([i]*len(value))
    result=[];offset=0
    while True:
        pos=stream.find(target,offset)
        if pos<0:break
        end=pos+len(target);offset=pos+1
        # Never take just a syllable out of a larger recognized word.
        if pos and owners[pos-1]==owners[pos]:continue
        if end<len(owners) and owners[end-1]==owners[end]:continue
        pair=(owners[pos],owners[end-1])
        if pair not in result:result.append(pair)
    return result
def identify(words,segment):
    expected=segment['translated_text'];candidates=spans(words,expected,normalize)
    if not candidates:candidates=spans(words,expected,phonetic)
    original=segment['original_text']
    # Matching source Japanese is valid only if target pronunciation is similar.
    a,b=phonetic(original),phonetic(expected)
    if not candidates and a and b and SequenceMatcher(None,a,b).ratio()>=.82:
        candidates=spans(words,original,normalize)
    return candidates
def covered_extras(words,segment):
    if not words:return True
    covered=set()
    references=list(segment.get('_guard_reference_texts',[]))
    a,b=phonetic(segment['original_text']),phonetic(segment['translated_text'])
    # A distinct Japanese rendition of this very line can precede the Korean one.
    # Cognates are deliberately excluded so "spy/mission" cannot be deleted as foreign.
    if a and b and SequenceMatcher(None,a,b).ratio()<.6:
        references.append({'original_text':segment['original_text'],'translated_text':''})
    for part in references:
        for field in ('original_text','translated_text'):
            for lo,hi in spans(words,part[field],normalize):covered.update(range(lo,hi+1))
    lexical={i for i,w in enumerate(words) if normalize(w.get('word',''))}
    return bool(lexical) and lexical<=covered

def replay_evidence(words,segment):
    """A missed target alone proves nothing. Require substantial donor speech.

    This only vetoes an output; it can never authorize deleting audio. A much
    longer recognized output containing a distinct donor phrase is evidence of
    reference replay, even when ASR renders Japanese in Korean characters.
    """
    heard=' '.join(w.get('word','') for w in words)
    wanted=normalize(segment['translated_text']);actual=normalize(heard)
    if not wanted or len(actual)<max(8,3*len(wanted)):return None
    if SequenceMatcher(None,phonetic(heard),phonetic(segment['translated_text'])).ratio()>=.45:return None
    for part in segment.get('_guard_reference_texts',[]):
        for field in ('original_text','translated_text'):
            text=part.get(field,'')
            # Do not treat an equivalent requested line as a foreign donor.
            if SequenceMatcher(None,phonetic(text),phonetic(segment['translated_text'])).ratio()>=.6:continue
            for converter,minimum in ((normalize,4),(phonetic,6)):
                a,b=converter(heard),converter(text)
                match=SequenceMatcher(None,a,b,autojunk=False).find_longest_match()
                if match.size>=max(minimum,2*len(converter(segment['translated_text']))):
                    return {'reference_field':field,'matched_units':match.size,
                        'comparison':'lexical' if converter is normalize else 'phonetic',
                        'recognized_units':len(actual),'requested_units':len(wanted)}
    return None
def quiet_cut(audio,rate,a,b):
    lo=max(0,round(a*rate));hi=min(len(audio),round(b*rate))
    if hi-lo<3:return None
    values=audio[lo:hi].mean(axis=1)
    width=min(len(values),max(1,round(rate*.01)));step=max(1,round(rate*.005))
    centers=np.arange(width//2,max(width//2+1,len(values)-width//2),step)
    energy=np.array([np.mean(values[max(0,c-width//2):min(len(values),c+width//2+1)]**2) for c in centers])
    if np.sqrt(energy.min())>max(.001,float(np.sqrt(energy.max()))*.1):return None
    quiet=energy<=np.quantile(energy,.4);runs=[];begin=None
    for i,v in enumerate([*quiet,False]):
        if v and begin is None:begin=i
        elif not v and begin is not None:runs.append((begin,i));begin=None
    x,y=max(runs,key=lambda pair:pair[1]-pair[0]);guess=lo+int(centers[(x+y-1)//2])
    left=max(lo,guess-round(rate*.002));right=min(hi,guess+round(rate*.002)+1)
    return left+int(np.argmin(np.abs(audio[left:right].mean(axis=1))))
def inspect_output(source,words,segment,destination):
    """No Japanese-character deletion rule; only verified extra-reference text."""
    candidates=identify(words,segment)
    report={'policy':POLICY,'recognized_text':' '.join(w.get('word','') for w in words),
        'expected_text':segment['translated_text'],'target_candidates':[list(c) for c in candidates]}
    if not candidates:
        # A whole Korean utterance may be rendered in katakana by multilingual ASR.
        # Permit a conservative whole-output KEEP only; this cannot authorize a cut.
        expected=phonetic(segment['translated_text']);heard=phonetic(''.join(w.get('word','') for w in words))
        def skeleton(s):return re.sub('[aeiou]','',s.translate(str.maketrans({'g':'k','d':'t','b':'p','z':'s','l':'r'})))
        if (len(expected)>=4 and .8<=len(heard)/len(expected)<=1.25 and
            skeleton(expected)==skeleton(heard) and SequenceMatcher(None,expected,heard).ratio()>=.6):
            return {**report,'action':'keep','trimmed':False,'reason':'protected_cross_script_phonetic_equivalent'}
        if words:
            from output_sentence_similarity import locate
            candidates,evidence=locate(words,segment,covered_extras)
            report.update(sentence_similarity=evidence,target_candidates=[list(c) for c in candidates])
    if len(candidates)!=1:
        replay=replay_evidence(words,segment) if not candidates else None
        if replay:return {**report,'action':'reject','reason':'reference_replay_without_target','replay_evidence':replay}
        return {**report,'action':'keep_uncertain','reason':'target_missing_or_ambiguous'}
    first,last=candidates[0];prefix=words[:first];suffix=words[last+1:]
    if not prefix and not suffix:return {**report,'action':'keep','trimmed':False}
    if not covered_extras(prefix,segment) or not covered_extras(suffix,segment):
        return {**report,'action':'keep_uncertain','reason':'unexplained_extra_words'}
    audio,rate=sf.read(source,dtype='float32',always_2d=True)
    lo=0;hi=len(audio)
    if prefix:
        lo=quiet_cut(audio,rate,prefix[-1]['end'],words[first]['start'])
        if lo is None:return {**report,'action':'reject','reason':'no_safe_prefix_gap','verified_reference_extras':True}
    if suffix:
        hi=quiet_cut(audio,rate,words[last]['end'],suffix[0]['start'])
        if hi is None:return {**report,'action':'reject','reason':'no_safe_suffix_gap','verified_reference_extras':True}
    if hi<=lo:return {**report,'action':'keep_uncertain','reason':'invalid_crop'}
    destination=Path(destination);temp=destination.with_suffix('.partial.wav')
    sf.write(temp,audio[lo:hi],rate,subtype='PCM_24');temp.replace(destination)
    return {**report,'action':'cropped','trimmed':True,'path':str(destination),
        'keep_start':lo/rate,'keep_end':hi/rate,'removed_seconds':(len(audio)-hi+lo)/rate,
        'source_frames':len(audio),'output_frames':hi-lo,'sample_rate':rate}
def short_vocalization(source,reference,expected):
    """Waveform supports a brief single utterance, not its lexical content."""
    from timing import envelope,measured_energy_gate,runs,HOP
    units=len(_normalize(expected))
    if not 0<units<=4 or reference is None:return None
    def layout(path):
        values,rate=sf.read(path,dtype='float32',always_2d=True)
        if rate!=SR or not len(values):return None
        levels=envelope(values)
        if not len(levels) or float(levels.max())<1e-5:return None
        gate,_=measured_energy_gate(levels,.035,values)
        spans=[(a,b) for a,b in runs(levels>=gate) if b-a>=3]
        return values,spans
    original=layout(reference);generated=layout(source)
    if original is None or generated is None or not original[1] or not generated[1]:return None
    ref_duration=(original[1][-1][1]-original[1][0][0])*HOP/SR
    merged=[];gap=max(.03,min(.10,ref_duration*.3))
    for a,b in generated[1]:
        if merged and (a-merged[-1][1])*HOP/SR<=gap:merged[-1][1]=b
        else:merged.append([a,b])
    if len(merged)!=1:return None
    duration=(merged[0][1]-merged[0][0])*HOP/SR
    if duration>max(ref_duration*2.5,.2*units+.25):return None
    return {'vocal_start':merged[0][0]*HOP/SR,'vocal_end':min(len(generated[0])/SR,merged[0][1]*HOP/SR),
        'reference_vocal_seconds':ref_duration,'vocal_seconds':duration,'vocal_groups':1,
        'content_verified':False,'boundary_source':'known_individual_output_slot_and_two_verified_neighbors'}
def check_batch(prepared,cfg,folder,progress=lambda *args:None):
    pending=[p for p in prepared if 'cached' not in p]
    usable=[]
    for p in pending:
        # A corrupt individual file must not poison all sixteen outputs or
        # consume an ASR request. Inspect waveform validity independently of STT.
        try:
            peak=0.
            with sf.SoundFile(p['generated']) as audio:
                if audio.samplerate!=SR or not audio.frames:raise ValueError('format')
                for block in audio.blocks(blocksize=SR,dtype='float32',always_2d=True):
                    if not np.isfinite(block).all():raise ValueError('nonfinite')
                    peak=max(peak,float(np.max(np.abs(block))))
            if peak<1e-5:raise ValueError('silent')
        except (RuntimeError,ValueError,OSError):
            report={'policy':POLICY,'action':'reject','reason':'unusable_generated_audio'}
            p['_guard_error']=report['reason'];p['generation']['tts_output_guard_policy']=POLICY
            p['generation']['tts_output_check']=report
            atomic_json(Path(p['folder'])/f'{p["index"]:04d}-output-check.json',report)
        else:usable.append(p)
    pending=usable
    if not pending:return prepared
    identity={'policy':POLICY,'parts':[{'index':p['index'],
        'sha256':hashlib.sha256(Path(p['generated']).read_bytes()).hexdigest(),
        'original_text':p['segment']['original_text'],'translated_text':p['segment']['translated_text'],
        'references':p['segment'].get('_guard_reference_texts',[])} for p in pending]}
    digest=hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    root=Path(folder);folder=root/digest;folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'input.json',identity)
    batch=folder/'generated.wav';mapping=[];cursor=0
    if not batch.exists():
        with sf.SoundFile(batch.with_suffix('.partial.wav'),'w',samplerate=SR,channels=1,subtype='PCM_16') as out:
            for p in pending:
                data,rate=sf.read(p['generated'],dtype='float32',always_2d=True)
                if rate!=SR:raise RuntimeError('TTS validation input sample rates do not match.')
                out.write(data.mean(axis=1));out.write(np.zeros(SR,dtype='float32'))
        batch.with_suffix('.partial.wav').replace(batch)
    for p in pending:
        frames=sf.info(p['generated']).frames
        mapping.append((p,cursor/SR,(cursor+frames)/SR));cursor+=frames+SR
    progress('checking',len(pending))
    # Guard rules may change without changing audio. Reuse paid Scribe results by
    # actual input bytes, including caches produced by older verification policies.
    batch_sha=hashlib.sha256(batch.read_bytes()).hexdigest()
    recognition=root/'scribe-audio'/batch_sha;recognition.mkdir(parents=True,exist_ok=True)
    if not (recognition/'scribe.json').exists():
        for legacy in root.glob('*/scribe/scribe.json'):
            old=json.loads(legacy.read_text('utf8'));signature=old.get('signature',{})
            if (signature.get('source_sha256')!=batch_sha or signature.get('model')!=MODEL or
                signature.get('language') is not None):continue
            for name in ('scribe.json','scribe-raw.json'):
                if (legacy.parent/name).exists():shutil.copy2(legacy.parent/name,recognition/name)
            break
    try:
        asr=transcribe_words(batch,recognition,cfg['key'],language=None,attempts=1,model=MODEL)
    except Exception as error:
        # Isolate provider failure; don't fail the video's other stages.
        atomic_json(folder/'error.json',{'type':type(error).__name__,'message':str(error)[:500]})
        for p in pending:
            report={'policy':POLICY,'action':'keep_uncertain','reason':'scribe_output_check_failed'}
            p['generation']['tts_output_guard_policy']=POLICY;p['generation']['tts_output_check']=report
            atomic_json(Path(p['folder'])/f'{p["index"]:04d}-output-check.json',report)
        return prepared
    for p,a,b in mapping:
        words=[];crossed=False
        for w in asr['words']:
            if not a<=((w['start']+w['end'])/2)<b:continue
            if w['start']<a-.03 or w['end']>b+.03:crossed=True
            words.append({**w,'start':max(0,w['start']-a),'end':min(b-a,w['end']-a)})
        destination=Path(p['folder'])/f'{p["index"]:04d}-guarded.wav'
        try:
            # Exact individual-file boundaries outrank drifting ASR timestamps.
            report=inspect_output(p['generated'],words,p['segment'],destination)
            if crossed:report['asr_boundary_clipped']=True
        except Exception as error:
            report={'policy':POLICY,'action':'keep_uncertain','reason':'output_check_processing_failed','error_type':type(error).__name__}
        atomic_json(Path(p['folder'])/f'{p["index"]:04d}-output-check.json',report)
        p['generation']['tts_output_guard_policy']=POLICY;p['generation']['tts_output_check']=report
        if report['action']=='cropped':p['generated']=destination
        elif report['action']=='reject':p['_guard_error']=report['reason']
    # A missed short interjection has a known individual output file. Infer only
    # its occupied slot from verified neighbors; never invent an ASR word or cut it.
    for i,(p,a,b) in enumerate(mapping):
        report=p['generation'].get('tts_output_check',{})
        if (i==0 or i==len(mapping)-1 or report.get('recognized_text','').strip() or
            report.get('reason')!='target_missing_or_ambiguous'):continue
        left=mapping[i-1][0]['generation'].get('tts_output_check',{})
        right=mapping[i+1][0]['generation'].get('tts_output_check',{})
        if left.get('action') not in ('keep','cropped') or right.get('action') not in ('keep','cropped'):continue
        evidence=short_vocalization(p['generated'],p.get('reference'),p['segment']['translated_text'])
        if evidence is None:continue
        report={**report,'action':'keep','trimmed':False,
            'reason':'inferred_short_vocalization_from_verified_neighbors','neighbor_inference':evidence,
            'neighbor_indices':[mapping[i-1][0]['index'],mapping[i+1][0]['index']]}
        p['generation']['tts_output_check']=report;p.pop('_guard_error',None)
        atomic_json(Path(p['folder'])/f'{p["index"]:04d}-output-check.json',report)
    return prepared
