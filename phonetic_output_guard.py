"""Local phonetic replay inspection; no cloud recognition or TTS requests."""
from pathlib import Path
from difflib import SequenceMatcher
import gc,threading
import soundfile as sf
from core import ROOT
from phonetic_replay import PhoneticEncoder,phone_kernel,propose_crop,PROBE_POLICY

POLICY={**PROBE_POLICY,'production_enabled':True,'runtime_version':1,
    'load':'lazy_once_per_worker','inference':'serialized_shared_model',
    'feature_cache':'content_addressed','failure':'preserve_waveform_inspection_result'}
_LOCK=threading.Lock()
_runtime=None


def eligible(references):
    from tts_output_guard import phonetic
    target=next((r for r in references if r.get('own')),None)
    if target is None or len(references)<2:return False
    a,b=phonetic(target['original_text']),phonetic(target['translated_text'])
    return bool(a and b and SequenceMatcher(None,a,b,autojunk=False).ratio()>=POLICY['minimum_complete_target_phonetic_ratio'])


def _get_runtime():
    global _runtime
    if _runtime is None:
        import torch
        device='cuda' if torch.cuda.is_available() else 'cpu'
        encoder=PhoneticEncoder(ROOT/'models/phonetic-xlsr',device=device)
        _runtime=(encoder,phone_kernel(encoder.vocab))
    return _runtime


def release_runtime():
    """Called after the inspection queue has drained, before final rendering."""
    global _runtime
    with _LOCK:
        if _runtime is None:return
        encoder=_runtime[0];torch=encoder.torch;device=encoder.device
        _runtime=None;del encoder;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()


def inspect(source,references,destination,previous):
    if previous.get('action')=='cropped':return previous
    if not eligible(references):
        return {**previous,'phonetic_check':{'policy':POLICY,'action':'skip',
            'reason':'no_complete_phonetic_target_or_no_donor'}}
    try:
        with _LOCK:
            encoder,kernel=_get_runtime()
            cache=ROOT/'work/phonetic-replay-features-v1'
            query=encoder.encode(source,cache)
            bank=[{**r,'features':encoder.encode(r['path'],cache)} for r in references]
            proposal=propose_crop(source,query,bank,kernel)
            proposal['policy']=POLICY
        if proposal.get('action')!='crop_candidate':return {**previous,'phonetic_check':proposal}
        values,rate=sf.read(source,dtype='float32',always_2d=True)
        lo=round(proposal['keep_start']*rate);hi=round(proposal['keep_end']*rate)
        if not 0<=lo<hi<=len(values):raise ValueError('Invalid phonetic crop bounds')
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.partial.wav')
        sf.write(temporary,values[lo:hi],rate,subtype='PCM_24');temporary.replace(destination)
        return {**previous,'action':'cropped','trimmed':True,
            'reason':'phonetic_target_and_reference_explained_extras','path':str(destination),
            'keep_start':lo/rate,'keep_end':hi/rate,'output_frames':hi-lo,
            'removed_seconds':(len(values)-hi+lo)/rate,'phonetic_check':proposal}
    except Exception as error:
        return {**previous,'phonetic_check':{'policy':POLICY,'action':'unavailable',
            'reason':'local_phonetic_comparison_failed','error_type':type(error).__name__}}
