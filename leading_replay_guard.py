"""Remove an evidenced source replay before a distinct, separated output turn.

This acoustic fallback does not certify the words or language of the remainder.
It never guesses a crop from duration, segment index, or a Japanese script test.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
from core import ROOT
from phonetic_replay import candidates,PROBE_POLICY

POLICY={'version':1,'method':'complete_own_reference_phone_match_then_distinct_utterance',
    'minimum_reference_score':PROBE_POLICY['minimum_target_score'],
    'minimum_specificity':PROBE_POLICY['minimum_target_uniqueness_margin'],
    'minimum_prefix_phone_coverage':.8,'minimum_remaining_phone_events':3,
    'utterance_count':2,'cut':'middle_of_verified_quiet_gap','cognates':'protect',
    'uncertain':'keep','content_verified':False,'asr_api_requests':0}


def propose(query,reference,kernel,groups):
    report={'policy':POLICY,'action':'keep','reason':'no_separate_reference_replay'}
    if len(groups)!=POLICY['utterance_count']:return report
    first,remaining=groups
    if first['end']>=remaining['start']:return report
    mid=(query['starts']+query['ends'])/2
    prefix=np.flatnonzero((mid>=first['start'])&(mid<=first['end']))
    rest=np.flatnonzero((mid>=remaining['start'])&(mid<=remaining['end']))
    if len(prefix)<3 or len(rest)<POLICY['minimum_remaining_phone_events']:return report
    # End-of-phone estimation may extend into a quiet region. Classify on the
    # midpoint, then insist the chosen complete-reference alignment stays here.
    hypotheses=candidates(reference,query,kernel)
    leading=[h for h in hypotheses if h['a']>=prefix[0] and h['b']<=prefix[-1]+1]
    trailing=[h for h in hypotheses if h['a']>=rest[0] and h['b']<=rest[-1]+1]
    if not leading:return report
    best=leading[0];other=max((h['score'] for h in trailing),default=0.)
    coverage=(best['b']-best['a'])/len(prefix)
    report.update(reference_match=best,remaining_reference_score=other,prefix_phone_coverage=coverage)
    if best['score']<POLICY['minimum_reference_score'] or coverage<POLICY['minimum_prefix_phone_coverage']:
        report['reason']='weak_or_partial_reference_replay';return report
    if best['score']-other<POLICY['minimum_specificity']:
        report['reason']='remaining_voice_too_similar_to_reference';return report
    # The first utterance must start with the reference, not just contain an
    # arbitrary matching word after meaningful unmatched speech.
    if best['a']!=prefix[0]:
        report['reason']='unexplained_prefix_before_reference';return report
    report.update(action='crop_candidate',reason='source_replay_followed_by_distinct_speech',
        gap_start=first['end'],gap_end=remaining['start'])
    return report


def inspect(source,references,destination,previous):
    from waveform_output_guard import protected_text
    if previous.get('action')=='cropped':return previous
    target=next((r for r in references if r.get('own')),None)
    groups=previous.get('utterances',[])
    if target is None or len(groups)!=2 or protected_text(target['original_text'],target['translated_text']):
        return {**previous,'leading_replay_check':{'policy':POLICY,'action':'skip','reason':'no_two_turn_non_cognate_case'}}
    try:
        from phonetic_output_guard import _LOCK,_get_runtime
        with _LOCK:
            encoder,kernel=_get_runtime();cache=ROOT/'work/phonetic-replay-features-v1'
            query=encoder.encode(source,cache);reference=encoder.encode(target['path'],cache)
            proposal=propose(query,reference,kernel,groups)
        if proposal['action']!='crop_candidate':return {**previous,'leading_replay_check':proposal}
        values,rate=sf.read(source,dtype='float32',always_2d=True)
        # voice_groups already verified this gap as quiet. The middle leaves
        # natural onset padding and keeps the complete remaining tail intact.
        cut=round((proposal['gap_start']+proposal['gap_end'])*.5*rate)
        if not 0<cut<len(values):raise ValueError('Invalid leading replay crop')
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.partial.wav')
        sf.write(temporary,values[cut:],rate,subtype='PCM_24');temporary.replace(destination)
        return {**previous,'action':'cropped','trimmed':True,'content_verified':False,
            'reason':'own_reference_replay_before_distinct_output','path':str(destination),
            'keep_start':cut/rate,'keep_end':len(values)/rate,'output_frames':len(values)-cut,
            'removed_seconds':cut/rate,'leading_replay_check':proposal}
    except Exception as error:
        return {**previous,'leading_replay_check':{'policy':POLICY,'action':'unavailable',
            'reason':'local_prefix_replay_check_failed','error_type':type(error).__name__}}
