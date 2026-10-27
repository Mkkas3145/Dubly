"""Conservative acoustic confidence and anchor-bounded CTC refinement.

Scores are diagnostics, not calibrated probabilities of timestamp correctness.
No LLM timestamps or new recognizer are used. Weak spans can receive fresh
local acoustics before conservative whole-utterance candidate selection.
"""
import math

VERSION=2
FRAME_SECONDS=.02


def describe(units,spans,emission,star,offset=0,duration=math.inf):
    words=[]
    # Exclude blank and wildcard from competing audible phonemes.
    best=emission[:,1:star].max(dim=-1).values
    for unit,tokens in zip(units,spans):
        if not tokens:raise ValueError('MMS returned an empty lexical alignment')
        logs=[];dominance=[];speech=[]
        for token in tokens:
            a,b=token.start,token.end
            selected=emission[a:b,token.token]
            logs.append(float(selected.mean()))
            dominance.append(float((selected-best[a:b]).clamp(max=0).exp().mean()))
            speech.append(float((best[a:b]>emission[a:b,0]).float().mean()))
        lexical=unit['roman']!='*'
        posterior=math.exp(sum(logs)/len(logs))
        minimum=math.exp(min(logs))
        competition=sum(dominance)/len(dominance)
        support=sum(speech)/len(speech)
        score=.6*posterior+.3*competition+.1*support
        reasons=[]
        if lexical:
            if posterior<.12:reasons.append('low_phoneme_likelihood')
            if minimum<.015:reasons.append('weak_individual_phoneme')
            if competition<.35:reasons.append('competing_phonemes')
        quality={'score':score,'posterior':posterior,'log_posterior':sum(logs)/len(logs),
                 'weakest_phoneme':minimum,'phoneme_dominance':competition,
                 'speech_support':support,'reasons':reasons,'lexical':lexical}
        words.append({'word':unit['word'],'start':(tokens[0].start+offset)*FRAME_SECONDS,
            'end':min(duration,(tokens[-1].end+offset)*FRAME_SECONDS),
            'score':float(sum(t.score for t in tokens)/len(tokens)),
            'alignment_quality':quality})
    if len(words)!=len(units):raise ValueError('MMS source token count mismatch')
    return words


def anchor(word):
    q=word['alignment_quality']
    return q['lexical'] and not q['reasons'] and q['posterior']>=.45 and q['phoneme_dominance']>=.65


def regions(words):
    """Only uncertain runs containing actual low-confidence words are retried."""
    start=0
    for end in range(len(words)+1):
        if end==len(words) or anchor(words[end]):
            if any(w['alignment_quality']['reasons'] for w in words[start:end]):yield start,end
            start=end+1


def improvement(old,new):
    if len(old)!=len(new) or [w['word'] for w in old]!=[w['word'] for w in new]:return False,'transcript_changed'
    if any(not 0<=w['start']<w['end'] for w in new):return False,'collapsed_candidate'
    a=[w['alignment_quality'] for w in old];b=[w['alignment_quality'] for w in new]
    # Every word is protected, even when it shares a retry with a weak neighbor.
    if any(y['score']<x['score']-.03 or y['log_posterior']<x['log_posterior']-.1 for x,y in zip(a,b)):
        return False,'neighbor_confidence_decreased'
    suspects=[i for i,x in enumerate(a) if x['reasons']]
    if not suspects:return False,'no_suspect'
    if any(b[i]['score']<.5 or b[i]['posterior']<.2 or b[i]['phoneme_dominance']<.6 for i in suspects):
        return False,'candidate_still_uncertain'
    gain=sum(b[i]['score']-a[i]['score'] for i in suspects)/len(suspects)
    log_gain=sum(b[i]['log_posterior']-a[i]['log_posterior'] for i in suspects)/len(suspects)
    if gain<.08 or log_gain<.35:return False,'insufficient_acoustic_gain'
    return True,'stronger_acoustic_evidence'


def refine(units,spans,tokenized,emission,aligner,star,duration,progress=lambda *a:None):
    words=describe(units,spans,emission,star,duration=duration)
    checks=[];pending=list(regions(words))
    for number,(lo,hi) in enumerate(pending):
        progress(number,len(pending))
        old=words[lo:hi]
        # Reliable neighboring words are immutable anchors on the original clock.
        left=spans[lo-1][-1].end if lo else 0
        right=spans[hi][0].start if hi<len(spans) else len(emission)
        check={'word_range':[lo,hi],'window':[left*FRAME_SECONDS,right*FRAME_SECONDS],
               'accepted':False,'before':old}
        checks.append(check)
        if any(u['roman']=='*' for u in units[lo:hi]):check['reason']='unvoiced_modifier';continue
        # Avoid a large unanchored re-search; retain the whole-recording result.
        if (right-left)*FRAME_SECONDS>45:check['reason']='insufficient_local_anchors';continue
        target=[[star],*tokenized[lo:hi],[star]]
        flat=[t for word in target for t in word]
        minimum=len(flat)+sum(a==b for a,b in zip(flat,flat[1:]))
        if right-left<minimum:check['reason']='insufficient_frames';continue
        try:
            local_emission=emission[left:right].clone()
            # MMS's default wildcard has log score zero. Leaving it unpenalized
            # would swallow the tail of even a strongly matching phoneme, making
            # every replacement one frame long. Let known acoustic evidence beat
            # the wildcard while still allowing unrelated audio around the text.
            local_emission[:,star]=local_emission[:,:star].max(dim=-1).values-math.log(2)
            local=aligner(local_emission,target)[1:-1]
            candidate=describe(units[lo:hi],local,emission[left:right],star,offset=left,duration=duration)
            accepted,reason=improvement(old,candidate)
            if any(w['start']<left*FRAME_SECONDS or w['end']>right*FRAME_SECONDS+1e-6 for w in candidate):
                accepted,reason=False,'anchor_boundary_crossed'
            if any(b['start']<a['end']-1e-6 for a,b in zip(candidate,candidate[1:])):
                accepted,reason=False,'candidate_overlap'
            check.update(accepted=accepted,reason=reason,candidate=candidate)
            if accepted:words[lo:hi]=candidate
        except (ValueError,RuntimeError,OSError) as error:
            # Confidence refinement is optional. A rejected candidate must never
            # discard a valid global alignment or fail an entire video.
            check.update(reason='candidate_alignment_failed',error=str(error))
    return words,{'version':VERSION,'checked_regions':len(checks),
                  'accepted_regions':sum(c['accepted'] for c in checks),'checks':checks}


def row_quality(words):
    quality=[w['alignment_quality'] for w in words if w.get('alignment_quality',{}).get('lexical')]
    if not quality:return {'status':'unscored'}
    suspects=[{'word':w['word'],'start':w['start'],'end':w['end'],
               'reasons':w['alignment_quality']['reasons']} for w in words if w.get('alignment_quality',{}).get('reasons')]
    return {'status':'uncertain' if suspects else 'supported','score':sum(q['score'] for q in quality)/len(quality),
            'weakest_word_score':min(q['score'] for q in quality),'suspect_words':suspects}


def refresh_uncertain(words,units,tokenized,row_ranges,emission_provider,aligner,star,duration,progress=lambda *a:None):
    """Fresh local acoustics, accepting complete utterances independently.

    An undecodable neighboring cry must not veto a recovered complete utterance.
    Text, confident anchors and every unaccepted word remain unchanged.
    """
    result=list(words);checks=[];pending=list(regions(words))
    for number,(lo,hi) in enumerate(pending):
        progress(number,len(pending))
        left=round(words[lo-1]['end']/FRAME_SECONDS) if lo else 0
        right=round(words[hi]['start']/FRAME_SECONDS) if hi<len(words) else max(1,int((round(duration*16000)-400)//320+1))
        check={'word_range':[lo,hi],'window':[left*FRAME_SECONDS,right*FRAME_SECONDS],
               'accepted':False,'utterances':[]};checks.append(check)
        groups=[(a,b) for a,b in row_ranges if lo<=a<b<=hi and any(w['alignment_quality']['reasons'] for w in result[a:b])]
        if not groups:check['reason']='no_complete_uncertain_utterance';continue
        if right<=left or (right-left)*FRAME_SECONDS>30:check['reason']='insufficient_local_anchors';continue
        if any(u['roman']=='*' for u in units[lo:hi]):check['reason']='unvoiced_modifier';continue
        target=[[star],*tokenized[lo:hi],[star]]
        flat=[t for word in target for t in word]
        if right-left<len(flat)+sum(a==b for a,b in zip(flat,flat[1:])):
            check['reason']='insufficient_frames';continue
        try:
            emission=emission_provider(left,right)
            local=emission.clone()
            local[:,star]=local[:,:star].max(dim=-1).values-math.log(2)
            spans=aligner(local,target)[1:-1]
            candidate=describe(units[lo:hi],spans,emission,star,offset=left,duration=duration)
            for a,b in groups:
                proposed=candidate[a-lo:b-lo]
                accepted,reason=improvement(result[a:b],proposed)
                lower=result[a-1]['end'] if a else 0
                upper=result[b]['start'] if b<len(result) else duration
                if any(w['start']<max(lower,left*FRAME_SECONDS)-1e-6 or
                       w['end']>min(upper,right*FRAME_SECONDS)+1e-6 for w in proposed):
                    accepted,reason=False,'neighbor_boundary_crossed'
                if any(y['start']<x['end']-1e-6 for x,y in zip(proposed,proposed[1:])):
                    accepted,reason=False,'candidate_overlap'
                entry={'word_range':[a,b],'before':result[a:b],'candidate':proposed,'accepted':accepted,'reason':reason}
                check['utterances'].append(entry)
                if accepted:result[a:b]=proposed;check['accepted']=True
            check['reason']='local_acoustics_reviewed'
        except (ValueError,RuntimeError,OSError) as error:
            check.update(reason='candidate_alignment_failed',error=str(error))
    return result,{'version':VERSION,'checked_regions':len(checks),
        'accepted_utterances':sum(u['accepted'] for c in checks for u in c['utterances']),'checks':checks}
