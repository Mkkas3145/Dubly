"""Utterance candidates from MMS acoustics, selected jointly in script order.

No recognized text or waveform is rewritten. Confidence is a ranking signal,
not a calibrated probability. Unsupported/ambiguous moves retain the baseline.
"""
import math
from mms_alignment_quality import describe,FRAME_SECONDS
VERSION=5


def metrics(words,units):
    weights=[max(1,len(u['roman'])) for u in units]
    total=sum(weights)
    def mean(key):return sum(w['alignment_quality'][key]*n for w,n in zip(words,weights))/total
    posterior=math.exp(mean('log_posterior'))
    dominance=mean('phoneme_dominance');speech=mean('speech_support')
    # The complete utterance matters; a distorted final phoneme need not veto
    # otherwise matching speech, but cannot alone justify a move either.
    return {'score':.6*posterior+.25*dominance+.15*speech,
            'posterior':posterior,'log_posterior':mean('log_posterior'),
            'dominance':dominance,'speech':speech}


def reliable(words,units):
    q=metrics(words,units)
    bad=sum(bool(w['alignment_quality']['reasons']) for w in words)
    return q['posterior']>=.3 and q['dominance']>=.75 and bad==0


def windows(left,right,start,end,phonemes=None):
    choices=[(left,right)]
    for margin in (1.,3.):choices.append((max(left,start-margin),min(right,end+margin)))
    # Search the anchor interval, not just the potentially incorrect baseline.
    if right-left>8:
        cursor=left
        while cursor<right:
            choices.append((cursor,min(right,cursor+8)))
            cursor+=4
    if phonemes is not None:
        width=max(2.,min(8.,1.2+.15*phonemes))
        cursor=left
        while cursor<right:
            choices.append((cursor,min(right,cursor+width)))
            cursor+=width/2
    result=[]
    for a,b in choices:
        a=round(a/FRAME_SECONDS);b=round(b/FRAME_SECONDS)
        if 0<b-a<=1500 and (a,b) not in result:result.append((a,b))
    return result


def sequence_scores(options):
    """Exact DAG search: weak early guesses cannot prune the valid full path.

    Forward/backward scores also measure ambiguity at EVERY dialogue, including
    alternatives that converge on the same final dialogue.
    """
    if not options:return [],[],0.
    forward=[];parents=[]
    for i,choices in enumerate(options):
        scores=[];links=[]
        for c in choices:
            if not i:score,parent=0.,None
            else:
                valid=[(forward[i-1][k],k) for k,p in enumerate(options[i-1])
                       if p['end']<=c['start']+1e-6 and math.isfinite(forward[i-1][k])]
                score,parent=max(valid,default=(-math.inf,None))
            scores.append(score+c['utility']);links.append(parent)
        forward.append(scores);parents.append(links)
    best=max(forward[-1],default=-math.inf)
    if not math.isfinite(best):return None,None,best
    index=max(range(len(forward[-1])),key=forward[-1].__getitem__);indices=[]
    for i in range(len(options)-1,-1,-1):
        indices.append(index);index=parents[i][index]
    indices.reverse()
    backward=[[0.]*len(options[-1])]
    for i in range(len(options)-2,-1,-1):
        tail=backward[-1];scores=[]
        for c in options[i]:
            scores.append(max((p['utility']+tail[k] for k,p in enumerate(options[i+1])
                if c['end']<=p['start']+1e-6),default=-math.inf))
        backward.append(scores)
    backward.reverse()
    totals=[[a+b for a,b in zip(f,g)] for f,g in zip(forward,backward)]
    return [options[i][j] for i,j in enumerate(indices)],totals,best


def select_paths(options,beam_size=24,all_paths=False):
    # Compatibility wrapper; inference itself no longer uses a pruned beam.
    path,_,score=sequence_scores(options)
    return [(score,path,path[-1]['end'] if path else 0.)] if all_paths and path is not None else path


def order_margin(evidence,tokens,aligner,star):
    """Measure phoneme-order specificity as a soft ranking signal, not a veto."""
    flat=[t for group in tokens for t in group]
    alternatives=[]
    for seq in (list(reversed(flat)),flat[1:]+flat[:1]):
        if seq!=flat and seq not in alternatives:alternatives.append(seq)
    if not alternatives:return None
    local=evidence.clone();local[:,star]=local[:,:star].max(dim=-1).values-math.log(2)
    def score(seq):
        spans=aligner(local,[[star],seq,[star]])[1]
        return sum(float(evidence[t.start:t.end,t.token].mean()) for t in spans)/len(spans)
    actual=score(flat)
    return actual-max(score(seq) for seq in alternatives)


def same_location(a,b):
    overlap=max(0.,min(a['end'],b['end'])-max(a['start'],b['start']))
    union=max(a['end'],b['end'])-min(a['start'],b['start'])
    # Disagreement on a weak first phoneme is boundary uncertainty, not a
    # different word occurrence, when the same vocal body/end recurs.
    return union>0 and overlap/union>=.5 and (abs(a['start']-b['start'])<=.12 or abs(a['end']-b['end'])<=.16)


def _search_ranges(words,units,tokenized,row_ranges,provider,aligner,star,duration,views=('clean',),progress=lambda *a:None,acoustic_support=None):
    baseline=[words[a:b] for a,b in row_ranges]
    locked=[reliable(ws,units[a:b]) for ws,(a,b) in zip(baseline,row_ranges)]
    checks=[];options=[];cache={}
    def support(start,end):
        if acoustic_support is None:return 0.
        try:return max(0.,min(1.,float(acoustic_support(start,end))))
        except (ValueError,RuntimeError,OSError):return 0.
    def acoustics(view,left,right):
        # Bounded cache: original recordings can run for hours.
        key=(view,left,right)
        if key not in cache:
            if len(cache)>=8:cache.pop(next(iter(cache)))
            cache[key]=provider(left,right,view)
        return cache[key]
    for i,((a,b),old) in enumerate(zip(row_ranges,baseline)):
        progress(i,len(row_ranges))
        if not old:raise ValueError('Empty source utterance')
        q=metrics(old,units[a:b])
        base={'start':old[0]['start'],'end':old[-1]['end'],'words':old,'quality':q,
              'utility':q['score']+.12*support(old[0]['start'],old[-1]['end']),'baseline':True,'view':'baseline'}
        choices=[base];check={'row':i,'locked':locked[i],'candidates':[]};checks.append(check)
        if locked[i] or not any(w['alignment_quality']['reasons'] for w in old):
            options.append(choices);continue
        left=next((baseline[j][-1]['end'] for j in range(i-1,-1,-1) if locked[j]),0.)
        right=next((baseline[j][0]['start'] for j in range(i+1,len(baseline)) if locked[j]),duration-.025)
        # A long unanchored region must not trigger a recording-wide combinatorial
        # search. Global acoustics remain the baseline; local views are bounded.
        if right-left>30:
            left=max(left,base['start']-8);right=min(right,base['end']+8)
        if any(u['roman']=='*' for u in units[a:b]):options.append(choices);continue
        targets=[[star],*tokenized[a:b],[star]]
        flat=[t for group in targets for t in group]
        minimum=len(flat)+sum(x==y for x,y in zip(flat,flat[1:]))
        proposals=[]
        def evaluate(lo,hi,view):
            if hi-lo<minimum:return
            try:
                evidence=acoustics(view,lo,hi)
                local=evidence.clone()
                local[:,star]=local[:,:star].max(dim=-1).values-math.log(2)
                spans=aligner(local,targets)[1:-1]
                candidate=describe(units[a:b],spans,evidence,star,offset=lo,duration=duration)
                m=metrics(candidate,units[a:b])
                start,end=candidate[0]['start'],candidate[-1]['end']
                if start<left-1e-6 or end>right+1e-6:return
                if any(y['start']<x['end']-1e-6 for x,y in zip(candidate,candidate[1:])):return
                gain=m['score']-q['score'];log_gain=m['log_posterior']-q['log_posterior']
                # Quiet but repeatable words are provisional evidence, not an
                # immediate rejection for narrowly missing an absolute gate.
                plausible=m['posterior']>=.08 and m['dominance']>=.6 and m['speech']>=.15
                if not plausible or gain<.05 or log_gain<.25:return
                margin=order_margin(evidence,tokenized[a:b],aligner,star)
                entry={'start':start,'end':end,'words':candidate,'quality':m,
                       'utility':m['score']+.02*max(-1.,min(1.,margin or 0.))+.12*support(start,end),
                       'acoustic_contrast':support(start,end),
                       'baseline':False,'view':view,'window':[lo*.02,hi*.02],
                       'order_margin':margin}
                proposals.append(entry)
            except (RuntimeError,ValueError,OSError) as error:
                check.setdefault('errors',[]).append(str(error))
        visited=set()
        def inspect_window(lo,hi,view):
            key=(view,lo,hi)
            if key not in visited:
                visited.add(key);evaluate(lo,hi,view)
        for lo,hi in windows(left,right,base['start'],base['end'],sum(len(t) for t in tokenized[a:b])):
            for view in views:inspect_window(lo,hi,view)
        # Recheck promising locations under small changes to acoustic context.
        # Neither the time nor the spelling of a particular word is special.
        seeds=[]
        for c in sorted(proposals,key=lambda c:c['utility'],reverse=True):
            if any(abs(c['start']-p['start'])<.25 and abs(c['end']-p['end'])<.3 for p in seeds):continue
            seeds.append(c)
            if len(seeds)==3:break
        for c in seeds:
            lo,hi=[round(t/.02) for t in c['window']]
            for margin in (10,20):
                inspect_window(max(round(left/.02),lo-margin),min(round(right/.02),hi+margin),c['view'])
        # Agreement across independent acoustic views/windows is corroboration.
        # Alternatives at a different location must win clearly; near-identical
        # boundaries do not count as competing hypotheses.
        for c in proposals:
            agrees=[p for p in proposals if same_location(c,p)
                    and (p['view'],p['window'])!=(c['view'],c['window'])]
            c['consensus']=len(agrees);c['utility']+=.03*min(2,len(agrees))
        proposals.sort(key=lambda c:c['utility'],reverse=True)
        check['provisional_candidates']=[{k:v for k,v in c.items() if k!='words'} for c in proposals[:12]]
        if proposals:
            # Ambiguity is resolved in the full ordered path, not by comparing
            # independent row maxima which may belong to later repeated lines.
            for c in proposals:
                strong=c['quality']['posterior']>=.2 and c['quality']['speech']>=.3
                if not strong and c['consensus']<2:continue
                if any(abs(c['start']-p['start'])<.04 and abs(c['end']-p['end'])<.04 for p in choices):continue
                choices.append(c)
                if len(choices)>=9:break
        check['candidates']=[{k:v for k,v in c.items() if k!='words'} for c in choices]
        options.append(choices)
    picked,totals,best=sequence_scores(options)
    if picked is None:return words,{'version':VERSION,'accepted_utterances':0,'reason':'no_consistent_path','checks':checks}
    ambiguous=[]
    for i,c in enumerate(picked):
        if c['baseline']:continue
        rival=max((totals[i][j] for j,p in enumerate(options[i]) if
            not same_location(p,c)),default=-math.inf)
        if best-rival<.04:ambiguous.append(i)
    for i in ambiguous:
        options[i]=[options[i][0]];checks[i]['reason']='ambiguous_complete_paths'
    if ambiguous:picked=select_paths(options)
    if picked is None:return words,{'version':VERSION,'accepted_utterances':0,'reason':'ambiguous_paths','checks':checks}
    result=[];accepted=[]
    for i,c in enumerate(picked):
        result.extend(c['words'])
        if not c['baseline']:accepted.append({'row':i,'before':[baseline[i][0]['start'],baseline[i][-1]['end']],
                **{k:v for k,v in c.items() if k not in ('words','baseline')}})
    return result,{'version':VERSION,'accepted_utterances':len(accepted),'accepted':accepted,'checks':checks}


def anchor_ranges(words,units,row_ranges):
    """Partition only the alignment search, never Gemini text or editor rows.

    Reliable INTERNAL words are anchors too. A merged multi-sentence dialogue
    must not hide an uncertain prefix behind its otherwise well-aligned body.
    """
    groups=[];owners=[]
    for owner,(a,b) in enumerate(row_ranges):
        cursor=a
        for i in range(a,b):
            if reliable(words[i:i+1],units[i:i+1]):
                if cursor<i:groups.append((cursor,i));owners.append(owner)
                groups.append((i,i+1));owners.append(owner);cursor=i+1
        if cursor<b:groups.append((cursor,b));owners.append(owner)
    return groups,owners


def search(words,units,tokenized,row_ranges,provider,aligner,star,duration,views=('clean',),progress=lambda *a:None,acoustic_support=None):
    groups,owners=anchor_ranges(words,units,row_ranges)
    result,audit=_search_ranges(words,units,tokenized,groups,provider,aligner,star,duration,
        views=views,progress=progress,acoustic_support=acoustic_support)
    audit['original_utterances']=len(row_ranges);audit['search_groups']=len(groups)
    accepted=audit.get('accepted',[])
    audit['accepted_groups']=len(accepted)
    for item in [*audit.get('checks',[]),*accepted]:
        group=item['row'];item['search_group']=group;item['word_range']=list(groups[group]);item['row']=owners[group]
    audit['accepted_utterances']=len({item['row'] for item in accepted})
    return result,audit
