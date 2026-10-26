"""Join adjacent source turns using their final acoustic boundaries."""
from copy import deepcopy
from rich_transcript import join_words


def merge_corrected(rows, gap_seconds=.8):
    result=[];audit=[]
    for original in rows:
        row=deepcopy(original)
        row['merged_source_ids']=list(row.get('merged_source_ids',[row['id']]))
        prior=result[-1] if result else None
        gap=row['start']-prior['end'] if prior else None
        same_voice=prior is not None and str(prior.get('speaker')) not in ('unknown','None','') and prior.get('speaker')==row.get('speaker')
        eligible=same_voice and not any(r.get('keep_original') or r.get('deleted') for r in (prior,row))
        # A true overlap can contain another voice; retain it for review.
        if eligible and -1e-7<=gap<=gap_seconds+1e-7:
            old_end=prior['end']
            prior['end']=max(prior['end'],row['end'])
            if prior.get('words') and row.get('words'):
                prior['words']+=row['words']
                prior['text']=join_words(prior['words'])
            else:
                prior['text']=prior['text'].rstrip()+' '+row['text'].lstrip()
            prior['merged_source_ids']+=row['merged_source_ids']
            audit.append({'method':'same_speaker_corrected_gap_merge','source_ids':list(prior['merged_source_ids']),
                'previous_end':old_end,'following_start':row['start'],'corrected_gap_seconds':max(0.,gap),
                'start':prior['start'],'end':prior['end']})
        else:result.append(row)
    for i,row in enumerate(result):row['id']=i
    return result,audit
