"""Verify omitted short leading words in unclaimed audio instead of guessing their times."""
from pathlib import Path
import unicodedata
import soundfile as sf
from core import SR,atomic_json
from timing import envelope,measured_energy_gate,runs,HOP

POLICY={'version':1,'method':'collapsed_first_word_unclaimed_island_exact_asr_verification'}

def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())

def repair(clean,rows,words,folder,cfg,language=None):
    from scribe import transcribe_words
    result=[dict(r) for r in rows];audit=[]
    folder=Path(folder)
    with sf.SoundFile(clean) as audio:
        for i,row in enumerate(result):
            if i==0:continue
            a=row['start'];previous=result[i-1]['end']
            lexical=[w for w in words if normalized(w.get('word',w.get('text',''))) and abs(w['start']-a)<=.15]
            if not lexical:continue
            first=min(lexical,key=lambda w:w['start']);word=first.get('word',first.get('text',''))
            if first['end']-first['start']>.04 or not normalized(row.get('text','')).startswith(normalized(word)):continue
            lo=max(previous+.08,a-2.);hi=a-.15
            if hi-lo<.2:continue
            # Include the anchored phrase when estimating the noise gate; the
            # unclaimed gap alone may consist mostly of low-level separator noise.
            analysis_end=min(audio.frames/SR,max(hi,a+.6))
            audio.seek(round(lo*SR));values=audio.read(round((analysis_end-lo)*SR),dtype='float32',always_2d=True)
            levels=envelope(values);gate,_=measured_energy_gate(levels,.015,values)
            levels=levels[:int((hi-lo)*SR/HOP)]
            spans=[]
            for x,y in runs(levels>gate):
                if spans and (x-spans[-1][1])*HOP/SR<=.12:spans[-1]=(spans[-1][0],y)
                else:spans.append((x,y))
            islands=[(lo+x*HOP/SR,lo+y*HOP/SR) for x,y in spans if (y-x)*HOP/SR>=.06
                and x*HOP/SR>=.04 and (hi-lo)-y*HOP/SR>=.08]
            if len(islands)!=1:continue
            start,end=islands[0];crop_start=max(lo,start-.04);crop_end=min(hi,end+.04)
            stage=folder/f'{i:04d}-{round(start*SR)}-{round(end*SR)}';stage.mkdir(parents=True,exist_ok=True)
            candidate=stage/'candidate.wav';audio.seek(round(crop_start*SR))
            sf.write(candidate,audio.read(round((crop_end-crop_start)*SR),dtype='float32',always_2d=True),SR,subtype='PCM_16')
            try:checked=transcribe_words(candidate,stage/'verification',cfg['key'],language=language)
            except RuntimeError as error:
                audit.append({'source_index':i,'action':'keep_original_boundary','reason':str(error)});continue
            recognized=normalized(checked.get('text',''))
            if not recognized:recognized=normalized(''.join(w.get('word','') for w in checked.get('words',[])))
            confident=all(w.get('probability',1.)>=.85 for w in checked.get('words',[]) if normalized(w.get('word','')))
            accepted=recognized==normalized(word) and confident
            evidence={'source_index':i,'original_start':a,'candidate_start':start,'candidate_end':end,
                'expected_word':word,'recognized_text':checked.get('text'),'verified':accepted,'policy':POLICY}
            if accepted:
                row['start']=max(previous,round((start-.02)*SR)/SR)
                evidence['start']=row['start']
            audit.append(evidence)
    return result,audit
