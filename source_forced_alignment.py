"""Align Gemini's unchanged source script; no recognizer or guessed final times."""
from pathlib import Path
import hashlib,json,math,unicodedata
import numpy as np
import soundfile as sf
from core import atomic_json
from local_resources import local_stage
from rich_transcript import spoken
from mms_alignment import align_batch,MODEL
from mms_alignment_quality import row_quality
from source_voice_release import POLICY as RELEASE_POLICY,recover_ends

def audio_digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def key(text):return "".join(c.casefold() for c in unicodedata.normalize("NFKC",text) if c.isalnum())

VERSION=17
LEXICAL_VERSION=16



def measured_rows(rows,words,duration,offset=0.):
    targets=[key(spoken(r['text'])) for r in rows]
    if any(not t for t in targets):raise ValueError('Source alignment requires audible lexical text.')
    if ''.join(targets)!=''.join(key(w['word']) for w in words):
        raise ValueError('Source forced alignment did not preserve the supplied transcript.')
    owners=[i for i,w in enumerate(words) for _ in key(w['word'])]
    cursor=0;result=[]
    for row,target in zip(rows,targets):
        ids=list(dict.fromkeys(owners[cursor:cursor+len(target)]));cursor+=len(target)
        lexical=[words[i] for i in ids if key(words[i]['word'])]
        if any(not all(math.isfinite(float(w[k])) for k in ('start','end')) or
               not 0<=w['start']<w['end']<=duration+.025 for w in lexical):
            raise ValueError(f'Invalid/collapsed forced timestamps for dialogue {row["id"]+1}.')
        if any(b['start']<a['start'] or b['end']<a['end'] for a,b in zip(lexical,lexical[1:])):
            raise ValueError(f'Nonmonotonic forced timestamps for dialogue {row["id"]+1}.')
        a,b=lexical[0]['start']+offset,lexical[-1]['end']+offset
        if result and a<result[-1]['end']-.025:
            raise ValueError(f'Overlapping forced timestamps for dialogue {row["id"]+1}.')
        result.append({**row,'start':a,'end':b,'coarse_start':row.get('start'),'coarse_end':row.get('end'),
            'alignment_quality':row_quality(lexical),'aligned_words':[{**w,'start':w['start']+offset,'end':w['end']+offset} for w in lexical]})
    return result


@local_stage
def align_rows(audio,rows,language,folder,progress=lambda *a:None,original_audio=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    language=language.lower().split('-')[0]
    if not rows:return [],[]
    info=sf.info(audio)
    identity={'version':VERSION,'model':MODEL,'language':language,'rows':rows,
              'audio_sha256':audio_digest(audio),'original_audio_sha256':audio_digest(original_audio) if original_audio else None,'timing':'whole_original_clock_ctc_no_llm_windows','audio_transform':'native_pitch_and_time_no_pitch_correction','voice_release_policy':RELEASE_POLICY}
    cache=folder/'aligned.json'
    if cache.exists():
        saved=json.loads(cache.read_text('utf8'))
        if saved.get('identity')==identity:return saved['rows'],saved['audit']
    # Full-recording alignment is independent of every LLM timestamp.
    # Acoustic feature extraction is chunked inside the worker, not the script.
    identity['timing']='whole_original_clock_ctc_no_llm_windows'
    progress(0,'MMS-FA 전체 원문 정렬 준비')
    part=folder/'whole-recording';part.mkdir(exist_ok=True)
    job={'audio':str(Path(audio).resolve()),'texts':[spoken(r['text']) for r in rows],
         'cache':str((part/'mms-alignment.json').resolve()),
         'identity':{'version':LEXICAL_VERSION,'language':language,'texts':[spoken(r['text']) for r in rows],
                     'audio_sha256':identity['audio_sha256']}}
    if original_audio:
        if abs(sf.info(original_audio).duration-info.duration)>.025:raise ValueError('Original and separated audio clocks differ')
        job['original_audio']=str(Path(original_audio).resolve())
        job['identity']['original_audio_sha256']=identity['original_audio_sha256']
    batch=align_batch([job],language,folder,progress)
    chunk=batch['results'][0]
    try:
        result=measured_rows(rows,chunk['words'],info.duration)
        with sf.SoundFile(audio) as stream:
            for row in result:
                a=round(row['start']*info.samplerate);b=round(row['end']*info.samplerate)
                stream.seek(a);values=stream.read(b-a,dtype='float32',always_2d=True)
                if not len(values) or float(np.sqrt(np.mean(values**2)))<1e-6:
                    raise ValueError(f'No acoustic support for source dialogue {row["id"]+1}.')
    except ValueError as error:
        atomic_json(folder/'validation-error.json',{'error':str(error),'rows':rows,'alignment':chunk})
        raise RuntimeError('MMS-FA source timing alignment failed: '+str(error)+' Gemini analysis and alignment logs were preserved.') from error
    progress(99,'MMS-FA 원본 정렬 완료 · 발화 끝 경계 확인')
    result,tail_review=recover_ends(audio,result)
    atomic_json(folder/'voice-release-review.json',tail_review)
    audit=[{'id':r['id'],'before':[r['coarse_start'],r['coarse_end']],
            'after':[r['start'],r['end']],'method':MODEL,'llm_time_used':False,'quality':r['alignment_quality'],'voice_release':r.get('voice_release')} for r in result]
    atomic_json(cache,{'identity':identity,'rows':result,'audit':audit,'runtime':{k:v for k,v in batch.items() if k!='results'}})
    progress(100,f'MMS-FA 원본 정렬 완료 · {len(result)}개 대사')
    return result,audit
