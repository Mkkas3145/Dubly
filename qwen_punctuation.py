"""Audio-grounded punctuation repair; words, identities and clocks stay fixed."""
from pathlib import Path
import base64,hashlib,json,re,time,unicodedata
from core import QWEN,atomic_json
from qwen_delivery_cues import request
from qwen_response import normalize as normalize_response,parse as parse_response

MARKS='.,!?;:…‥。、！？；：'
POLICY={'version':5,'model':QWEN,'audio':'whole_separated_dialogue_original_clock',
    'base64_limit':8_000_000,'max_tokens':65536,'allowed_changes':'sentence_punctuation_and_audible_stutter_prefixes',
    'words_speakers_times':'immutable','delivery_cues':False}
PROMPT='''Listen to the audio and repair only the punctuation of each supplied transcript line to reflect the audible intonation. Distinguish statements, questions, exclamations, surprised questions (?!) and unfinished speech/pauses. Use question/exclamation marks where the actual delivery warrants them, including inside a line when phrasing changes. Loudness alone does not imply a question. Keep neutral statements neutral; do not exaggerate or mechanically turn every loud line into an exclamation.
You may edit expressive sentence punctuation broadly, not only question/exclamation marks: commas, periods, ellipses (.../…/……), repetitions (!!/???), combinations (?!/!?), tildes (~ / 〜 / ～) for an audibly drawn-out or lilting delivery, quotation marks, long dashes and other language-appropriate punctuation. Choose what the actual audio warrants; do not mechanically decorate every line. Preserve every original word, character, spelling, digit, repetition, hesitation and whitespace exactly. Do not change decimal/thousands separators, percent signs or other numeric notation, or insert punctuation inside words or numbers. Do not transcribe again, correct words, translate, or add any bracketed emotion, acting or sound-event cues. Preserve any existing bracketed text exactly. Preserve line IDs and order; never split or merge lines. Times locate the speech and cannot be modified. Return only {"segments":[{"id":0,"text":"unchanged words with repaired punctuation?!"}]} covering every supplied ID exactly once.'''
PROMPT+=''' One limited exception to character preservation: when clearly audible in the supplied audio, you MAY restore a stutter/repeated opening sound before the first word, separated by a comma or pause punctuation. For example, 知らない may become し、知らない, 俺が may become お、俺が, and 아무것도 may become 아, 아무것도 ONLY if that repeated sound is actually heard. Keep the entire supplied line after that opening intact except punctuation. Never invent a stutter for expressiveness, add new semantic words, replace a word, change a number, or alter speaker/timing. If unsure whether an extra opening sound is audible, keep the original wording.'''


def editable_punctuation(char):
    return char in '~〜～' or (unicodedata.category(char).startswith('P') and char not in '%‰‱#@&*\\/_')


def immutable_text(text):
    # Internal Latin-word/numeric punctuation is lexical (e.g. 1.5 or 1,000).
    protected=set();i=0
    while i<len(text):
        if not editable_punctuation(text[i]):i+=1;continue
        start=i
        while i<len(text) and editable_punctuation(text[i]):i+=1
        if start and i<len(text) and text[start-1].isascii() and text[start-1].isalnum() and text[i].isascii() and text[i].isalnum():
            protected.update(range(start,i))
    return ''.join(c for i,c in enumerate(text) if not editable_punctuation(c) or i in protected)


def validate(value,original):
    if not isinstance(value,str) or not value.strip():return 'invalid_text'
    if re.findall(r'\[[^\]]*\]',value)!=re.findall(r'\[[^\]]*\]',original):return 'bracketed_text_changed'
    if immutable_text(value)!=immutable_text(original) and not stutter_prefix(value,original):return 'non_punctuation_changed'
    return None


def phonetic_key(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    if re.search(r'[ぁ-んァ-ヶ一-龯]',text):
        from source_word_boundaries import japanese_tokenizer
        text=''.join(token.reading_form() for token in japanese_tokenizer().tokenize(text))
    return ''.join(chr(ord(c)-0x60) if 'ァ'<=c<='ヶ' else c for c in text if c.isalpha())


def stutter_prefix(value,original):
    before,after=immutable_text(original),immutable_text(value)
    if not before or not after.endswith(before):return False
    extra=after[:-len(before)].strip()
    if not extra or not all(c.isalpha() for c in extra):return False
    # Compare the added sound to the actual first lexical word's pronunciation,
    # rather than maintaining lists of allowed Japanese/Korean interjections.
    first=before.split()[0]
    if re.search(r'[ぁ-んァ-ヶ一-龯]',first):
        from source_word_boundaries import japanese_tokenizer
        tokens=list(japanese_tokenizer().tokenize(first))
        if not tokens:return False
        first=tokens[0].reading_form()
    sound,word=phonetic_key(extra),phonetic_key(first)
    return bool(sound and word and any(sound==word[:n]*(len(sound)//n)
        for n in range(1,min(len(sound),len(word))+1) if len(sound)%n==0))


def normalize_punctuation_response(value):
    # Native clocks and other echoed metadata are never returned to the caller.
    return normalize_response(value)


def retry_rejected(values,rows,audio,run,cfg,report,progress):
    """Re-ask only rejected lines; successful punctuation is never discarded."""
    for attempt in range(2):
        rejected=[r for r in rows if validate(values.get(str(r['id'])),r['text'])]
        if not rejected:break
        ids=[r['id'] for r in rejected]
        progress(90,f'문장부호 보정 검증 · {len(ids)}개 대사 다시 확인 중')
        item={'type':'object','properties':{'id':{'type':'integer'},'text':{'type':'string'}},'required':['id','text'],'additionalProperties':False}
        schema={'type':'object','properties':{'segments':{'type':'array','items':item}},'required':['segments'],'additionalProperties':False}
        payload={'model':QWEN,'temperature':0,'max_tokens':POLICY['max_tokens'],'reasoning':{'enabled':False},'stream':False,
            'response_format':{'type':'json_schema','json_schema':{'name':'punctuation_retry','strict':True,'schema':schema}},
            'messages':[{'role':'system','content':PROMPT+''' Your previous attempt changed characters outside the permitted punctuation/opening-stutter edits and was rejected. This is NOT re-transcription. Do not add unrelated words, rewrite or remove source characters. Only punctuation and a clearly audible opening-sound repetition are permitted. If uncertain, return the supplied line unchanged.'''},
                {'role':'user','content':[
                    {'type':'text','text':json.dumps([{k:r[k] for k in ('id','start','end','text')} for r in rejected],ensure_ascii=False)},
                    {'type':'input_audio','input_audio':{'format':'mp3','data':base64.b64encode(audio.read_bytes()).decode()}}]}]}
        retry_folder=run/f'validation-retry-{attempt+1}-{time.time_ns()}';retry_folder.mkdir()
        report['requests']+=1
        record={'attempt':attempt+1,'ids':ids,'accepted_ids':[]}
        try:
            response,receipt=request(payload,cfg['key'],retry_folder,list(map(str,ids)),lambda *a:None,
                operation='문장부호 재검증',normalizer=normalize_punctuation_response,
                content_parser=lambda text:parse_response(text,ids))
            entries=normalize_response(response,ids)['segments']
            for entry in entries:
                row=next(r for r in rejected if r['id']==entry['id'])
                if validate(entry['text'],row['text']) is None:
                    values[str(row['id'])]=entry['text'];record['accepted_ids'].append(row['id'])
            record['receipt']=receipt
        except (RuntimeError,ValueError,KeyError,TypeError) as error:record['error']=str(error)
        report.setdefault('validation_retries',[]).append(record)
    return values


def repair_source(clean,rows,work,cfg,progress=lambda *a:None):
    from qwen_audio_pipeline import encode_audio,digest
    report={'requests':0,'policy':POLICY,'reviews':[]}
    if not rows:return rows,{**report,'corrected_regions':0}
    folder=Path(work)/'qwen-punctuation';(folder/'audio').mkdir(parents=True,exist_ok=True)
    try:
        audio,manifest=encode_audio(Path(clean),folder/'audio',digest(clean),lambda *a:None,base64_limit=POLICY['base64_limit'])
        inputs=[{k:r[k] for k in ('id','start','end','text')} for r in rows]
        identity={'policy':POLICY,'prompt':PROMPT,'regions':inputs,'audio':manifest}
        run=folder/hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest();run.mkdir(exist_ok=True)
        atomic_json(run/'input.json',identity);cache=run/'result.json'
        saved=json.loads(cache.read_text('utf8')) if cache.exists() else {}
        if saved.get('signature')==identity:
            values=saved['values'];report.update(saved['receipt'],requests=0)
        else:
            item={'type':'object','properties':{'id':{'type':'integer'},'text':{'type':'string'}},'required':['id','text'],'additionalProperties':False}
            schema={'type':'object','properties':{'segments':{'type':'array','items':item}},'required':['segments'],'additionalProperties':False}
            payload={'model':QWEN,'max_tokens':POLICY['max_tokens'],'reasoning':{'enabled':False},'stream':False,
                'response_format':{'type':'json_schema','json_schema':{'name':'punctuation_only','strict':True,'schema':schema}},
                'messages':[{'role':'system','content':PROMPT},{'role':'user','content':[
                    {'type':'text','text':'Repair punctuation only in these immutable lines:\n'+json.dumps(inputs,ensure_ascii=False)},
                    {'type':'input_audio','input_audio':{'format':'mp3','data':base64.b64encode(audio.read_bytes()).decode()}}]}]}
            report['requests']=1
            values,receipt=request(payload,cfg['key'],run,[str(r['id']) for r in rows],
                lambda p,d:progress(p,'원본 억양에 맞춰 문장부호 보정 중'),operation='문장부호 보정',normalizer=normalize_punctuation_response,
                content_parser=lambda text:parse_response(text,[r['id'] for r in rows]))
            entries=normalize_punctuation_response(values)['segments']
            if {e['id'] for e in entries}!={r['id'] for r in rows}:raise ValueError('region_ids_changed')
            values={str(e['id']):e['text'] for e in entries}
            report.update(receipt);atomic_json(cache,{'signature':identity,'values':values,'receipt':receipt})
        values=retry_rejected(values,rows,audio,run,cfg,report,progress)
        if report.get('validation_retries'):
            cached=json.loads(cache.read_text('utf8'))
            atomic_json(cache,{**cached,'values':values,'validation_retries':report['validation_retries']})
        result=[]
        for row in rows:
            value=values.get(str(row['id']));error=validate(value,row['text'])
            if error:
                report['reviews'].append({'id':row['id'],'action':'keep_original_punctuation','reason':error});result.append(dict(row))
            else:result.append({**row,'text':value,'unpunctuated_text':row['text'],
                **({'audible_stutter_prefix_restored':True} if stutter_prefix(value,row['text']) else {})})
    except (RuntimeError,ValueError,KeyError,TypeError) as exc:
        report['error']=str(exc);result=rows
        report['reviews']=[{'id':r['id'],'action':'keep_original_punctuation','reason':'punctuation_stage_failed'} for r in rows]
    report['corrected_regions']=sum(r['text']!=old['text'] for r,old in zip(result,rows))
    atomic_json(folder/'audit.json',report)
    progress(100,f'문장부호 보정 완료 · {report["corrected_regions"]}개 반영')
    return result,report
