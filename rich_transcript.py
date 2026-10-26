"""Carry ASR punctuation and vocal cues onto existing word times, never new times."""
import re,unicodedata
from difflib import SequenceMatcher
SPEAKER=re.compile(r'<\|speaker:\d+\|>')
ANNOTATION=re.compile(r'<\|speaker:\d+\|>|\[[^\]\n]*\]')
TTS_CUE_POLICY={'version':2,'reference_cues':'preserve','output_cues':'preserve_inline_annotations'}
POLICY={'version':2,'timing':'existing_asr_words_only','speaker_tags':'removed_from_tts',
        'emotion_tags':'preserve_and_translate_position_verbatim'}
def without_speakers(text):return SPEAKER.sub('',text)
def without_delivery_cues(text):return re.sub(r'\[[^\]\n]*\][ \t]*', '', text).strip()
def spoken(text):return ANNOTATION.sub('',text).strip()
def normalize(text):
    return ''.join(c.lower() for c in unicodedata.normalize('NFKC',spoken(text)) if unicodedata.category(c)[0] in 'LN')
def enrich(result):
    words=result.get('words',[]);raw=result.get('text','')
    if not words or not raw:return result
    chars=[];tags=[];speakers=[];active=[];speaker=None;last=0
    for match in ANNOTATION.finditer(raw):
        for c in raw[last:match.start()]:chars.append(c);tags.append(tuple(active));speakers.append(speaker)
        if match.group().startswith('<|speaker:'):active=[];speaker=match.group()
        else:active=[match.group()]
        last=match.end()
    for c in raw[last:]:chars.append(c);tags.append(tuple(active));speakers.append(speaker)
    raw_norm=[];positions=[]
    for i,c in enumerate(chars):
        for v in unicodedata.normalize('NFKC',c).lower():
            if unicodedata.category(v)[0] in 'LN':raw_norm.append(v);positions.append(i)
    word_norm='';owners=[]
    for i,w in enumerate(words):
        value=normalize(w.get('word',w.get('text','')));word_norm+=value;owners.extend([i]*len(value))
    mapped={}
    for block in SequenceMatcher(None,''.join(raw_norm),word_norm,autojunk=False).get_matching_blocks():
        for k in range(block.size):mapped.setdefault(owners[block.b+k],[]).append(positions[block.a+k])
    enriched=[];carried_punctuation='';numeric_spans=[(m.start(),m.end(),m.group()) for m in re.finditer(r'\d+', ''.join(chars))]
    for i,w in enumerate(words):
        if i not in mapped:
            value=w.get('word',w.get('text','')).strip()
            if value and all(unicodedata.category(c).startswith('P') for c in value) and value in carried_punctuation:
                # Already copied from this exact source span onto the preceding
                # lexical word. Preserve raw repetitions; do not emit them twice.
                at=carried_punctuation.index(value);carried_punctuation=carried_punctuation[at+len(value):]
                enriched.append({**w,'rich_surface':'','punctuation_carried_by_previous_word':True})
            else:enriched.append(dict(w));carried_punctuation=''
            continue
        indexes=mapped[i];lo,hi=min(indexes),max(indexes)+1
        # Carry immediately following punctuation, not another lexical word.
        while hi<len(chars) and not chars[hi].isalnum() and chars[hi] not in '\r\n':hi+=1
        carried_punctuation=''.join(chars[max(indexes)+1:hi]).strip()
        surface=''.join(chars[lo:hi]).strip()
        extra={}
        if normalize(w.get('word',w.get('text',''))).isdigit():
            span=next(((a,b,t) for a,b,t in numeric_spans if a<=min(indexes) and max(indexes)<b),None)
            if span:extra={'numeric_span':[span[0],span[1]],'numeric_literal':span[2]}
        enriched.append({**w,'rich_surface':surface,'source_char_start':lo,'source_char_end':hi,'space_after':bool(hi>lo and chars[hi-1].isspace()),'vocal_tags':list(tags[lo]),'fish_speaker':speakers[lo],**extra})
    return {**result,'words':enriched,'rich_transcript_policy':POLICY}
def join_words(words):
    output='';active=None;space_after=False
    for w in words:
        value=w.get('rich_surface',w.get('word',w.get('text',''))).strip()
        if not value:continue
        current=tuple(w.get('vocal_tags',[]))
        if space_after and output and not output[-1].isspace():output+=' '
        if current and current!=active:output+=(' ' if output else '')+''.join(current)
        if output and value and output[-1].isalnum() and value[0].isalnum():
            if not (is_cjk(output[-1]) and is_cjk(value[0])):output+=' '
        output+=value;active=current;space_after=w.get('space_after',False)
    return without_speakers(output.strip())
def is_cjk(c):return '\u3040'<=c<='\u30ff' or '\u3400'<=c<='\u9fff' or c=='々'
