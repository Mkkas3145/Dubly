"""Audio-grounded emotion turns and open-set voice correction on fixed ASR words."""
from pathlib import Path
from copy import deepcopy
import base64,hashlib,json,re,time
from core import QWEN,LANGUAGES,atomic_json
from rich_transcript import join_words
from qwen_response import unique,literal_container
from qwen_nickname_language import instruction as nickname_language_instruction
from qwen_delivery_cues import request,CUE_ACOUSTIC_RULE

POLICY={'version':30,'max_attempts':3,'partial_resume':True,'model':QWEN,'reasoning':{'enabled':False},'max_tokens':65536,
    'base64_limit':8_000_000,'audio':'entire_separated_dialogue_original_clock',
    'stage':'after_corrected_gap_merge_before_translation','source_words':'lexical_characters_and_times_immutable_punctuation_editable',
    'timestamps':'existing_word_boundaries_only','speakers':'existing_or_new_nonnegative_ids_no_count_cap',
    'emotion':'cue_first_split_when_separate_reference_needed_for_faithful_acting','post_merge':False,
    'response_shape':'annotations_only_end_speaker_cues_punctuation_no_source_echo','nicknames':'output_language_names_prompt_only_after_voice_identity_v3','stream':True,'early_validation':True}
# Kept only for recovery of accepted annotations from older cached responses.
LEGACY_PROMPT='Listen to the entire attached dialogue recording. Every bracket cue MUST describe actual audible vocal execution, not just intent or emotion. Review the supplied source words for FOUR tasks: (1) prefer inline acting cues within one part; split the same speaker only when you judge cues alone are unlikely to reproduce the audible performance faithfully and separate local reference segments are needed; always separate actual speaker changes; (2) correct speaker identity from the actual voice, including speaker changes inside an existing line; (3) add, remove or correct sentence punctuation to match the actual audible intonation; (4) insert detailed English [acting/delivery cues] into each part text at sentence starts and meaningful audible changes in emotion, attitude, intensity, pace or rhetorical flow, including mid-sentence changes.\nASR speaker labels are extremely unreliable and may merge clearly different people or split one person across many IDs. Do NOT trust ASR speaker IDs, even as reference evidence for voice identity. Judge speaker identity independently from the actual audio throughout the recording; your audio-grounded judgment takes priority over every supplied ASR label. Do not assume that two regions with the same input ID are the same person, or that different input IDs mean different people. Existing speaker labels are provisional, not a fixed set. You MAY assign ANY new nonnegative integer speaker ID not present in the input when a distinct audible person was missed or merged. There is NO preset speaker count and NO 8/16-speaker cap. Reuse the same identity everywhere that voice returns throughout the recording. Assign consistent output IDs to the identities you independently hear. An existing number may be reused only after its identity is independently verified from audio; preserving ASR numbering is not a goal. Freely split a merged ASR identity into new IDs and unify different ASR IDs when the audio clearly belongs to one person. Use an unused ID for a newly distinguished voice. Never reuse an ID for two people. Group by the actual audible voice identity. Reuse one output speaker ID whenever the same voice returns, even across different scenes or character labels. Similarity alone is insufficient: do not merge distinct voices merely because they share age, gender, pitch range or speaking style. Distinguish persistent differences in timbre, resonance, articulation and vocal texture rather than temporary delivery changes. Character names, inferred roles and guessed nicknames must never override the audio-based identity decision. Do not force different speaker IDs solely because the dialogue suggests different story characters or roles. Never invent an explanation or a combined nickname to rationalize merging distinct audible voices. A change in emotion, pitch, loudness, whispering, shouting, crying, narration or internal monologue alone does NOT make a new person. Speaker identity evidence priority: first compare stable audible voice traits across clear examples (timbre, resonance, articulation, vocal texture and habitual rhythm). Use conversational context ONLY AS A LAST RESORT when acoustic comparison remains inconclusive, and only as a cautious tie-breaker among acoustically plausible identities. Context must NEVER override clear voice evidence, merge audibly distinct people or split one clearly matching voice. Do not force an assignment from story roles, character names, assumed turn-taking or external plot knowledge. If uncertainty remains, return speaker:null rather than guessing. Use null when voice identity is genuinely unclear or simultaneous voices cannot be assigned reliably.\nFor the SAME speaker, decide whether to split by the NEED FOR INDEPENDENT LOCAL REFERENCE AUDIO, not by whether the change is abrupt. First consider whether well-placed inline English acting cues can express the audible changes faithfully within ONE part. If so, keep the line together, even when the change sounds sudden or dramatic. Split only when, based on the actual audible performance, you judge that cues alone are unlikely to make TTS reproduce the distinct deliveries faithfully and each part needs its own focused reference. For example, contrasting vocal modes, incompatible breath/voice states, or delivery contours that would likely be averaged away in one generated utterance may justify separating coherent phrases. These are possible reasons, not automatic split rules: a whisper-to-shout change can remain together if inline cues are sufficient. A gradual change may justify a split if a separate reference is genuinely needed; abruptness alone never requires one. This is a generation-planning judgment, not a claim that you have tested TTS or know exactly how it will respond. When there is no clear need for independent references, prefer ONE part and use inline cues. Do not split merely because a phrase needs another cue, or merely to shorten a line or equalize lengths. Topic, scene or background-music changes alone never justify a split. Normal syllable stress, breaths and small variations alone are not split triggers. Actual speaker changes still MUST be separated, independently of this same-speaker criterion. Never split inside a lexical word, detach a hesitation from its word, or move punctuation into an independent part. Short meaningful replies remain valid.\nPrioritize complete sentences and coherent semantic units when choosing split boundaries. Prefer a sentence ending or a completed clause/phrase. A brief pause, breath, ordinary emphasis or subtle emotional variation alone is NOT a reason to split a connected expression. Never cut a word stem from its ending, a particle or auxiliary from the expression it completes, or a dependent construction from its continuation, even when the ASR exposes separate character/word IDs. These rules apply to every source language. A mid-sentence split is allowed only for an actual change of speaker or when the audible performance requires separate local references because inline acting cues alone are judged insufficient. For the same speaker, place such a split at the nearest meaningful completed clause or phrase that still matches the audible transition; if none exists, keep the semantic unit together. An actual change of speaker must remain separated: never move a boundary across another person\'s words merely to finish a sentence. Before returning, check both sides of each proposed split for incomplete meaning; avoid orphaned continuations. Preserve the exact source words and existing word times.\nSpoken word characters and word timestamps are immutable, but sentence punctuation is editable. DO NOT re-transcribe, rewrite words, translate, return seconds, or invent words. The word IDs locate the audio. Each output region must partition ONLY its own supplied word list into contiguous nonempty parts. Cover EVERY region and EVERY word exactly once in order, with no omissions, duplicates, gaps or cross-region moves. Each part uses w:[first_word_id,last_word_id] with inclusive source word IDs, s for speaker identity, and t for source text with cues. Return one part for an unchanged region. Express acting only through detailed English cues in text. Do not return a separate emotion label, explanation, reasoning or plot summary. Do not merge regions. The audio clock includes all original silence. For EACH part, return text containing exactly that part\'s source words, with punctuation corrected from the audio. You MAY add, delete or modify ?, !, ?!, !?, ~, 〜, ～, commas, periods, ellipses (... / …), repeated marks and other language-appropriate sentence punctuation, including inside the part. No fixed limit on repeated punctuation is required. Keep neutral speech neutral: do not decorate every line. Preserve every spoken character, spelling, number, hesitation, repetition and whitespace exactly. Preserve numeric notation such as 1.5, 1,000 and percentages, and punctuation inside lexical words. Insert only audio-grounded, detailed ENGLISH [acting/delivery cues] grounded in the actual audio. Put a cue at each sentence start and immediately before a meaningful internal delivery change, never inside a word or after the final spoken word. Capture communicative intent and subtle or mixed expressions, not generic emotion alone; do not invent acting or sounds. Preserve any existing bracketed text exactly. These cues are annotations, never spoken words. Punctuation edits must never move words or change timestamps.\nAFTER finalizing voice identities from audio, give EVERY identified speaker a short distinct nickname in NICKNAME_LANGUAGE (64 characters maximum) that helps a human identify the character while editing. For nicknames, you MAY reason from the dialogue content, names, forms of address, relationships, narrative situation and apparent role. Prefer a supported character name or a recognizable content-based role/relationship, such as guild leader, innkeeper, the person explaining the situation, or the protagonist\'s friend, over generic low/high/male/female voice descriptions. You MAY infer a likely character name from context; mark uncertain inferred names as tentative in NICKNAME_LANGUAGE, for example 이름(추정) in Korean. If a real name is not inferable, invent a useful distinctive role-based nickname from the content rather than inventing a supposedly factual personal name. Use an audible-voice description only when the dialogue provides no useful content clues. A nickname is a display label, NOT evidence for speaker grouping: never change independently established audible voice identities to fit a guessed name, relationship or plot expectation. Do not split the same audible voice solely to accommodate different inferred character names. Keep a supplied existing user nickname if one is provided.\nReturn JSON only: {"speakers":[{"id":0,"nickname":"short label"},{"id":3,"nickname":"another label"}],"segments":[{"id":0,"parts":[{"w":[0,4],"s":0,"t":"[soft low voice, steady measured pace, calm] source words with corrected punctuation."},{"w":[5,9],"s":3,"t":"[raised tense voice, clipped forceful delivery, angry] remaining source words!"}]}]}. IDs in this example are illustrative, not a required speaker count.\nMANDATORY CUE COVERAGE: EVERY output dialogue line and EVERY split part MUST begin with at least one sufficiently detailed English [acting/delivery cue]. No exceptions for short replies, interjections, neutral speech or unchanged parts. A cue on a preceding line or part does not count; each part must carry its own cue. If the emotion or intent is uncertain, describe only the audible manner of speaking rather than omitting the cue or inventing an emotion. For neutral or matter-of-fact speech, describe its actual audible delivery as well; a bare neutral label is insufficient. Check every part for its own leading English cue before returning. Describe BOTH communicative intent and audible delivery together in the same adequately detailed cue when both are identifiable. Keep the intent (such as seeking an answer, teasing, reassuring or warning), and independently describe HOW the voice actually sounds: relative loudness, register, softness or firmness, breathiness or strain, pace and pauses as needed. Describe register relative to this speaker, without changing voice identity. A question, accusation or demand can be quiet, gentle and low-voiced; do not turn it sharp, forceful or loud merely because of its meaning or punctuation. Conversely, do not erase its intent just because the voice is soft. Preserve mixed qualities when audible. Do not simplify, summarize or compress the performance into generic labels such as angry shout or harsh tone. Preserve every distinguishable audible feature that matters to reproducing this performance, including simultaneous or changing qualities. Do not omit trembling, tears, broken voice, breath control, strained onset, emphasized words, pacing or intensity changes when they are actually audible. Do not add features merely to fill a checklist, repeat synonyms, explain the plot or speculate. Audio-grounded delivery constrains the intent label; never invent either intent or vocal traits from the script alone. Do not stop after identifying two or three representative descriptors. There is no fixed descriptor count: preserve the distinct audible qualities needed to reproduce this particular delivery. Listen for qualities that coexist or seem contradictory; a firm request for confirmation can also sound hesitant, restrained or unsure. Firmness describes neither fluency nor certainty, and communicative intent does not replace how the words are delivered. If hesitation is audible, retain its actual form (such as a tentative onset, faltering rhythm, a restart or a meaningful pause) alongside firmness or pressure, rather than letting the dominant intent erase it. For example, [firm, pressing for confirmation] is incomplete IF the audio also contains hesitation; [firm but hesitant, pressing for confirmation, brief faltering pauses] is appropriate ONLY if those pauses are actually heard. The example is not a default cue and must not be copied onto unrelated lines. Before finalizing each cue, check whether the dominant labels have hidden another audible quality that changes the acting; include that quality without padding, synonym lists or invented traits. Use inline cues when the delivery changes within the line. REQUIRED CONTENT OF EVERY CUE: Describe the AUDIBLE VOCAL EXECUTION FIRST. Each English bracket cue MUST say how the voice actually sounds or how the words are physically delivered, not merely what the speaker wants or feels. An intention-only or emotion-only cue is INVALID, even if its interpretation is correct. [panicked demand for information], [firm, accusatory statement] and [pressing for confirmation] are INCOMPLETE because they omit concrete audible delivery. Keep the intent when identifiable, but include the actual speaking manner and every other distinguishable delivery feature that matters; do not stop after the first feature. A valid cue might describe a soft low voice and measured pacing alongside pressure, or a strained raised voice and rushed uneven rhythm alongside panic, ONLY when heard. Do not equate panic with shouting, accusation with harshness, or firmness with fluent confidence. If an intent is uncertain, omit that intent and describe the audible voice/delivery instead. Stable neutral speech still requires its actual delivery, not just [neutral] or [matter-of-fact]. Before returning each part, silently check whether a voice actor could tell HOW to say these words from its cue; if it only says WHY or with WHAT EMOTION, replace it with an audio-grounded delivery direction before emitting. Keep one combined bracket cue with optional later inline cues; do not output extra analysis fields or explanations. This is a semantic requirement, not a minimum word count or a vocabulary checklist. Be greedy and uncapped in capturing distinguishable audible acting and delivery details: do not impose an arbitrary descriptor-count or cue-length cap. Retain all relevant details actually supported by the audio, including subtle or coexisting qualities, without padding, repetition or invented traits.'
_voice_rule='ASR speaker labels are extremely unreliable and may merge clearly different people or split one person across many IDs. Do NOT trust ASR speaker IDs, even as reference evidence for voice identity. Judge speaker identity independently from the actual audio throughout the recording; your audio-grounded judgment takes priority over every supplied ASR label. Do not assume that two regions with the same input ID are the same person, or that different input IDs mean different people. Existing speaker labels are provisional, not a fixed set. You MAY assign ANY new nonnegative integer speaker ID not present in the input when a distinct audible person was missed or merged. There is NO preset speaker count and NO 8/16-speaker cap. Reuse the same identity everywhere that voice returns throughout the recording. Assign consistent output IDs to the identities you independently hear. An existing number may be reused only after its identity is independently verified from audio; preserving ASR numbering is not a goal. Freely split a merged ASR identity into new IDs and unify different ASR IDs when the audio clearly belongs to one person. Use an unused ID for a newly distinguished voice. Never reuse an ID for two people. Group by the actual audible voice identity. Reuse one output speaker ID whenever the same voice returns, even across different scenes or character labels. Similarity alone is insufficient: do not merge distinct voices merely because they share age, gender, pitch range or speaking style. Distinguish persistent differences in timbre, resonance, articulation and vocal texture rather than temporary delivery changes. Character names, inferred roles and guessed nicknames must never override the audio-based identity decision. Do not force different speaker IDs solely because the dialogue suggests different story characters or roles. Never invent an explanation or a combined nickname to rationalize merging distinct audible voices. A change in emotion, pitch, loudness, whispering, shouting, crying, narration or internal monologue alone does NOT make a new person. Speaker identity evidence priority: first compare stable audible voice traits across clear examples (timbre, resonance, articulation, vocal texture and habitual rhythm). Use conversational context ONLY AS A LAST RESORT when acoustic comparison remains inconclusive, and only as a cautious tie-breaker among acoustically plausible identities. Context must NEVER override clear voice evidence, merge audibly distinct people or split one clearly matching voice. Do not force an assignment from story roles, character names, assumed turn-taking or external plot knowledge. If uncertainty remains, return speaker:null rather than guessing. Use null when voice identity is genuinely unclear or simultaneous voices cannot be assigned reliably.\nRecognize returning voices by comparison with earlier speech: retain an internal auditory reference for each speaker from their clearest earlier utterances in THIS recording. When a voice sounds familiar, recall those earlier examples and actively compare stable timbre, resonance, articulation and habitual rhythm before assigning its ID. Prefer direct cross-segment acoustic comparison over a vague impression of familiarity. Reuse verified identities across intervening speakers, scenes and delivery changes; do not force an existing identity when the voice is distinct or evidence is insufficient. Correct earlier provisional assignments if clearer later examples contradict them. These references must come only from the supplied recording, not assumed character names, known actors or outside memories. Do not output extra reference summaries.\n'
_split_rule="For the SAME speaker, decide whether to split by the NEED FOR INDEPENDENT LOCAL REFERENCE AUDIO, not by whether the change is abrupt. First consider whether well-placed inline English acting cues can express the audible changes faithfully within ONE part. If so, keep the line together, even when the change sounds sudden or dramatic. Split only when, based on the actual audible performance, you judge that cues alone are unlikely to make TTS reproduce the distinct deliveries faithfully and each part needs its own focused reference. For example, contrasting vocal modes, incompatible breath/voice states, or delivery contours that would likely be averaged away in one generated utterance may justify separating coherent phrases. These are possible reasons, not automatic split rules: a whisper-to-shout change can remain together if inline cues are sufficient. A gradual change may justify a split if a separate reference is genuinely needed; abruptness alone never requires one. This is a generation-planning judgment, not a claim that you have tested TTS or know exactly how it will respond. When there is no clear need for independent references, prefer ONE part and use inline cues. Do not split merely because a phrase needs another cue, or merely to shorten a line or equalize lengths. Topic, scene or background-music changes alone never justify a split. Normal syllable stress, breaths and small variations alone are not split triggers. Actual speaker changes still MUST be separated, independently of this same-speaker criterion. Never split inside a lexical word, detach a hesitation from its word, or move punctuation into an independent part. Short meaningful replies remain valid.\nPrioritize complete sentences and coherent semantic units when choosing split boundaries. Prefer a sentence ending or a completed clause/phrase. A brief pause, breath, ordinary emphasis or subtle emotional variation alone is NOT a reason to split a connected expression. Never cut a word stem from its ending, a particle or auxiliary from the expression it completes, or a dependent construction from its continuation, even when the ASR exposes separate character/word IDs. These rules apply to every source language. A mid-sentence split is allowed only for an actual change of speaker or when the audible performance requires separate local references because inline acting cues alone are judged insufficient. For the same speaker, place such a split at the nearest meaningful completed clause or phrase that still matches the audible transition; if none exists, keep the semantic unit together. An actual change of speaker must remain separated: never move a boundary across another person's words merely to finish a sentence. Before returning, check both sides of each proposed split for incomplete meaning; avoid orphaned continuations. Preserve the exact source words and existing word times.\nKeep dependent short endings with their preceding phrase: do not isolate a same-speaker sentence ending, trailing particle, brief attached afterthought, filler or hesitation that completes or belongs to that phrase. A softer, lingering, thoughtful or otherwise changed delivery on such an ending should be expressed with an inline English cue within the SAME part, rather than a tiny separate part. Apply this by grammatical/semantic dependence and audible continuity, not by a fixed minimum duration or word count. This does not prohibit short independent replies, reactions or genuinely standalone utterances, and actual speaker changes MUST still be separated.\n"
_name_rule="AFTER finalizing voice identities from audio, give EVERY identified speaker a short distinct nickname in NICKNAME_LANGUAGE (64 characters maximum) that helps a human identify the character while editing. For nicknames, you MAY reason from the dialogue content, names, forms of address, relationships, narrative situation and apparent role. Prefer a supported character name or a recognizable content-based role/relationship, such as guild leader, innkeeper, the person explaining the situation, or the protagonist's friend, over generic low/high/male/female voice descriptions. You MAY infer a likely character name from context; mark uncertain inferred names as tentative in NICKNAME_LANGUAGE, for example 이름(추정) in Korean. If a real name is not inferable, invent a useful distinctive role-based nickname from the content rather than inventing a supposedly factual personal name. Use an audible-voice description only when the dialogue provides no useful content clues. A nickname is a display label, NOT evidence for speaker grouping: never change independently established audible voice identities to fit a guessed name, relationship or plot expectation. Do not split the same audible voice solely to accommodate different inferred character names. Keep a supplied existing user nickname if one is provided.\n"
PROMPT='This is annotation, NOT transcription. Listen to the entire attached recording and annotate the supplied fixed word IDs for FOUR tasks: distinguish actual speakers, choose only necessary performance split boundaries, describe actual audible delivery in English cues, and correct expressive sentence punctuation. The input transcript, words and timestamps are LOCKED. Do not re-transcribe, translate, echo, summarize or return any source dialogue. Return ONLY annotation decisions, not a copy of the transcript. Source words, spelling, repetitions and numeric notation remain unchanged in code.\n'+_voice_rule+_split_rule+CUE_ACOUSTIC_RULE+'\n'+_name_rule+'\nCOMPACT OUTPUT FORMAT:\n{"segments":[{"id":0,"parts":[{"e":null,"s":0,"c":"soft low voice, measured pacing"}]}],"speakers":[{"id":0,"nickname":"short editing label"}]}\nFor each region return its id and nonempty parts in spoken order. Every part has ONLY:\n- e: inclusive final source word ID for a nonfinal part. Use null for the FINAL part, meaning all remaining source words in this region. The first part automatically begins at the first supplied word, and each next part begins after the previous end. Never return start word IDs or source text. These rules guarantee every source word is retained without reprinting it.\n- s: the audio-grounded speaker ID, or null if genuinely uncertain.\n- c: the mandatory English audible-delivery cue for this part, WITHOUT brackets. Include actual vocal execution, not just intent or emotion. Code inserts this cue at the part start.\nOPTIONAL fields, omit when unused:\n- m: mid-part cues, [{"i":word_id,"c":"audible delivery change"}]. Code inserts each cue immediately before that existing word. Never place one inside a source word or outside the part.\n- p: sentence punctuation edits, [{"i":word_id,"b":"leading punctuation","a":"trailing punctuation"}]. b/a replace only the punctuation at the beginning/end of that fixed word. Omit b or a to preserve that side; an empty string deletes punctuation there. Supply punctuation characters ONLY, never spoken letters or digits. Examples of permitted punctuation are ?, !, ?!, ~, ..., …, commas and language-appropriate marks. Preserve lexical punctuation inside words/numbers. Do not output unchanged punctuation or entire words. New punctuation within a line uses the relevant existing word boundary.\nKeep all IDs in their own region. Every region must have exactly one FINAL e:null part; all earlier ends must leave words for it and use allowed split boundaries (an end is immediately before an allowed next start). Speaker changes still require separation; ordinary changes can use m instead. Return all requested region IDs exactly once, followed by speaker nicknames. Use compact JSON, no Markdown, no explanation, no source dialogue, no extra emotion-summary field.'


def decode(content):
    """Accept fenced JSON and literal data; never execute model output."""
    if not isinstance(content,str):return content
    candidates=re.findall(r'```(?:[\w+-]+)?\s*([\s\S]*?)```',content,flags=re.I) or [content.strip()]
    found=[]
    for text in candidates:
        try:found.append(json.loads(text,object_pairs_hook=unique));continue
        except json.JSONDecodeError:pass
        try:found.append(literal_container(text.strip()));continue
        except (ValueError,SyntaxError,TypeError):pass
        decoder=json.JSONDecoder(object_pairs_hook=unique);cursor=0
        while cursor<len(text):
            match=re.search(r'[\[{]',text[cursor:])
            if not match:break
            start=cursor+match.start()
            try:value,end=decoder.raw_decode(text,start)
            except json.JSONDecodeError:cursor=start+1;continue
            found.append(value);cursor=end
        if not found:
            for opening,closing in (('{','}'),('[',']')):
                start,end=text.find(opening),text.rfind(closing)
                if start<0 or end<start:continue
                try:found.append(literal_container(text[start:end+1]));break
                except (ValueError,SyntaxError,TypeError):pass
    if len(found)!=1:raise ValueError('ambiguous_or_missing_review_json')
    return found[0]


def number(value):
    if type(value) is int and value>=0:return value
    if type(value) is float and value>=0 and value.is_integer():return int(value)
    if isinstance(value,str):value=value.strip()
    if isinstance(value,str) and re.fullmatch(r'\d+',value):return int(value)
    raise ValueError('invalid_nonnegative_id')


def pick(value,names,default=None):
    found=[value[k] for k in names if k in value]
    if found and any(v!=found[0] for v in found[1:]):raise ValueError('conflicting_review_fields')
    return found[0] if found else default


def part_data(part):
    if isinstance(part,list) and len(part)==3 and isinstance(part[2],str) and not part[2].strip().isdigit():
        part={'e':part[0],'s':part[1],'c':part[2]}
    if isinstance(part,list) and len(part) in (3,4,5):
        part=dict(zip(('first_word_id','last_word_id','speaker','emotion','text'),part))
    if not isinstance(part,dict):raise ValueError('invalid_review_part')
    span=pick(part,('word_ids','word_range','range','w'))
    lo=pick(part,('first_word_id','start_word_id','first_word','start_id','word_start'),span[0] if isinstance(span,list) and len(span)==2 else None)
    hi=pick(part,('last_word_id','end_word_id','last_word','end_id','word_end'),span[1] if isinstance(span,list) and len(span)==2 else None)
    speaker_keys=('speaker','speaker_id','voice','voice_id','s')
    if not any(k in part for k in speaker_keys):raise ValueError('missing_speaker_identity')
    voice=pick(part,speaker_keys)
    if isinstance(voice,list) and len(voice)==1:voice=voice[0]
    if isinstance(voice,str):
        voice=voice.strip()
        match=re.fullmatch(r'(?:speaker|voice|화자)[_ :#-]*(\d+)',voice,re.I)
        if match:voice=match[1]
    voice=None if voice is None or voice in (-1,'-1','unknown','null','uncertain','Unknown') else number(voice)
    if 'e' in part or 'delivery_cue' in part:
        end=pick(part,('e','last_word_id'))
        if isinstance(end,list) and len(end)==1:end=end[0]
        cue=pick(part,('c','delivery_cue'))
        if isinstance(cue,list) and cue and all(isinstance(x,str) for x in cue):cue=''.join(cue)
        if not isinstance(cue,str) or not cue.strip():raise ValueError('missing_audible_delivery_cue')
        inline=pick(part,('m','inline_cues'),[]);marks=pick(part,('p','punctuation_marks'),[])
        if isinstance(inline,dict):inline=[inline]
        if isinstance(marks,dict):marks=[marks]
        if not isinstance(inline,list) or not isinstance(marks,list):raise ValueError('invalid_cue_or_punctuation_positions')
        return {'first_word_id':None,'last_word_id':None if end is None else number(end),'speaker':voice,
                'emotion':'','delivery_cue':cue,'inline_cues':inline,'punctuation_marks':marks}
    emotion=pick(part,('emotion','delivery','affect','e'),'')
    if isinstance(emotion,list) and all(isinstance(x,str) for x in emotion):emotion=', '.join(emotion)
    if emotion is None:emotion=''
    if not isinstance(emotion,str) or len(emotion)>500:raise ValueError('invalid_emotion_description')
    text=pick(part,('text','punctuated_text','original_text','transcript','t'))
    # Some providers ignore the string schema and wrap text in string fragments.
    # Preserve the fragments literally; lexical validation still rejects rewrites.
    if isinstance(text,list) and text and all(isinstance(fragment,str) for fragment in text):text=''.join(text)
    if text is not None and (not isinstance(text,str) or not text.strip() or len(text)>8000):raise ValueError('invalid_punctuated_text')
    return {'first_word_id':number(lo),'last_word_id':number(hi),'speaker':voice,'emotion':emotion,**({'text':text} if text is not None else {})}


def nicknames(value):
    names={}
    def add(identifier,name):
        if isinstance(identifier,str):
            match=re.fullmatch(r'(?:speaker|voice|화자)[_ :#-]*(\d+)',identifier.strip(),re.I)
            if match:identifier=match[1]
        identifier=str(number(identifier))
        if not isinstance(name,str) or not name.strip() or len(name.strip())>64:raise ValueError('invalid_speaker_nickname')
        name=name.strip()
        if identifier in names and names[identifier]!=name:raise ValueError('conflicting_speaker_nicknames')
        names[identifier]=name
    def scan(container):
        if isinstance(container,dict):
            for key in ('speaker_names','speaker_nicknames','nicknames','speakers'):
                data=container.get(key)
                if isinstance(data,dict):
                    for identifier,name in data.items():
                        if isinstance(name,dict):name=pick(name,('nickname','name','alias','label'))
                        add(identifier,name)
                elif isinstance(data,list):
                    for item in data:
                        if isinstance(item,dict):add(pick(item,('id','speaker_id','speaker')),pick(item,('nickname','name','alias','label')))
                        elif isinstance(item,list) and len(item)==2:add(*item)
            # Also accept aliases on individual turns, and wrapper objects.
            if any(k in container for k in ('nickname','alias')) and any(k in container for k in ('speaker','speaker_id')):
                voice=pick(container,('speaker','speaker_id'))
                if voice is not None:add(voice,pick(container,('nickname','alias')))
            for key,item in container.items():
                if key not in ('speaker_names','speaker_nicknames','nicknames','speakers') and isinstance(item,(dict,list)):scan(item)
        elif isinstance(container,list):
            for item in container:scan(item)
    scan(value)
    return names


def normalize(value,rows=None):
    value=decode(value)
    names=nicknames(value)
    for _ in range(4):
        if not isinstance(value,dict):break
        if value and all(re.fullmatch(r'\d+',str(k)) for k in value):
            entries=[]
            for k,v in value.items():
                if isinstance(v,dict):
                    existing=pick(v,('id','region_id','segment_id','index'))
                    if existing is not None and number(existing)!=number(k):raise ValueError('conflicting_region_id')
                    v={**v,'id':number(k)}
                else:v={'id':number(k),'parts':v}
                entries.append(v)
            value=entries;break
        if any(k in value for k in ('first_word_id','start_word_id','word_range','w')):
            value=[value];break
        if 'parts' in value and any(k in value for k in ('id','region_id','segment_id','index')):
            value=[value];break
        keys=[k for k in ('segments','regions','results','turns','data','result','output','speaker_segments') if isinstance(value.get(k),(list,dict))]
        if len(keys)!=1:raise ValueError('ambiguous_review_container')
        value=value[keys[0]]
    if not isinstance(value,list):raise ValueError('invalid_review_segments')
    entries=[];seen=set();flat=[]
    for row in value:
        if isinstance(row,list):
            if len(row)==2 and isinstance(row[1],list):row={'id':row[0],'parts':row[1]}
            else:flat.append(part_data(row));continue
        if not isinstance(row,dict):raise ValueError('invalid_review_region')
        identifier=pick(row,('id','region_id','segment_id','index'))
        raw_parts=pick(row,('parts','splits','turns','subsegments'))
        if raw_parts is None:
            part=part_data(row)
            if identifier is None:flat.append(part);continue
            raw_parts=[part]
        elif identifier is None:
            flat.extend(part_data(p) for p in raw_parts);continue
        identifier=number(identifier)
        seen.add(identifier)
        if isinstance(raw_parts,dict):raw_parts=[raw_parts]
        if not isinstance(raw_parts,list):raise ValueError('invalid_review_parts')
        parts=[part_data(p) for p in raw_parts]
        if not parts:raise ValueError('missing_review_parts')
        entries.append({'id':identifier,'parts':parts})
    if flat:
        if rows is None:raise ValueError('flat_turns_require_source_word_identity')
        owners={w['word_id']:r['id'] for r in rows for w in r['words']};grouped={}
        for part in flat:
            owner=owners.get(part['first_word_id'])
            if owner is None or owners.get(part['last_word_id'])!=owner:raise ValueError('flat_turn_crosses_source_regions')
            grouped.setdefault(owner,[]).append(part)
        entries.extend({'id':i,'parts':parts} for i,parts in grouped.items())
    # Explicit IDs determine order; formatting order is not semantic authority.
    grouped={}
    for entry in entries:grouped.setdefault(entry['id'],[]).extend(entry['parts'])
    entries=[{'id':identifier,'parts':sorted(parts,key=lambda p:p['first_word_id'] if p['first_word_id'] is not None else p['last_word_id'] if p['last_word_id'] is not None else float('inf'))} for identifier,parts in sorted(grouped.items())]
    return {'segments':entries,'speaker_names':names}


def fixed_rows(rows):
    result=deepcopy(rows);wid=0
    for row in result:
        if not row.get('words'):raise ValueError('review_requires_source_words')
        for word in row['words']:
            word.setdefault('asr_word_id',word.get('word_id'))
            word['word_id']=wid;wid+=1
    return result


def review_region_context(context,row,*,include_text=True):
    """Expose only word clocks consistent with the final corrected audio range."""
    a,b=row['start'],row['end'];words=row['words'];valid=[]
    for w in words:
        valid.append(a<=w['start']<w['end']<=b)
    allowed=[]
    for i,w in enumerate(words):
        if not i or not valid[i] or not any(c.isalnum() for c in w.get('word','')):continue
        preceding=[p for j,p in enumerate(words[:i]) if valid[j] and any(c.isalnum() for c in p.get('word',''))]
        if preceding and max(p['end'] for p in preceding)<=w['start']:allowed.append(w['word_id'])
    result={**context,'source_text':row['text'],
        'words':[[w[0],w[1] if valid[i] else None,w[2] if valid[i] else None,w[3]] for i,w in enumerate(context['words'])],
        'split_before_word_ids':allowed,
        'timing_note':'Only audio_range is the corrected authoritative region. null word times are untrusted; never use them to place a split.'}
    if not include_text:
        result.pop('source_text',None);result.pop('speaker',None)
        result['words']=[w[:3] for w in result['words']]
    return result


def include_unspoken_words(words,parts):
    """Restore omitted punctuation IDs, never infer missing lexical speech."""
    parts=deepcopy(parts);positions={w['word_id']:i for i,w in enumerate(words)}
    owners=[None]*len(words);restored=[]
    for slot,part in enumerate(parts):
        lo,hi=part['first_word_id'],part['last_word_id']
        if lo not in positions or hi not in positions or positions[hi]<positions[lo]:raise ValueError('review_invalid_source_word_range')
        for i in range(positions[lo],positions[hi]+1):
            if owners[i] is not None:raise ValueError('review_duplicate_source_words')
            owners[i]=slot
    for i,word in enumerate(words):
        if owners[i] is not None:continue
        if any(c.isalnum() for c in word.get('word','')):raise ValueError('review_unassigned_spoken_words')
        previous=next((owners[j] for j in range(i-1,-1,-1) if owners[j] is not None),None)
        following=next((owners[j] for j in range(i+1,len(words)) if owners[j] is not None),None)
        owner=previous if previous is not None else following
        if owner is None:raise ValueError('review_without_source_words')
        owners[i]=owner;restored.append(word['word_id'])
    for slot,part in enumerate(parts):
        indices=[i for i,owner in enumerate(owners) if owner==slot]
        if not indices or indices!=list(range(indices[0],indices[-1]+1)):raise ValueError('review_noncontiguous_source_words')
        part.update(first_word_id=words[indices[0]]['word_id'],last_word_id=words[indices[-1]]['word_id'])
    return parts,restored


def ranges_from_locked_text(words,parts):
    """Recover mistyped IDs only from an exact, complete lexical partition.

    No fuzzy matching, missing-word completion, speaker inference or new times.
    Every part must match consecutive existing whole words, in their original order.
    """
    from qwen_punctuation import immutable_text
    def key(text):
        return re.sub(r'\s+','',immutable_text(re.sub(r'\[[^\]\n]*\]','',text)))
    if not parts or any(not isinstance(p.get('text'),str) for p in parts):return None
    source=[key(w.get('rich_surface',w.get('word',''))) for w in words]
    targets=[key(p['text']) for p in parts]
    if any(not t for t in targets) or ''.join(source)!=''.join(targets):return None
    ends={};offset=0
    for i,text in enumerate(source):
        offset+=len(text);ends[offset]=i  # Trailing punctuation stays with its word.
    result=deepcopy(parts);start=0;offset=0
    for part,text in zip(result,targets):
        offset+=len(text);end=ends.get(offset)
        if end is None or end<start:return None  # Never cut through a source word.
        part.update(first_word_id=words[start]['word_id'],last_word_id=words[end]['word_id'])
        start=end+1
    return result if start==len(words) else None


def expand_annotations(row,parts):
    """Build speech from locked input; Qwen returns no original dialogue text."""
    from qwen_punctuation import editable_punctuation
    words=row['words'];positions={w['word_id']:i for i,w in enumerate(words)}
    result=[];start=0
    def cue(value):
        if not isinstance(value,str):raise ValueError('invalid_delivery_cue')
        value=value.strip()
        if value.startswith('[') and value.endswith(']'):value=value[1:-1].strip()
        if not value or not value.isascii() or not any(c.isalpha() for c in value) or any(c in '[]\r\n' for c in value):raise ValueError('invalid_english_cue')
        return '['+value+']'
    for slot,part in enumerate(parts):
        if 'delivery_cue' not in part:
            result.append(part);start=positions.get(part['last_word_id'],-1)+1;continue
        last=part['last_word_id']
        if last is None:
            if slot!=len(parts)-1:raise ValueError('final_part_marker_before_last_part')
            end=len(words)-1
        else:end=positions.get(last,-1)
        if not 0<=start<=end<len(words):raise ValueError('invalid_annotation_word_range')
        chunk=deepcopy(words[start:end+1]);local={w['word_id']:w for w in chunk};rejected=[]
        for edit in part['punctuation_marks']:
            try:
                if not isinstance(edit,dict):raise ValueError('invalid_marks')
                identifier=number(edit.get('i'));word=local[identifier]
                before,after=edit.get('b'),edit.get('a')
                if not any(k in edit for k in ('b','a')):raise ValueError('empty_marks')
                for marks in (before,after):
                    if marks is not None and (not isinstance(marks,str) or any(not editable_punctuation(c) and not c.isspace() for c in marks)):raise ValueError('non_punctuation_edit')
                text=word.get('rich_surface',word.get('word',''));a=0;b=len(text)
                while a<b and (editable_punctuation(text[a]) or text[a].isspace()):a+=1
                while b>a and (editable_punctuation(text[b-1]) or text[b-1].isspace()):b-=1
                # Never replace the lexical middle, including decimal separators.
                word['rich_surface']=(text[:a] if before is None else before)+text[a:b]+(text[b:] if after is None else after)
            except (ValueError,KeyError,TypeError):rejected.append(deepcopy(edit))
        inline={}
        for item in part['inline_cues']:
            if not isinstance(item,dict):raise ValueError('invalid_inline_cue')
            identifier=number(item.get('i'))
            if identifier not in local or identifier in inline:raise ValueError('invalid_inline_cue_word')
            inline[identifier]=cue(item.get('c'))
        for word in chunk:
            if word['word_id'] in inline:word['rich_surface']=inline[word['word_id']]+' '+word.get('rich_surface',word.get('word',''))
        text=cue(part['delivery_cue'])+' '+join_words(chunk)
        result.append({**part,'first_word_id':words[start]['word_id'],'last_word_id':words[end]['word_id'],
                       'text':text,'rejected_annotation_marks':rejected})
        start=end+1
    return result


def annotated_text_error(text,original):
    from qwen_punctuation import immutable_text
    pattern=r'\[[^\]\n]*\]'
    existing=re.findall(pattern,original);remaining=list(existing)
    found=re.findall(pattern,text)
    if text.count('[')!=len(found) or text.count(']')!=len(found):return 'malformed_cue'
    for tag in found:
        if tag in remaining:remaining.remove(tag);continue
        body=tag[1:-1]
        if not body.strip() or not body.isascii() or not any(c.isalpha() for c in body):return 'invalid_english_cue'
    if remaining:return 'existing_bracketed_text_changed'
    if re.search(pattern+r'\s*$',text):return 'cue_without_following_speech'
    # Cue insertion may add adjacent spaces; never allow changed spoken characters.
    locked=lambda value:re.sub(r'\s+','',immutable_text(re.sub(pattern,'',value)))
    if locked(text)!=locked(original):return 'changed_spoken_text'
    return None


def apply(rows,value):
    entries=normalize(value,rows)['segments'];by_id={row['id']:row['parts'] for row in entries}
    if set(by_id)!={r['id'] for r in rows}:raise ValueError('review_region_coverage_changed')
    result=[];audit=[]
    for row in rows:
        words=row['words'];positions={w['word_id']:i for i,w in enumerate(words)};offset=0;punctuation_edits=[];rejected_punctuation=[]
        proposed=expand_annotations(row,by_id[row['id']]);recovered=ranges_from_locked_text(words,proposed)
        ranges_repaired=bool(recovered and recovered!=proposed)
        parts,restored=include_unspoken_words(words,recovered if recovered is not None else proposed)
        # Coverage remains strict; an unusable timing annotation is a separate
        # issue and must not abort otherwise valid voice review for the video.
        covered=[w['word_id'] for p in parts for w in words[positions[p['first_word_id']]:positions[p['last_word_id']]+1]]
        if covered!=[w['word_id'] for w in words]:raise ValueError('review_duplicates_omits_or_moves_source_words')
        unapplied_splits=[]
        if len(parts)>1:
            previous_end=None;problem=None
            for slot,p in enumerate(parts):
                chunk=words[positions[p['first_word_id']]:positions[p['last_word_id']]+1]
                timed=[w for w in chunk if w['end']>w['start'] and any(c.isalnum() for c in w.get('word',''))]
                if not timed:problem='no_reliable_word_duration';break
                a=row['start'] if slot==0 else timed[0]['start']
                b=row['end'] if slot==len(parts)-1 else max(w['end'] for w in timed)
                if not row['start']<=a<b<=row['end']+1e-7:problem='word_times_outside_corrected_region';break
                if previous_end is not None and a<previous_end-1e-7:problem='overlapping_word_times';break
                previous_end=b
            if problem:
                unapplied_splits=[{'reason':problem,'parts':deepcopy(parts)}]
                voices={p['speaker'] for p in parts}
                # Never merge two different people's identities. Without a
                # trustworthy internal clock, retain the region as unresolved.
                parts=[{'first_word_id':words[0]['word_id'],'last_word_id':words[-1]['word_id'],
                    'speaker':next(iter(voices)) if len(voices)==1 else None,
                    'emotion':', '.join(dict.fromkeys(p['emotion'] for p in parts if p['emotion']))[:500],
                    'text':row['text']}]
        for slot,part in enumerate(parts):
            lo,hi=part['first_word_id'],part['last_word_id']
            if lo not in positions or hi not in positions or positions[lo]!=offset or positions[hi]<offset:
                raise ValueError('review_duplicates_omits_or_moves_source_words')
            offset=positions[hi]+1;chunk=deepcopy(words[positions[lo]:offset])
            positive=[w for w in chunk if w['end']>w['start'] and any(c.isalnum() for c in w.get('word',''))]
            if not positive and len(parts)!=1:raise ValueError('review_part_without_timed_speech')
            if slot and positive[0]['start']<result[-1]['end']-1e-7:raise ValueError('review_boundary_cuts_overlapping_speech')
            voice='unknown' if part['speaker'] is None else str(part['speaker'])
            # Retain acoustic outer edges. Internal cuts use fixed ASR word
            # boundaries, so no LLM timestamp can shift a line on the timeline.
            start=row['start'] if slot==0 else positive[0]['start']
            end=row['end'] if slot==len(parts)-1 else max(w['end'] for w in positive)
            if not row['start']<=start<end<=row['end']+1e-7:raise ValueError('review_part_outside_source_region')
            for word in chunk:word['speaker']=voice
            original=row['text'] if len(parts)==1 else join_words(chunk)
            text=part.get('text',original)
            from qwen_punctuation import immutable_text
            if annotated_text_error(text,original):
                # Optional punctuation must not discard valid word ranges or
                # voice assignments. Reject the rewrite, not the entire review.
                rejected_punctuation.append({'part_index':slot,'reason':'changed_spoken_text','original':original,'rejected':text})
                text=original
            if text!=original:punctuation_edits.append({'part_index':slot,'before':original,'after':text})
            result.append({**row,'id':len(result),'parent_source_region_id':row['id'],'part_index':slot,
                'start':start,'end':end,'speaker':voice,'words':chunk,'text':text,'audible_emotion':part['emotion'],
                'qwen_emotion_boundary_before':slot>0,'qwen_speaker_reviewed':True})
        if offset!=len(words):raise ValueError('review_unassigned_source_words')
        if ranges_repaired or unapplied_splits or rejected_punctuation or punctuation_edits or len(parts)>1 or any(('unknown' if p['speaker'] is None else str(p['speaker']))!=str(row['speaker']) for p in parts):
            audit.append({'source_region_id':row['id'],'previous_speaker':row['speaker'],'parts':parts,'word_ranges_recovered_from_exact_text':ranges_repaired,'punctuation_word_ids_restored':restored,'punctuation_edits':punctuation_edits,'rejected_punctuation':rejected_punctuation,'unapplied_splits':unapplied_splits})
    return result,audit


def complete_nicknames(values,rows,cfg,folder,progress):
    """Missing display labels never discard validated audio/word assignments."""
    values=normalize(values,rows);names=values['speaker_names']
    used={r['speaker'] for r in rows if r['speaker']!='unknown'}
    missing=sorted(identifier for identifier in used if identifier not in names)
    if not missing:return values,{'requests':0}
    schema={'type':'object','properties':{'speakers':{'type':'array','items':{'type':'object','properties':{'id':{'type':'integer'},'nickname':{'type':'string'}},'required':['id','nickname'],'additionalProperties':False}}},'required':['speakers'],'additionalProperties':False}
    context=[{'speaker':r['speaker'],'text':r['text']} for r in rows]
    payload={'model':QWEN,'max_tokens':4096,'reasoning':{'enabled':False},'stream':False,'provider':{'require_parameters':True},
        'response_format':{'type':'json_schema','json_schema':{'name':'speaker_nicknames','strict':True,'schema':schema}},
        'messages':[{'role':'system','content':nickname_language_instruction(cfg['language'])+' Speaker identities are already verified. Do not change them or split lines. Give each requested speaker a short distinct editing nickname in '+LANGUAGES[cfg['language']]+'. Use dialogue content, names and relationships; mark inferred names as tentative. Return only {"speakers":[{"id":0,"nickname":"label"}]}. The IDs are the requested existing identities.'},
            {'role':'user','content':json.dumps({'requested_speaker_ids':missing,'existing_names':names,'dialogue':context},ensure_ascii=False)}]}
    report={'requests':0};errors=[]
    for attempt in range(2):
        target=folder/f'nicknames-{attempt+1}';target.mkdir(exist_ok=True)
        try:
            extra,_=request(payload,cfg['key'],target,missing,lambda p,d:progress(p,'화자 별명 정리 중'),
                operation='화자 별명',normalizer=nicknames,content_parser=decode)
            # Only the requested missing labels may be added; no identity changes.
            for identifier in missing:
                if identifier in extra:names[identifier]=extra[identifier]
            if any(identifier not in names for identifier in missing):raise ValueError('missing_speaker_nicknames')
            report['requests']=attempt+1;break
        except (ValueError,TypeError,KeyError,RuntimeError) as error:
            errors.append(str(error));report['requests']=attempt+1
    if errors:report['errors']=errors
    report['missing_speaker_ids']=[identifier for identifier in missing if identifier not in names]
    atomic_json(folder/'nickname-review.json',report)
    return values,report



AUDIO_ONLY_PROMPT = """Listen to the complete original-clock dialogue recording. No transcript or ASR speaker labels are provided. Inputs contain region IDs, corrected audio_range, and word anchors [ID,start,end] ONLY. Anchor IDs locate positions in the audio; never transcribe, translate, quote speech, output dialogue, or invent timestamps.
Tasks: identify consistent audible speakers; supply detailed ENGLISH delivery cues for every part; use inline cues at meaningful delivery changes; split only for an actual speaker change or when cues alone cannot represent distinct deliveries faithfully and separate reference audio is needed. Keep connected semantic units together; do not detach short continuations or split words. Use only permitted split_before_word_ids for additional parts. A region with no permitted split has ONE part; use speaker null if multiple voices cannot be separated safely.
Compare familiar voices with earlier clear examples using timbre, resonance, articulation and habitual vocal texture. Emotion, shouting, whispering or crying alone does not make a different person. Similar age, pitch or gender does not make two distinct voices the same person. Assign any consistent nonnegative speaker ID, with no count cap. Use conversational context only as a last resort for acoustically plausible candidates; never let imagined roles override audible identity. Use null if genuinely uncertain.
Describe what is HEARD, not emotions inferred from dialogue meaning. Describe actual loudness, softness, register, breath, tension, tremble, tears, breaks, pacing, hesitations, stress, intensity and changes when audible. Intent may be included only when supported by audible delivery and must not replace HOW the voice sounds. Do not reduce mixed delivery to two generic labels, invent checklist traits, or exaggerate neutral speech. Audio length never excuses skipping details or later regions. Every part requires its own English cue, including neutral speech.
Return compact JSON only: {"segments":[{"id":0,"parts":[{"e":null,"s":0,"c":"quiet, breathy, hesitant","m":[{"i":3,"c":"voice tightens, trembling"}]}]}],"speakers":[{"id":0,"nickname":"name"}]}. e is the inclusive LAST word-anchor ID of a part; use null ONLY for the final part of EACH region. s is speaker integer or null. c is the initial English cue without brackets; optional m gives internal cues before anchor ID i. Cover each supplied region and anchor exactly once in order, without moving anchors across regions. Output only annotations, never source speech or punctuation edits. Give useful short speaker nicknames in NICKNAME_LANGUAGE based on the heard recording; mark inferred names uncertain. Nicknames must never determine speaker grouping. Return segments first, then speakers. No Markdown, reasoning or explanations."""


def review(clean,rows,work,cfg,progress=lambda *a:None,*,send_transcript=True):
    from qwen_audio_pipeline import encode_audio,digest
    if not rows:return rows,{'policy':POLICY,'requests':0,'changes':[]}
    inputs=fixed_rows(rows);folder=Path(work)/'qwen-performance-turns';(folder/'audio').mkdir(parents=True,exist_ok=True)
    audio,manifest=encode_audio(Path(clean),folder/'audio',digest(clean),lambda *a:None,base64_limit=POLICY['base64_limit'])
    context=[{'id':r['id'],'speaker':r['speaker'],'audio_range':[r['start'],r['end']],
        'words':[[w['word_id'],w['start'],w['end'],w.get('rich_surface',w['word'])] for w in r['words']]} for r in inputs]
    prompt=nickname_language_instruction(cfg['language'])+'\n'+PROMPT.replace('NICKNAME_LANGUAGE',LANGUAGES[cfg['language']])+' Emit compact JSON without indentation or extra whitespace, using e, s and c for each part, with optional m and p; never echo source dialogue. Emit the segments array first, then speakers. Start directly with the JSON object, without commentary or Markdown. Never use an empty parts array for an unchanged region.'
    if not send_transcript:prompt=AUDIO_ONLY_PROMPT.replace('NICKNAME_LANGUAGE',LANGUAGES[cfg['language']])
    signature={'policy':POLICY,'prompt':prompt,'regions':context,'audio':manifest}
    signature['input_mode']='transcript_and_audio' if send_transcript else 'audio_and_numeric_anchors_only'
    run=folder/hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest();run.mkdir(exist_ok=True)
    atomic_json(run/'input.json',signature);cache=run/'result.json'
    saved=json.loads(cache.read_text('utf8')) if cache.exists() else {}
    if saved.get('signature')==signature:
        values=saved['values'];receipt={**saved['receipt'],'requests':0};revised,audit=apply(inputs,values)
    else:
        part={'type':'object','properties':{
            'e':{'anyOf':[{'type':'integer'},{'type':'null'}]},'s':{'anyOf':[{'type':'integer'},{'type':'null'}]},'c':{'type':'string'},
            'm':{'type':'array','items':{'type':'object','properties':{'i':{'type':'integer'},'c':{'type':'string'}},'required':['i','c'],'additionalProperties':False}},
            'p':{'type':'array','items':{'type':'object','properties':{'i':{'type':'integer'},'b':{'type':'string'},'a':{'type':'string'}},'required':['i'],'additionalProperties':False}}},
            'required':['e','s','c'],'additionalProperties':False}
        if not send_transcript:part['properties'].pop('p')
        item={'type':'object','properties':{'id':{'type':'integer'},'parts':{'type':'array','items':part,'minItems':1}},'required':['id','parts'],'additionalProperties':False}
        speaker={'type':'object','properties':{'id':{'type':'integer'},'nickname':{'type':'string'}},'required':['id','nickname'],'additionalProperties':False}
        schema={'type':'object','properties':{'segments':{'type':'array','items':item},'speakers':{'type':'array','items':speaker}},'required':['segments','speakers'],'additionalProperties':False}
        payload={'model':QWEN,'max_tokens':POLICY['max_tokens'],'reasoning':{'enabled':False},'stream':True,'provider':{'require_parameters':True},
            'response_format':{'type':'json_schema','json_schema':{'name':'performance_turns','strict':True,'schema':schema}},
            'messages':[{'role':'system','content':prompt},{'role':'user','content':[
                {'type':'text','text':json.dumps([review_region_context(r,inputs[i],include_text=send_transcript) for i,r in enumerate(context)],ensure_ascii=False,separators=(',',':'))},
                {'type':'input_audio','input_audio':{'format':'mp3','data':base64.b64encode(audio.read_bytes()).decode()}}]}]}
        partial=run/'partial.json';accepted={};collected_names={};failures=[];receipts=[];requests_made=0
        by_id={row['id']:row for row in inputs}
        legacy_signature={**signature,'prompt':nickname_language_instruction(cfg['language'])+'\n'+LEGACY_PROMPT.replace('NICKNAME_LANGUAGE',LANGUAGES[cfg['language']])+' Emit compact JSON without indentation or extra whitespace, using e, s and c for each part, with optional m and p; never echo source dialogue. Emit the segments array first, then speakers. Start directly with the JSON object, without commentary or Markdown. Never use an empty parts array for an unchanged region.'}
        legacy_runs=[]
        for old_input in folder.glob('*/input.json'):
            if old_input.parent==run:continue
            if json.loads(old_input.read_text('utf8'))==legacy_signature:legacy_runs.append(old_input.parent)
        for saved_partial in [*(p/'partial.json' for p in legacy_runs),partial]:
            if not saved_partial.exists():continue
            previous=json.loads(saved_partial.read_text('utf8'))
            if previous.get('signature') in (signature,legacy_signature):
                for entry in previous.get('segments',[]):
                    identifier=entry['id']
                    if identifier not in by_id:continue
                    apply([by_id[identifier]],{'segments':[entry]})
                    accepted[identifier]=entry
        def accept_region(value):
            entry=normalize({'segments':[value]})['segments'][0];identifier=entry['id']
            apply([by_id[identifier]],{'segments':[entry]})
            accepted[identifier]=entry
            atomic_json(partial,{'signature':signature,'segments':[accepted[i] for i in sorted(accepted)]})
        # Reuse completed entries from streams previously stopped by the old
        # punctuation gate. Input signature and strict word coverage still match.
        from qwen_review_stream import ReviewProgress
        for previous_attempt in sorted(p for source in [*legacy_runs,run] for p in source.glob('attempt-*')):
            raw_path=previous_attempt/'raw.json';stream_path=previous_attempt/'stream-content.txt'
            if not raw_path.exists() or not stream_path.exists():continue
            raw=json.loads(raw_path.read_text('utf8'))
            if raw.get('stream_error') not in {'review_punctuation_changed_spoken_text','review_part_outside_source_region',
                'review_part_without_timed_speech','review_boundary_cuts_overlapping_speech'}:continue
            replay=ReviewProgress(inputs,lambda row,value:apply([row],{'segments':[value]}),lambda *a:None,
                on_region=lambda value:accept_region(value) if number(value.get('id',value.get('region_id',-1))) not in accepted else None)
            try:replay.feed(stream_path.read_text('utf8'))
            except (ValueError,TypeError,KeyError):pass  # Keep validated prefix of a truncated stream.
        initial_count=len(accepted)
        for attempt in range(POLICY['max_attempts']):
            pending=[row for row in inputs if row['id'] not in accepted]
            if not pending:break
            pending_ids={row['id'] for row in pending}
            item['properties']['id']['enum']=sorted(pending_ids)
            schema['properties']['segments'].update(minItems=len(pending),maxItems=len(pending))
            attempt_dir=run/f'attempt-{time.time_ns()}-{attempt+1}';attempt_dir.mkdir(exist_ok=True)
            from qwen_review_stream import ReviewProgress
            monitor=ReviewProgress(pending,lambda row,value:apply([row],{'segments':[value]}),progress,
                on_region=accept_region,completed_count=len(accepted),total=len(inputs))
            # A fresh prompt never copies malformed or runaway output into context.
            payload['max_tokens']=min(POLICY['max_tokens'],max(2048,len(pending)*256+2048))
            locked=[{'id':row['id'],'audio_range':[row['start'],row['end']],
                     'voices':[{'w':[p['first_word_id'],p['last_word_id']],'s':p['speaker']} for p in expand_annotations(row,accepted[row['id']]['parts'])]}
                for row in inputs if row['id'] in accepted]
            current=[review_region_context(row,by_id[row['id']],include_text=send_transcript) for row in context if row['id'] in pending_ids]
            payload['messages'][1]['content'][0]['text']=json.dumps({'pending_regions':current,'accepted_regions':locked},ensure_ascii=False,separators=(',',':'))
            extra=' Review ONLY pending_regions and return each of them with nonempty parts. accepted_regions are already validated: do not return or change them. Keep the same speaker ID when the same audible voice returns; unused IDs remain allowed. All word IDs and the full attached audio clock remain unchanged.'
            extra+=' Split timing constraint: audio_range is the corrected region. Some original ASR word times are unusable and are supplied as null. Start each additional part ONLY at a word ID in split_before_word_ids. An empty list means return exactly ONE part covering all words; still review its speaker and punctuation. If multiple people speak in such an unsplittable region, use speaker:null instead of merging their identities or inventing split times. Never split a zero-duration or unanchored hesitation into a separate part.'
            if not send_transcript:extra=extra.replace('speaker and punctuation','speaker and delivery')
            if failures and send_transcript:extra+=' The last response error was '+failures[-1][:160]+'. The last response was invalid. Preserve the audible details in inline English cues; return JSON with no commentary, empty parts or repeated regions.'
            payload['messages'][0]['content']=prompt+extra
            def waiting(p,d):
                progress(min(99,100*len(accepted)/len(inputs)),f'감정 전환·화자 보정 응답 대기 · 완료 {len(accepted)}/{len(inputs)}개 · 시도 {attempt+1}/{POLICY["max_attempts"]}') if p==0 else progress(p,d)
            try:
                requests_made+=1
                values,receipt=request(payload,cfg['key'],attempt_dir,[str(r['id']) for r in pending],waiting,
                    operation='감정 전환·화자 보정',normalizer=lambda v:normalize(v,pending),content_parser=decode,stream_validator=monitor)
                receipts.append(receipt)
                values=normalize(values,pending)
                for entry in values['segments']:
                    if entry['id'] not in pending_ids:
                        if accepted.get(entry['id'])==entry:continue
                        raise ValueError('review_conflicting_or_unrequested_region')
                    accept_region(entry)
                collected_names.update(values['speaker_names'])
                if len(accepted)!=len(inputs):raise ValueError('review_missing_regions:'+','.join(str(r['id']) for r in inputs if r['id'] not in accepted))
                break
            except (ValueError,TypeError,KeyError,RuntimeError,OSError) as error:
                failures.append(str(error));atomic_json(run/'validation-errors.json',failures)
                if len(accepted)==len(inputs):break
                progress(min(99,100*len(accepted)/len(inputs)),f'감정 전환·화자 보정 응답 오류 · 완료 {len(accepted)}/{len(inputs)}개 보관 · 다시 시도 중')
                if attempt<POLICY['max_attempts']-1 and re.search(r'HTTP (?:429|5\d\d)',str(error)):time.sleep(2**(attempt+1))
        if len(accepted)!=len(inputs):
            raise RuntimeError(f'Emotion and speaker correction response validation failed. Completed dialogue {len(accepted)}/{len(inputs)} was preserved; retry resumes the remainder.')
        values={'segments':[accepted[i] for i in sorted(accepted)],'speaker_names':collected_names}
        revised,audit=apply(inputs,values)
        receipt={'requests':requests_made,'attempt_receipts':receipts,'resumed_regions':initial_count}
        if send_transcript:
            values,nickname_report=complete_nicknames(values,revised,cfg,run,progress)
        else:
            # Never send source dialogue in a separate nickname fallback request.
            names=values.setdefault('speaker_names',{})
            for row in revised:
                if row['speaker']!='unknown':names.setdefault(str(row['speaker']),('화자 ' if cfg['language']=='ko' else 'Speaker ')+str(row['speaker']))
            nickname_report={'requests':0,'fallback':'local_labels_only_no_transcript_request'}
        receipt['nickname_review']=nickname_report
        atomic_json(cache,{'signature':signature,'values':values,'receipt':receipt})
    names=normalize(values,inputs)['speaker_names']
    report={'policy':POLICY,'transcript_sent':send_transcript,**receipt,'changes':audit,'speaker_names':names,'input_regions':len(rows),'output_regions':len(revised),'audio':manifest}
    atomic_json(folder/'audit.json',report)
    progress(100,f'감정 전환·화자 구분 확인 완료 · {len(revised)}개 대사')
    return revised,report
