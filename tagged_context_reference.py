"""Current reference first, all donors under speaker 1, with language wrappers."""
from pathlib import Path
import hashlib,json
import numpy as np
import soundfile as sf
from core import SR,atomic_json
from rich_transcript import without_speakers

POLICY={'version':35,'max_parts':None,'minimum_seconds':30.,'target_seconds':30.,
    'selection':'current_then_nearest_same_speaker_donors_in_source_order_no_repetition',
    'reference_tags':'current_untagged_donors_single_speaker_1','output_tags':'none','language_tags':'per_reference_part_and_output',
    'reference_gap_seconds':.5,'outer_silence_seconds':.5,
    'output_duration_hint':None,
    'generation':'individual','instructions':'native_speaker_pronunciation_intonation_stress_rhythm_reference_acting_v7'}


def prepare(segments,clean,folder,only_indices=None):
    from render import reference_audio
    bank=Path(folder)/'tagged-context-references';bank.mkdir(parents=True,exist_ok=True)
    selected=range(len(segments)) if only_indices is None else only_indices
    selected=[i for i in selected if not segments[i].get('deleted') and not segments[i].get('keep_original')]
    recordings={}
    def read(i):
        if i not in recordings:
            path=bank/f'{i:04d}-source.wav';reference_audio(clean,segments[i],path)
            values,rate=sf.read(path,dtype='int16',always_2d=True)
            if rate!=SR or not len(values) or not np.isfinite(values).all():raise RuntimeError('Invalid reference audio format.')
            recordings[i]=(values,hashlib.sha256(path.read_bytes()).hexdigest())
        return recordings[i]
    for index in selected:
        row=segments[index];voice=row.get('speaker_id')
        known=voice is not None and str(voice) not in ('','unknown')
        donors=[i for i,s in enumerate(segments) if i!=index and known and s.get('speaker_id')==voice
            and not s.get('deleted') and s.get('original_text','').strip()]
        donors=sorted(donors,key=lambda i:(abs(segments[i]['lo']-row['lo']),i))
        chosen=[];content_frames=len(read(index)[0]);minimum_frames=round(POLICY['minimum_seconds']*SR)
        for donor in donors:
            if content_frames>=minimum_frames:break
            chosen.append(donor);content_frames+=len(read(donor)[0])
        indices=[index,*sorted(chosen)]
        parts=[{'index':i,'sha256':read(i)[1],'text':without_speakers(segments[i]['original_text'])} for i in indices]
        text=parts[0]['text']+('\n<|speaker:1|>'+ ' '.join(part['text'] for part in parts[1:]) if len(parts)>1 else '')
        identity={'policy':POLICY,'parts':parts,'reference_text':text}
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        path=bank/f'{digest}.wav'
        padding=np.zeros((round(POLICY['outer_silence_seconds']*SR),read(index)[0].shape[1]),dtype=np.int16)
        gap=np.zeros((round(POLICY['reference_gap_seconds']*SR),read(index)[0].shape[1]),dtype=np.int16)
        if not path.exists():
            chunks=[padding]
            for slot,i in enumerate(indices):
                if slot:chunks.append(gap)
                chunks.append(read(i)[0])
            chunks.append(padding)
            sf.write(path,np.concatenate(chunks),SR,subtype='PCM_16')
        atomic_json(path.with_suffix('.json'),{**identity,'target_index':index,'reference_indices':indices,
            'reference_seconds':sf.info(path).duration,'content_seconds':sum(len(read(i)[0]) for i in indices)/SR,
            'outer_silence_frames_per_side':len(padding),'inter_dialogue_silence_frames':len(gap),
            'inter_dialogue_silence_count':len(indices)-1,'output_speaker_tag':None})
        row.update(_shared_reference_path=str(path),_shared_reference_text=text,
            _shared_reference_indices=indices,_shared_reference_speaker=voice,
            _shared_reference_donor_tag=False,_shared_reference_output_tag=None,
            _preserve_reference_tags=True,_preserve_speaker_tags=False,_omit_instructions=False,
            _structured_instructions=True,_reference_language_wrappers=True,_reference_language_parts=[part['text'] for part in parts])



def language_wrap(text,language):
    from core import LANGUAGE_CATALOG
    from tts_prompt import language_code
    code=language_code(language)
    if not code:return text
    tag=LANGUAGE_CATALOG['languages'][code]['name'].lower().replace(' ','_')
    return f'<{tag}>{text}</{tag}>'


def request_texts(reference_text,output_text,source_language,target_language,reference_parts=None):
    if reference_parts:
        current=reference_parts[0];donors=reference_parts[1:]
    else:
        current,marker,tail=reference_text.partition('<|speaker:1|>')
        current=current.strip();donors=[tail.strip()] if marker else []
    reference=language_wrap(current,source_language)
    if donors:reference+=' <|speaker:1|> '+' '.join(language_wrap(part,source_language) for part in donors)
    return reference,language_wrap(output_text,target_language)
