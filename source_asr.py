"""One ASR request on the full separated dialogue, on its original clock."""
from scribe import transcribe_words as _transcribe_words
from rich_transcript import enrich,POLICY as RICH_POLICY
MODEL='fish-audio/transcribe-1-pro'

CACHE_DIRECTORY='source-asr-fish-transcribe-1-pro-full-v1'
INPUT_POLICY={'version':3,'input':'full_separated_dialogue','requests':1,
    'sample_rate':'preserve_until_size_requires_reduction','encoding':'native_flac_then_mp3',
    'channels':'preserve_up_to_stereo','tag_audio_events':True,'diarize':'auto',
    'silence':'preserve_original','removed_intervals':False,'inserted_intervals':False,
    'clock':'original_video_no_time_restoration','rich_transcript':RICH_POLICY}

def transcribe_words(source,folder,key,language=None,progress=lambda *args:None,*,timeline=None):
    # Fish Pro defaults retain emotion/event cues and detect speakers.
    # Send the whole recording without silence removal or clock remapping.
    result=enrich(_transcribe_words(source,folder,key,language,progress,attempts=1,model=MODEL))
    if timeline is None:return result
    from pathlib import Path
    from core import atomic_json
    folder=Path(folder)
    result.update(timestamp_clock='original_video',asr_input_duration=result.get('duration'),
        source_asr_input_policy=INPUT_POLICY)
    atomic_json(folder/'source-transcription.json',result)
    return result
