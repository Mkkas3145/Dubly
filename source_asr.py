"""One ASR request on the full separated dialogue, on its original clock."""
from scribe import transcribe_words as _transcribe_words
from rich_transcript import enrich,POLICY as RICH_POLICY
MODEL='fish-audio/transcribe-1-pro'

CACHE_DIRECTORY='source-asr-fish-transcribe-1-pro-full-v1'
INPUT_POLICY={'version':2,'input':'full_separated_dialogue','requests':1,
    'sample_rate':16000,'encoding':'flac','channels':1,
    'silence':'preserve_original','removed_intervals':False,'inserted_intervals':False,
    'clock':'original_video_no_time_restoration','rich_transcript':RICH_POLICY}

def transcribe_words(source,folder,key,language=None,progress=lambda *args:None,*,timeline=None):
    # Nemotron still controls dubbing boundaries and reference groups, but must
    # not remove context from the recording supplied to ASR. No clock remapping.
    result=enrich(_transcribe_words(source,folder,key,language,progress,attempts=1,model=MODEL))
    if timeline is None:return result
    from pathlib import Path
    from core import atomic_json
    folder=Path(folder)
    result.update(timestamp_clock='original_video',asr_input_duration=result.get('duration'),
        source_asr_input_policy=INPUT_POLICY)
    atomic_json(folder/'source-transcription.json',result)
    return result
