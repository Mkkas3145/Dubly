from pathlib import Path
import numpy as np,soundfile as sf
from core import command
from qwen_audio_pipeline import encode_audio

def test_low_bitrate_uses_mpeg2_instead_of_silent_32k_clamp(tmp_path):
 source=tmp_path/'source.flac';sf.write(source,np.zeros(441000,np.float32),44100)
 folder=tmp_path/'encoded';folder.mkdir()
 path,manifest=encode_audio(source,folder,'test-hash',lambda *a:None,base64_limit=50000)
 assert manifest['bitrate_kbps']==24 and manifest['sample_rate']==24000
 assert manifest['base64_bytes']<=50000 and abs(sf.info(path).duration-10)<.1
 assert manifest['silence_removed'] is False and manifest['audio_parts']==1
