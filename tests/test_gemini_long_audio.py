from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import json
from gemini_source_pipeline import source
from scribe import prepare_input,AUDIO_BYTE_BUDGET


def test_three_hour_audio_uses_gemini_budget_without_reducing_native_sample_rate(tmp_path):
    folder=tmp_path/'input';folder.mkdir();source=tmp_path/'source.wav'
    flac=folder/'scribe-input-native.flac'
    # Sparse stand-in avoids encoding or storing a real three-hour recording.
    with flac.open('wb') as stream:stream.truncate(80_000_000)
    (folder/'input-native.json').write_text(json.dumps({'source_sha256':'digest'}),'utf8')
    info=SimpleNamespace(frames=10800*44100,samplerate=44100,channels=2,duration=10800)
    commands=[]
    def encode(args):
        commands.append(args)
        bitrate=int(args[args.index('-b:a')+1][:-1])
        with Path(args[-1]).open('wb') as stream:stream.truncate(10800*bitrate*1000//8)
    budget=(90_000_000-131072)//4*3
    with patch('scribe.sf.info',return_value=info),patch('scribe.command',side_effect=encode):
        audio,fmt,duration=prepare_input(source,folder,'digest',audio_byte_budget=budget)
    assert fmt=='mp3' and duration==10800
    assert audio.stat().st_size<=budget
    assert commands[0][commands[0].index('-b:a')+1]=='48k'
    assert commands[0][commands[0].index('-ar')+1]=='44100'
    assert AUDIO_BYTE_BUDGET<budget
