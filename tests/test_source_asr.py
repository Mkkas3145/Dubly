import io,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from source_asr import transcribe_words,MODEL


def test_fish_source_request_uses_word_times_and_reuses_only_own_cache():
    with tempfile.TemporaryDirectory() as temporary:
        p=Path(temporary);source=p/'voice.wav';sf.write(source,np.zeros(16000),16000)
        answer={'text':'hello','words':[{'word':'hello','start':.1,'end':.5}]}
        with patch('scribe.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(answer).encode())) as call:
            result=transcribe_words(source,p/'cache','test-key')
        payload=json.loads(call.call_args.args[0].data)
        assert call.call_count==1
        assert payload['model']==MODEL=='fish-audio/transcribe-1-pro'
        assert 'provider' not in payload
        assert payload['timestamp_granularities']==['word']
        assert result['signature']['model']==MODEL
        with patch('scribe.urllib.request.urlopen',side_effect=AssertionError('duplicate request')):
            assert transcribe_words(source,p/'cache','test-key')==result


def test_source_asr_is_one_whole_recording_attempt():
    with patch('source_asr._transcribe_words',return_value={'words':[]}) as call:
        transcribe_words('whole.wav','cache','test-key')
    call.assert_called_once()
    assert call.call_args.args[0]=='whole.wav'
    assert call.call_args.kwargs['attempts']==1
    assert call.call_args.kwargs['model']==MODEL
