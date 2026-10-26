import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import qwen_recovery_asr as recovery


class RecoveryCacheConcurrencyTests(unittest.TestCase):
    def test_transient_windows_replace_lock_is_retried_without_repeating_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination=Path(tmp)/'qwen.json';data={'text':'cached response'}
            real=recovery.atomic_json;attempts=[]
            def replace(path,value):
                attempts.append(path)
                if len(attempts)==1:raise PermissionError('Windows sharing violation')
                return real(path,value)
            with patch.object(recovery,'atomic_json',side_effect=replace),patch.object(recovery.time,'sleep'):
                recovery.write_cache(destination,data)
            self.assertEqual(len(attempts),2)
            self.assertEqual(json.loads(destination.read_text('utf8')),data)

    def test_locked_parallel_cache_is_skipped_and_identical_paid_recognition_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.wav';source.write_bytes(b'actual source content')
            target=root/'target';bad=root/'parallel'/'qwen.json';good=root/'completed'/'qwen.json'
            bad.parent.mkdir();good.parent.mkdir();bad.write_text('{}','utf8')
            identity=dict(model=recovery.MODEL,version=recovery.VERSION,
                audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),language='ja',
                response_format='verbose_json',timestamp_granularities=['word'],
                provider_options={'azure':{'diarization':{'enabled':True},'enhancedMode':{'modelOptions':{'transcribeStyle':'verbatim'}}}})
            expected=dict(identity=identity,text='previously paid recognition')
            good.write_text(json.dumps(expected),'utf8');original=Path.read_text
            def read(path,*args,**kwargs):
                if path==bad:raise PermissionError('parallel atomic replacement')
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',read),patch.object(Path,'glob',return_value=iter([bad,good])),\
                 patch.object(recovery,'command',side_effect=AssertionError('must reuse cache')),\
                 patch('urllib.request.urlopen',side_effect=AssertionError('must not request ASR again')):
                result=recovery.transcribe(source,target,'unused','ja')
            self.assertEqual(result,expected)
            self.assertEqual(json.loads((target/'qwen.json').read_text('utf8')),expected)


if __name__=='__main__':unittest.main()
