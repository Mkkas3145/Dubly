from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audio_refresh import copy_cached_inputs

class AudioRefreshCacheTests(unittest.TestCase):
    def test_reuses_alignment_cache_named_after_fitted_output(self):
        with tempfile.TemporaryDirectory() as temp:
            src=Path(temp)/'source';dst=Path(temp)/'dest';src.mkdir();dst.mkdir()
            name='0002-fitted-mfa.json';data=b'{"identity":"same speech"}'
            (src/name).write_bytes(data)
            copy_cached_inputs(src,dst,2)
            self.assertEqual((dst/name).read_bytes(),data)

    def test_does_not_overwrite_newer_retry_alignment_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            src=Path(temp)/'source';dst=Path(temp)/'dest';src.mkdir();dst.mkdir()
            name='0002-fitted-mfa.json'
            (src/name).write_bytes(b'old');(dst/name).write_bytes(b'new')
            copy_cached_inputs(src,dst,2)
            self.assertEqual((dst/name).read_bytes(),b'new')

if __name__=='__main__':unittest.main()
