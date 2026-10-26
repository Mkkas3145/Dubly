import copy,unittest
from render import validate_mux

class MuxValidationTests(unittest.TestCase):
    def case(self):
        source={'format':{'duration':'1446.236033','start_time':'0'},'streams':[
            {'codec_type':'video','start_time':'.083','duration':'1446.153033'}]}
        result={'format':{'duration':'1446.570'},'streams':[
            {'codec_type':'video','start_time':'.083','duration':'1446.153033'},
            {'codec_type':'audio','start_time':'0','duration':'1446.236009'},
            {'codec_type':'subtitle','start_time':'0','duration':'1446.570'}]}
        return source,result

    def test_subtitle_tail_does_not_falsely_reject_intact_media(self):
        source,result=self.case();report=validate_mux(result,source,1446.236033)
        self.assertTrue(report['passed'])

    def test_short_audio_is_still_rejected_even_when_container_looks_right(self):
        source,result=self.case();result['streams'][1]['duration']='1440'
        self.assertFalse(validate_mux(result,source,1446.236033)['passed'])

    def test_short_video_is_still_rejected(self):
        source,result=self.case();result['streams'][0]['duration']='1440'
        self.assertFalse(validate_mux(result,source,1446.236033)['passed'])

    def test_missing_audio_is_rejected(self):
        source,result=self.case();del result['streams'][1]
        with self.assertRaisesRegex(RuntimeError,'트랙이 없습니다'):validate_mux(result,source,1446.236033)

    def test_tts_tail_may_extend_past_original_video(self):
        source,result=self.case();result['streams'][1]['duration']='1448.0'
        self.assertTrue(validate_mux(result,source,1448.)['passed'])

if __name__=='__main__':unittest.main()
