import unittest,tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from core import SR
from render import reference_audio,prepare_segments

class NaturalMarginTests(unittest.TestCase):
    def render_reference(self,values,segment):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source.wav';dest=Path(temp)/'reference.wav'
            sf.write(source,values,SR,subtype='FLOAT')
            reference_audio(source,segment,dest)
            return sf.read(dest,dtype='int16')[0]

    def test_exact_clean_margin_preserves_core_and_tight_timing(self):
        values=np.full((3*SR,2),.0001,dtype=np.float32)
        values[SR:2*SR]=.1
        segment=prepare_segments({'segments':[{'start':1.,'end':2.}]},len(values))[0]
        extended=self.render_reference(values,segment)
        tight=self.render_reference(values,{'lo':SR,'hi':2*SR})
        self.assertEqual(len(extended),SR+2*round(.2*SR))
        np.testing.assert_array_equal(extended[round(.2*SR):-round(.2*SR)],tight)
        self.assertEqual((segment['lo'],segment['hi']),(SR,2*SR))

    def test_stops_before_attack_never_skips_over_noise(self):
        values=np.zeros((3*SR,2),dtype=np.float32);values[SR:2*SR]=.1
        values[round(2.08*SR):round(2.1*SR)]=.2
        segment=prepare_segments({'segments':[{'start':1.,'end':2.}]},len(values))[0]
        result=self.render_reference(values,segment)
        tail=len(result)-SR-round(.2*SR)
        self.assertGreater(tail,round(.07*SR));self.assertLessEqual(tail,round(.08*SR))
        self.assertLess(np.max(np.abs(result[-tail:])),10)

    def test_known_exclusion_and_asr_boundaries_limit_quiet_margin(self):
        values=np.zeros((3*SR,2),dtype=np.float32);values[SR:2*SR]=.1
        segment=prepare_segments({'segments':[{'start':1.,'end':2.}],
            'excluded_intervals':[{'start':2.05,'end':2.2}]},len(values),
            {'words':[{'start':.7,'end':.96}]})[0]
        result=self.render_reference(values,segment)
        self.assertLessEqual(len(result)-SR,round(.09*SR))
        self.assertGreater(len(result),SR)

    def test_active_neighbor_means_no_added_margin(self):
        values=np.full((3*SR,2),.1,dtype=np.float32)
        segment=prepare_segments({'segments':[{'start':1.,'end':2.}]},len(values))[0]
        self.assertEqual(len(self.render_reference(values,segment)),SR)

if __name__=='__main__':unittest.main()
