import unittest
import numpy as np
from diarizen_stability import stabilize

class StabilityTests(unittest.TestCase):
    def test_uncertain_switchback_restores_continuous_speech(self):
        votes=np.tile([.6,.4],(40,1));votes[7:11]=[.49,.51]
        active=np.eye(2)[votes.argmax(1)]
        result,reports=stabilize(votes,active)
        self.assertTrue(np.all(result[:,0]==1));self.assertGreater(len(reports),0)
    def test_short_confident_different_voice_is_preserved(self):
        votes=np.tile([.99,.01],(40,1));votes[7:9]=[.01,.99]
        active=np.eye(2)[votes.argmax(1)]
        result,_=stabilize(votes,active)
        np.testing.assert_array_equal(result,active)
    def test_silence_and_overlap_are_not_crossed_or_changed(self):
        votes=np.tile([.6,.4],(40,1));active=np.eye(2)[votes.argmax(1)]
        active[10]=0;active[20]=1;active[25:30]=[0,1];votes[25:30]=[.01,.99]
        result,_=stabilize(votes,active)
        np.testing.assert_array_equal(result[[10,20]],active[[10,20]])
        np.testing.assert_array_equal(result[25:30],active[25:30])
    def test_clear_sustained_turn_change_survives(self):
        votes=np.r_[np.tile([.7,.3],(50,1)),np.tile([.3,.7],(50,1))]
        active=np.eye(2)[votes.argmax(1)]
        result,_=stabilize(votes,active)
        np.testing.assert_array_equal(result,active)
if __name__=='__main__':unittest.main()
