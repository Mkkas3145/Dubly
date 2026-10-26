import unittest
import numpy as np
from nemotron_linking import link,Components

def window(start,active,vectors=None):
    active=np.asarray(active,dtype=bool)
    return {'start_frame':start,'end_frame':start+len(active),'active':active,
            'embeddings':vectors or {}}

class NemotronLinkingTests(unittest.TestCase):
    def test_animation_profile_keeps_native_activity_and_stricter_identity_calibration(self):
        from nemotron_policy import PROFILE,WINDOW_POLICY
        from nemotron_linking import POLICY
        self.assertEqual(PROFILE,POLICY['profile'])
        self.assertEqual(WINDOW_POLICY['activity_threshold'],.5)
        self.assertEqual(WINDOW_POLICY['embedding_pure_probability'],.9)
        self.assertEqual(WINDOW_POLICY['embedding_max_samples'],8)
        self.assertEqual(WINDOW_POLICY['embedding_sample_seconds'],2.)
        self.assertEqual(POLICY['negative_quantile'],.999)
        self.assertIsNone(POLICY['global_speaker_cap'])

    def test_overlap_permutation_links_exact_shared_speech(self):
        # A then B. Local IDs are deliberately reversed in the second window.
        a=np.zeros((300,2),bool);a[:150,0]=1;a[150:,1]=1
        b=np.zeros((300,2),bool);b[:50,1]=1;b[50:,0]=1
        turns,audit=link([window(0,a),window(100,b)],400)
        self.assertEqual([(t['start'],t['end'],t['speaker']) for t in turns],[(0.,1.5,'E0'),(1.5,4.,'E1')])
        self.assertEqual(audit['global_speakers'],2)
        self.assertEqual(len(audit['overlap_anchors']),2)

    def test_more_than_sixteen_speakers_without_global_cap(self):
        windows=[];count=24
        for wi in range(3):
            active=np.zeros((800,8),bool)
            for i in range(8):active[i*100:(i+1)*100,i]=1
            samples={i:[np.eye(count)[wi*8+i],np.eye(count)[wi*8+i]] for i in range(8)}
            windows.append(window(wi*800,active,samples))
        turns,audit=link(windows,2400)
        self.assertEqual(len({t['speaker'] for t in turns}),24)
        self.assertEqual(audit['global_speakers'],24)

    def test_returning_speaker_linked_by_embedding_without_overlap(self):
        active=np.zeros((200,2),bool);active[:100,0]=1;active[100:,1]=1
        vectors={0:[np.array([1,0,0]),np.array([1,0,0])],1:[np.array([0,1,0]),np.array([0,1,0])]}
        later={0:[np.array([1,0,0]),np.array([1,0,0])],1:[np.array([0,0,1]),np.array([0,0,1])]}
        turns,audit=link([window(0,active,vectors),window(200,active,later)],400)
        self.assertEqual(audit['global_speakers'],3)
        self.assertEqual(turns[0]['speaker'],turns[2]['speaker'])
        self.assertEqual(len(audit['embedding_merges']),1)

    def test_local_distinct_speakers_are_hard_constraints(self):
        components=Components([{'window':0},{'window':0},{'window':1}])
        self.assertFalse(components.join(0,1))
        self.assertTrue(components.join(0,2))
        self.assertFalse(components.join(1,2))

    def test_advisory_short_view_does_not_lock_in_a_false_identity_split(self):
        primary=window(0,np.ones((200,1),bool))
        split=np.zeros((200,2),bool);split[:100,0]=1;split[100:,1]=1
        advisory={**window(0,split),'trusted_constraints':False}
        edges=[{'nodes':[[0,0],[1,0]]},{'nodes':[[0,0],[1,1]]}]
        _,audit=link([primary,advisory],200,verified_edges=edges,use_embeddings=False)
        self.assertEqual(audit['global_speakers'],1)
        self.assertTrue(all(e['accepted'] for e in audit['same_model_verifications']))

    def test_insufficient_embeddings_do_not_guess_new_identity_matches(self):
        a=np.ones((100,1),bool)
        turns,audit=link([window(0,a),window(100,a)],200)
        self.assertEqual(audit['global_speakers'],2)
        self.assertIn('insufficient',audit['calibration']['status'])

    def test_seam_ownership_keeps_overlapping_speakers_and_no_duplicate_frames(self):
        a=np.ones((300,2),bool)
        turns,audit=link([window(0,a)],300)
        self.assertEqual(len(turns),2)
        self.assertTrue(all(t['start']==0 and t['end']==3 for t in turns))
        with self.assertRaises(ValueError):link([window(20,a)],320)

if __name__=='__main__':unittest.main()
