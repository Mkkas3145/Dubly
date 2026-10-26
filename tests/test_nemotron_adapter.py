import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import native_timeline,nemotron_boundaries
class NemotronAdapterTests(unittest.TestCase):
 def test_selected_backend_has_no_diarizen_fallback(self):
  with patch('nemotron_boundaries.prepare',return_value={'turns':[]}) as run:
   native_timeline.prepare_selected(Path('source.wav'),{})
   self.assertEqual(run.call_count,1)
  with self.assertRaises(RuntimeError):
   native_timeline.prepare_selected(Path('source.wav'),{'diarizer':'diarizen'})
 def test_matching_cache_does_not_launch_worker(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);data={'identity':{'hash':'good'},'boundary_authority':'nemotron_eend_vc','turns':[]}
   (root/'timeline.json').write_text(json.dumps(data))
   with patch('nemotron_boundaries.identity_for',return_value={'hash':'good'}),patch('nemotron_boundaries.subprocess.Popen',side_effect=AssertionError('Cached video must not reload weights')):
    self.assertEqual(nemotron_boundaries.prepare(root/'source.wav',root)['raw_turns'],data['turns'])
class BoundaryMergeTests(unittest.TestCase):
 def merge(self,turns):
  return nemotron_boundaries.merge_same_speaker({'identity':{},'turns':turns})
 def test_short_dropout_keeps_entire_voice(self):
  result=self.merge([{'start':128.98,'end':129.06,'speaker':'E4'},{'start':129.37,'end':131.73,'speaker':'E4'}])
  self.assertEqual(result['turns'],[{'start':128.98,'end':131.73,'speaker':'E4'}])
  self.assertEqual(len(result['raw_turns']),2)
 def test_merge_preserves_source_identity_for_translation_audio(self):
  source={'identity':{'source_sha256':'audio-hash','revision':'pinned'},'turns':[]}
  result=nemotron_boundaries.merge_same_speaker(source)
  self.assertEqual(result['identity']['source_sha256'],'audio-hash')
  self.assertEqual(result['identity']['revision'],'pinned')
  self.assertIn('boundary_merge_policy',result['identity'])
 def test_threshold_and_other_voice_are_respected(self):
  def turn(a,b,s):return {'start':a,'end':b,'speaker':s}
  self.assertEqual(len(self.merge([turn(0,1,'A'),turn(1.8,2,'A')])['turns']),1)
  self.assertEqual(len(self.merge([turn(0,1,'A'),turn(1.81,2,'A')])['turns']),2)
  self.assertEqual(len(self.merge([turn(0,1,'A'),turn(1.2,1.3,'B'),turn(1.4,2,'A')])['turns']),3)
  self.assertEqual(len(self.merge([turn(0,1.5,'B'),turn(.1,1,'A'),turn(1.4,2,'A')])['turns']),3)
if __name__=='__main__':unittest.main()
