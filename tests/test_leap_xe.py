from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import numpy as np,soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from leap_xe import stream,virtual_part,prepare_pair

class LeapXeTests(unittest.TestCase):
 def test_explicit_leap_background_uses_the_shared_leap_pair(self):
  from background import prepare_background
  with patch('leap_xe.prepare_pair',return_value=('voice','background')) as pair:
   self.assertEqual(prepare_background(Path('source'),Path('work'),'progress',model_name='leap-xe-vocals'),'background')
   pair.assert_called_once_with(Path('source'),Path('work/runs/leap-xe-vocals-v1'),'progress')
 def test_virtual_context_matches_full_reflect_padding(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'x.wav'
   for size in [1,2,49,50,99,100,101,102,201,655]:
    wave=np.random.default_rng(size).normal(0,.1,(size,2)).astype('float32');sf.write(path,wave,44100,subtype='FLOAT')
    border=51 if size>102 else 0
    full=np.pad(wave,((border,border),(0,0)),mode='reflect') if border else wave
    with sf.SoundFile(path) as f:
     for start in range(0,len(full),50):
      got,n,length,_=virtual_part(f,start,101);part=full[start:start+101]
      expected=np.pad(part,((0,101-len(part)),(0,0)),mode='reflect' if len(part)>50 else 'constant')
      np.testing.assert_array_equal(got,expected);self.assertEqual(length,len(full));self.assertEqual(n,len(part))
 def test_stream_preserves_stereo_edges_and_odd_chunk_lengths(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for size in [1,2,50,100,101,102,103,401,2000]:
    with self.subTest(size=size):
     wave=np.random.default_rng(size).normal(0,.1,(size,2)).astype('float32');sf.write(root/'in.wav',wave,44100,subtype='FLOAT')
     stream(root/'in.wav',root/'out.wav',lambda x:x,lambda *a:None,chunk=101,batch_size=2)
     actual,sr=sf.read(root/'out.wav',dtype='float32',always_2d=True)
     self.assertEqual(sr,44100);np.testing.assert_allclose(actual,wave,atol=1e-7)
 def test_pair_reuses_cache_and_rejects_modified_stem(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);wave=np.random.default_rng(5).normal(0,.1,(500,2)).astype('float32');sf.write(root/'in.wav',wave,44100,subtype='FLOAT')
   def infer(src,dest,progress):sf.write(dest,sf.read(src,dtype='float32',always_2d=True)[0]*.25,44100,subtype='FLOAT')
   with patch('leap_xe.separate',side_effect=infer) as mock:
    voice,bg=prepare_pair(root/'in.wav',root/'pair',lambda *a:None)
    actual=sf.read(bg,dtype='float32',always_2d=True)[0];np.testing.assert_allclose(actual,wave*.75,atol=1e-7)
    prepare_pair(root/'in.wav',root/'pair',lambda *a:None);self.assertEqual(mock.call_count,1)
    sf.write(voice,np.zeros_like(wave),44100,subtype='FLOAT')
    prepare_pair(root/'in.wav',root/'pair',lambda *a:None);self.assertEqual(mock.call_count,2)
if __name__=='__main__':unittest.main()
