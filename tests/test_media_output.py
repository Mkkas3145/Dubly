import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from media_output import publish,resolve
class MediaOutputTests(unittest.TestCase):
 def test_locked_previous_file_keeps_playing_and_new_file_is_published(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);dest=folder/'final.mp4';temp=folder/'final.partial.mp4'
   dest.write_bytes(b'old');temp.write_bytes(b'new');original=Path.replace
   def replace(source,target):
    if Path(target)==dest:raise PermissionError('Open video')
    return original(source,target)
   with patch.object(Path,'replace',replace):actual=publish(temp,dest)
   self.assertEqual(dest.read_bytes(),b'old');self.assertEqual(resolve(dest),actual)
   self.assertEqual(actual.read_bytes(),b'new')
   temp.write_bytes(b'newest');self.assertEqual(publish(temp,dest),dest)
   self.assertEqual(resolve(dest),dest);self.assertEqual(dest.read_bytes(),b'newest')
 def test_pointer_cannot_escape_output_folder(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);(folder/'current-output.json').write_text('{"filename":"../other.mp4"}')
   with self.assertRaises(RuntimeError):resolve(folder/'final.mp4')
if __name__=='__main__':unittest.main()
