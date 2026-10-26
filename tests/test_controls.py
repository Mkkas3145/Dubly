from pathlib import Path
import os
import sys
import tempfile
import time
import threading
import unittest
import uuid
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import app as api
from storage import Store
from queue_worker import VideoQueue


class ControlTests(unittest.TestCase):
    def test_delete_requires_confirmation_and_removes_only_selected_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);store=Store(root);queue=VideoQueue(store)
            ids=[str(uuid.uuid4()) for _ in range(2)]
            for video_id in ids:
                store.add(video_id,'video.mp4')
                for base in ['work/videos','outputs']:
                    folder=root/base/video_id;folder.mkdir(parents=True);(folder/'file.bin').write_bytes(b'cached')
                (root/'logs').mkdir(exist_ok=True);(root/'logs'/f'{video_id}.log').write_text('log')
            original=root/'original.mp4';original.write_bytes(b'original')
            with patch.object(api,'store',store),patch.object(api,'queue',queue),TestClient(api.app) as client:
                path=f'/api/videos/{ids[0]}'
                self.assertEqual(client.delete(path).status_code,422)
                self.assertEqual(client.request('DELETE',path,json={'confirmed':False}).status_code,422)
                self.assertIsNotNone(store.get(ids[0]))
                self.assertEqual(client.request('DELETE',path,json={'confirmed':True}).status_code,200)
                self.assertIsNone(store.get(ids[0]))
                for base in ['work/videos','outputs']:
                    self.assertFalse((root/base/ids[0]).exists());self.assertTrue((root/base/ids[1]/'file.bin').exists())
                self.assertFalse((root/'logs'/f'{ids[0]}.log').exists())
                self.assertEqual(original.read_bytes(),b'original')
                store.update(ids[1],state='running')
                self.assertEqual(client.request('DELETE',f'/api/videos/{ids[1]}',json={'confirmed':True}).status_code,409)
                self.assertTrue((root/'work/videos'/ids[1]/'file.bin').exists())
                self.assertEqual(client.request('DELETE','/api/videos/not-a-uuid',json={'confirmed':True}).status_code,404)

    def test_stop_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=Store(Path(temporary));queue=VideoQueue(store)
            with patch.object(api,'store',store),patch.object(api,'queue',queue),TestClient(api.app) as client:
                with patch.object(queue,'stop') as stop:
                    self.assertEqual(client.post('/api/stop').status_code,422)
                    self.assertEqual(client.post('/api/stop',json={'confirmed':False}).status_code,422)
                    stop.assert_not_called()
                    self.assertEqual(client.post('/api/stop',json={'confirmed':True}).status_code,200)
                    stop.assert_called_once()

    def test_deleting_job_is_not_claimed_and_delete_failure_is_retryable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);store=Store(root);video_id=str(uuid.uuid4());store.add(video_id,'a.mp4')
            folder=root/'work/videos'/video_id;folder.mkdir(parents=True);(folder/'x').touch()
            with patch('storage.shutil.rmtree',side_effect=PermissionError('locked')):
                with self.assertRaises(RuntimeError):store.delete(video_id)
            self.assertEqual(store.get(video_id)['state'],'error');self.assertIsNone(store.claim())
            store.delete(video_id);self.assertIsNone(store.get(video_id));self.assertFalse(folder.exists())

    def test_linked_video_directory_never_deletes_external_files(self):
        with tempfile.TemporaryDirectory() as temporary,tempfile.TemporaryDirectory() as outside:
            root=Path(temporary);store=Store(root);video_id=str(uuid.uuid4());store.add(video_id,'a.mp4')
            external=Path(outside);(external/'keep').write_text('keep');folder=root/'work/videos'/video_id;folder.parent.mkdir(parents=True)
            try:folder.symlink_to(external,target_is_directory=True)
            except OSError:self.skipTest('Windows symlink privileges unavailable')
            try:
                with self.assertRaises(RuntimeError):store.delete(video_id)
                self.assertTrue((external/'keep').exists());self.assertIsNotNone(store.get(video_id))
            finally:folder.unlink()

    def test_immediate_stop_kills_worker_and_child_and_keeps_next_queued(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);store=Store(root)
            store.add('a','a.mp4');store.add('b','b.mp4')
            child="import time; from pathlib import Path\nwhile True:\n with Path('heartbeat').open('a') as f:f.write('x')\n time.sleep(.03)\n"
            (root/'pipeline.py').write_text('import subprocess,sys,time\nfrom pathlib import Path\n'+
                f'subprocess.Popen([sys.executable,"-c",{child!r}])\n'+
                'Path("ready").touch()\nwhile True:time.sleep(.1)\n',encoding='utf-8')
            queue=VideoQueue(store);queue.start();thread=queue.thread
            try:
                deadline=time.monotonic()+8
                while time.monotonic()<deadline and not (root/'heartbeat').exists():time.sleep(.03)
                self.assertTrue((root/'heartbeat').exists())
                queue.stop();thread.join(8)
                self.assertFalse(queue.running)
                self.assertEqual(store.get('a')['state'],'cancelled');self.assertEqual(store.get('b')['state'],'queued')
                size=(root/'heartbeat').stat().st_size;time.sleep(.15)
                self.assertEqual((root/'heartbeat').stat().st_size,size)
                queue.retry('a');self.assertEqual(store.get('a')['state'],'queued')
            finally:
                queue.stop()
                if thread:thread.join(8)

    def test_stop_before_worker_spawn_and_active_deletion_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);store=Store(root);video_id=str(uuid.uuid4());store.add(video_id,'a.mp4')
            entered=threading.Event();release=threading.Event()
            def process(job):entered.set();release.wait(5)
            queue=VideoQueue(store,process);queue.start();thread=queue.thread
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(RuntimeError):queue.delete(video_id)
                queue.stop();release.set();thread.join(5)
                self.assertEqual(store.get(video_id)['state'],'cancelled')
                queue.delete(video_id);self.assertIsNone(store.get(video_id))
            finally:release.set();thread.join(5)


if __name__=='__main__':unittest.main()
