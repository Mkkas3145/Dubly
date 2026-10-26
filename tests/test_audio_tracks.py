import io,uuid,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,soundfile as sf,pytest
from fastapi.testclient import TestClient
import app as api
from core import command,probe
from storage import Store
from queue_worker import VideoQueue
from audio_tracks import initialize,select,selected,extract,inspect
from browser_preview import prepare


def sample(path):
 path.parent.mkdir(parents=True,exist_ok=True)
 command(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','color=c=black:s=160x90:r=10:d=2',
  '-f','lavfi','-i','sine=frequency=440:duration=2','-f','lavfi','-i','sine=frequency=880:duration=2',
  '-map','0:v','-map','1:a','-map','2:a','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',
  '-metadata:s:a:0','language=eng','-metadata:s:a:0','title=English Dub',
  '-metadata:s:a:1','language=jpn','-metadata:s:a:1','title=Japanese Original',str(path)])


def frequency(values,rate):
 if values.ndim>1:values=values.mean(axis=1)
 return np.argmax(np.abs(np.fft.rfft(values)))*rate/len(values)


def test_selected_track_drives_both_extraction_and_browser_proxy(tmp_path):
 id=str(uuid.uuid4());folder=tmp_path/'work/videos'/id;source=folder/'source.mp4';sample(source)
 data=initialize(id,probe(source),tmp_path)
 assert len(data['tracks'])==2 and data['selection_required']
 with pytest.raises(RuntimeError):selected(id,tmp_path)
 select(id,2,tmp_path);assert selected(id,tmp_path)==2
 extract(source,folder/'original.wav',selected(id,tmp_path),2)
 x,sr=sf.read(folder/'original.wav');assert abs(frequency(x,sr)-880)<2
 preview=prepare(id,tmp_path);info=probe(preview)
 audio=[s for s in info['streams'] if s['codec_type']=='audio']
 assert len(audio)==1 and audio[0]['tags']['language']=='jpn'
 extract(preview,folder/'preview.wav',audio[0]['index'],2)
 x,sr=sf.read(folder/'preview.wav');assert abs(frequency(x,sr)-880)<2
 select(id,1,tmp_path);prepare(id,tmp_path)
 assert [s for s in probe(preview)['streams'] if s['codec_type']=='audio'][0]['tags']['language']=='eng'


def test_upload_requires_valid_explicit_choice_before_analysis(tmp_path):
 idroot=tmp_path/'fixture/source.mp4';sample(idroot)
 store=Store(tmp_path);queue=VideoQueue(store)
 (tmp_path/'work/videos').mkdir(parents=True,exist_ok=True)
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start'),TestClient(api.app) as client:
  r=client.post('/api/videos',data={'source_language':'auto','target_language':'ko'},files={'file':('dual.mp4',idroot.read_bytes(),'video/mp4')})
  assert r.status_code==201;id=r.json()['id'];base='/api/videos/'+id
  assert r.json()['audio_tracks']['selection_required']
  assert client.post('/api/studio/'+id+'/analyze',json={}).status_code==409
  for invalid in [0,3,True,'1']:
   assert client.post(base+'/audio-track',json={'index':invalid}).status_code==422
  assert client.post(base+'/audio-track',json={'index':2}).status_code==200
  assert client.get(base+'/audio-track-preview?index=999').status_code==422
  audio=client.get(base+'/audio-track-preview?index=2');assert audio.status_code==200 and len(audio.content)>1000
  assert client.post('/api/studio/'+id+'/analyze',json={}).status_code==200
  assert client.post(base+'/audio-track',json={'index':1}).status_code==409
