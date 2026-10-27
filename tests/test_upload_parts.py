import json,math,uuid
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
import app as api
import upload_parts as parts
from storage import Store
from queue_worker import VideoQueue
from core import command,probe


def test_balanced_plan_prefers_silent_keyframes_and_has_no_gaps():
 info={'streams':[{'codec_type':'audio','index':1}]}
 with patch.object(parts,'_keys',side_effect=lambda source,a,b:[(a+b)/2]),patch.object(parts,'_quiet',side_effect=lambda source,a,b,index:[(a,b)]):
  plan=parts.plan(Path('source.mp4'),10800,info,max_seconds=1800)
 assert plan[0]['start']==0 and plan[-1]['end']==10800
 assert all(0<p['end']-p['start']<1800 for p in plan)
 assert all(a['end']==b['start'] for a,b in zip(plan,plan[1:]))
 assert all(p['boundary']=='silence_keyframe' for p in plan[:-1])


def test_missing_silence_uses_scene_before_duration_fallback():
 info={'streams':[{'codec_type':'audio','index':1}]}
 with patch.object(parts,'_keys',return_value=[]),patch.object(parts,'_quiet',return_value=[]),patch.object(parts,'_scenes',side_effect=lambda source,a,b:[(a+b)/2]):
  plan=parts.plan(Path('source.mp4'),60,info,max_seconds=30)
 assert all(p['boundary']=='scene' for p in plan[:-1])
 assert all(p['end']-p['start']<=30 for p in plan)


@pytest.fixture
def media(tmp_path):
 source=tmp_path/'long.mp4'
 command(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','testsrc2=size=160x90:rate=10:duration=13','-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=13','-f','lavfi','-i','sine=frequency=880:sample_rate=16000:duration=13','-filter_complex',"[1:a]volume=0:enable='between(t,4,4.8)+between(t,8.4,9.4)'[jp]",'-map','0:v','-map','[jp]','-map','2:a','-c:v','libx264','-preset','ultrafast','-g','30','-c:a','aac','-metadata:s:a:0','language=jpn','-metadata:s:a:1','language=eng','-disposition:a:0','default','-disposition:a:1','0','-shortest',str(source)])
 return source


def test_long_upload_becomes_ordered_independent_videos_and_preserves_audio_tracks(tmp_path,media):
 store=Store(tmp_path);queue=VideoQueue(store);(tmp_path/'work/videos').mkdir()
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(api,'MAX_PART_SECONDS',6),TestClient(api.app) as client:
  response=client.post('/api/videos',data={'source_language':'ja','target_language':'ko','tts_preset':'consistent'},files={'file':('Example.mp4',media.read_bytes(),'video/mp4')})
  assert response.status_code==201,response.text
  result=response.json();assert result['split'] and len(result['videos'])==3
  records=store.list();assert [r['filename'] for r in records]==['Example (1).mp4','Example (2).mp4','Example (3).mp4']
  assert all(r['state']=='uploaded' and r['source_language']=='ja' and r['target_language']=='ko' and r['tts_preset']=='consistent' for r in records)
  total=0;metadata=[]
  for record in records:
   folder=tmp_path/'work/videos'/record['id'];p=folder/'source.mp4';info=probe(p);seconds=float(info['format']['duration']);total+=seconds
   assert 0<seconds<=6
   assert [s['tags']['language'] for s in info['streams'] if s['codec_type']=='audio']==['jpn','eng']
   command(['ffmpeg','-nostdin','-v','error','-i',str(p),'-map','0:v:0','-map','0:a','-f','null','-'])
   metadata.append(json.loads((folder/'split-origin.json').read_text('utf8')))
  assert 13<=total<13.5
  assert all(a['end']==b['start'] for a,b in zip(metadata,metadata[1:]))
  first=result['id'];track=result['videos'][0]['audio_tracks']['tracks'][1]['index']
  assert client.post(f'/api/videos/{first}/audio-track',json={'index':track}).status_code==200
  for record in records:
   tracks=client.get(f'/api/videos/{record["id"]}/audio-tracks').json()
   assert not tracks['selection_required'] and tracks['tracks'][1]['index']==tracks['selected']


def test_failed_split_leaves_no_partial_upload_records(tmp_path,media):
 store=Store(tmp_path);queue=VideoQueue(store);(tmp_path/'work/videos').mkdir()
 actual=parts._write;calls=[]
 def write(*a):
  calls.append(1)
  if len(calls)==2:raise RuntimeError('분할 실패 테스트')
  return actual(*a)
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(api,'MAX_PART_SECONDS',6),patch.object(parts,'_write',side_effect=write),TestClient(api.app) as client:
  response=client.post('/api/videos',data={'source_language':'ja','target_language':'ko'},files={'file':('Example.mp4',media.read_bytes(),'video/mp4')})
  assert response.status_code==400
 assert store.list()==[] and list((tmp_path/'work/videos').iterdir())==[]


def test_cleanup_cannot_remove_parent_or_outside_directory(tmp_path):
 (tmp_path/'work/videos').mkdir(parents=True)
 with pytest.raises(RuntimeError):parts.cleanup([tmp_path],tmp_path)

def test_three_hour_default_accepts_full_recording_and_balances_longer_uploads():
 info={'streams':[{'codec_type':'audio','index':1}]}
 assert parts.MAX_PART_SECONDS==10800
 with patch.object(parts,'_keys',side_effect=AssertionError('No split below the limit')):
  assert parts.plan(Path('source.mp4'),10800,info)==[{'start':0.,'end':10800,'start_key':True,'end_key':True,'boundary':'whole'}]
 with patch.object(parts,'_keys',side_effect=lambda source,a,b:[(a+b)/2]),patch.object(parts,'_quiet',side_effect=lambda source,a,b,index:[(a,b)]):
  plan=parts.plan(Path('source.mp4'),4*3600,info)
 assert len(plan)==2
 assert all(0<p['end']-p['start']<=10800 for p in plan)
 assert plan[0]['start']==0 and plan[-1]['end']==14400 and plan[0]['end']==plan[1]['start']
