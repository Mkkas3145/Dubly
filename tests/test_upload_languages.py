import sqlite3,json,uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np,soundfile as sf
from fastapi.testclient import TestClient
import app as api
from storage import Store
from queue_worker import VideoQueue
from core import video_languages
from mai_pipeline import transcribe_source

def test_upload_languages_duration_and_single_file(tmp_path):
 store=Store(tmp_path);queue=VideoQueue(store)
 info={'streams':[{'codec_type':'audio'},{'codec_type':'video'}],'format':{'duration':'1800'}}
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(api,'probe',return_value=info),TestClient(api.app) as client:
  (tmp_path/'work/videos').mkdir(parents=True)
  def upload(data,files=None):return client.post('/api/videos',data=data,files=files or {'file':('test.mp4',b'mp4','video/mp4')})
  assert upload({'source_language':'en','target_language':'en'}).status_code==400
  assert upload({'source_language':'xx','target_language':'ko'}).status_code==422
  assert upload({'source_language':'en'}).status_code==400
  assert upload({'source_language':'en','target_language':'ko'},[('file',('a.mp4',b'a')),('file',('b.mp4',b'b'))]).status_code==400
  assert store.list()==[]
  r=upload({'source_language':'en','target_language':'ja'});assert r.status_code==201
  job=store.get(r.json()['id']);assert job['source_language']=='en' and job['target_language']=='ja'
  defaults={'language':'ko','key':'test'};cfg=video_languages(defaults,job);assert cfg['language']=='ja' and cfg['source_language']=='en';assert defaults['language']=='ko'
  info['format']['duration']='nan';assert upload({'source_language':'ja','target_language':'ko'}).status_code==400
  assert len(store.list())==1
  info['format']['duration']='10'
  with patch.object(api,'MAX_UPLOAD_BYTES',2):assert upload({'source_language':'ja','target_language':'ko'}).status_code==413
  assert len(store.list())==1

def test_existing_queue_migrates_without_changing_language(tmp_path):
 (tmp_path/'work').mkdir()
 db=sqlite3.connect(tmp_path/'work/queue.sqlite3');db.execute('CREATE TABLE videos(position INTEGER PRIMARY KEY,id TEXT,filename TEXT,state TEXT,phase TEXT,phase_progress REAL,progress REAL,detail TEXT,error TEXT,created REAL,updated REAL)');db.execute("INSERT INTO videos VALUES(1,'existing','old.mp4','done','완료',100,100,'',NULL,0,0)");db.commit();db.close()
 store=Store(tmp_path);job=store.get('existing');assert job['state']=='done' and job['source_language'] is None and job['target_language'] is None
 resolved=video_languages({'language':'ko'},job);assert resolved['language']=='ko' and resolved['tts_provider_options']=={'temperature':1.0,'top_p':1.0}

def test_source_language_hint_is_sent_and_part_of_cache(tmp_path):
 source=tmp_path/'audio.wav';sf.write(source,np.zeros(4410,np.float32),44100)
 raw={'segments':[{'start':0,'end':.1,'speaker':0,'text':'hello'}]};seen=[]
 class Reply:
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def read(self):return json.dumps(raw).encode()
 def request(req,**kwargs):seen.append(json.loads(req.data));return Reply()
 with patch('mai_pipeline.urllib.request.urlopen',side_effect=request):
  for language in ['en','en','ja']:transcribe_source(source,tmp_path/'asr','test',lambda *a:None,language=language)
 assert [p['language'] for p in seen]==['en','ja']


def test_expanded_language_catalog_validates_and_persists_every_selectable_language(tmp_path):
 from core import LANGUAGE_CATALOG,SOURCE_LANGUAGES,TARGET_LANGUAGES,LANGUAGES
 assert len(SOURCE_LANGUAGES)>=70 and len(TARGET_LANGUAGES)==83
 assert SOURCE_LANGUAGES==set(LANGUAGE_CATALOG['model_languages']['scribe_v2'])
 store=Store(tmp_path)
 for source in SOURCE_LANGUAGES:
  target='ko' if source!='ko' else 'en';id=str(uuid.uuid4());job=store.add(id,'source.mp4',source_language=source,target_language=target)
  cfg=video_languages({'language':'ko'},job);assert cfg['source_language']==source and cfg['language']==target
 for target in TARGET_LANGUAGES:
  source='en' if target!='en' else 'ja';id=str(uuid.uuid4());job=store.add(id,'target.mp4',source_language=source,target_language=target)
  assert video_languages({},job)['language']==target and target in LANGUAGES
  row=LANGUAGE_CATALOG['languages'][target];assert len(row['iso639_3'])==3
  assert all(row['labels'].get(locale) for locale in ['ko','en','ja'])


def test_new_language_uploads_and_catalog_endpoint_share_one_source_of_truth(tmp_path):
 from core import LANGUAGE_CATALOG
 store=Store(tmp_path);queue=VideoQueue(store);info={'streams':[{'codec_type':'audio'},{'codec_type':'video'}],'format':{'duration':'10'}}
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(api,'probe',return_value=info),TestClient(api.app) as client:
  (tmp_path/'work/videos').mkdir(parents=True)
  script=client.get('/static/languages.js');assert script.status_code==200
  assert json.loads(script.text.removeprefix('window.DublyLanguages=').removesuffix(';'))==LANGUAGE_CATALOG
  for source,target in [('fr','de'),('zh','ar'),('es','hi'),('as','zh'),('nb','nn'),('fil','jv')]:
   response=client.post('/api/videos',data={'source_language':source,'target_language':target},files={'file':('sample.mp4',b'mp4','video/mp4')})
   assert response.status_code==201,response.text
   job=store.get(response.json()['id']);assert job['source_language']==source and job['target_language']==target
  for source,target in [('fr','fr'),('xx','ko'),('en','xx'),('xx-ASR','ko'),('en','or')]:
   response=client.post('/api/videos',data={'source_language':source,'target_language':target},files={'file':('sample.mp4',b'mp4','video/mp4')})
   assert response.status_code in {400,422},(source,target,response.text)


def test_auto_source_upload_persists_selection_but_sends_no_language_hint(tmp_path):
 store=Store(tmp_path);queue=VideoQueue(store);info={'streams':[{'codec_type':'audio'},{'codec_type':'video'}],'format':{'duration':'10'}}
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(api,'probe',return_value=info),TestClient(api.app) as client:
  (tmp_path/'work/videos').mkdir(parents=True)
  response=client.post('/api/videos',data={'source_language':'auto','target_language':'ja'},files={'file':('sample.mp4',b'mp4','video/mp4')})
  assert response.status_code==201
  job=store.get(response.json()['id']);assert job['source_language']=='auto'
  cfg=video_languages({'source_language':'ko'},job);assert cfg['source_language'] is None and cfg['source_language_auto'] and cfg['language']=='ja'
  assert client.post('/api/videos',data={'source_language':'ja','target_language':'auto'},files={'file':('sample.mp4',b'mp4','video/mp4')}).status_code==422
 source=tmp_path/'audio.wav';sf.write(source,np.zeros(4410,np.float32),44100);seen=[]
 raw={'language':'English','segments':[{'start':0,'end':.1,'speaker':0,'text':'hello'}]}
 class Reply:
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def read(self):return json.dumps(raw).encode()
 def request(req,**kwargs):seen.append(json.loads(req.data));return Reply()
 with patch('mai_pipeline.urllib.request.urlopen',side_effect=request):
  for language in [None,None,'en']:transcribe_source(source,tmp_path/'asr','test',lambda *a:None,language=language)
 assert len(seen)==2 and 'language' not in seen[0] and seen[1]['language']=='en'


def test_detected_language_is_recorded_and_equal_target_stops_before_later_models(tmp_path,monkeypatch):
 import pytest
 from mai_native_pipeline import record_auto_language,detected_languages
 id=str(uuid.uuid4());store=Store(tmp_path);store.add(id,'auto.mp4',source_language='auto',target_language='ko')
 monkeypatch.setenv('DUBLY_VIDEO_ID',id)
 with patch('storage.Store',return_value=store):
  record_auto_language({'language':'Japanese'},{'source_language_auto':True,'language':'ko'})
  job=store.get(id);assert job['source_language']=='auto' and job['detected_source_language']=='ja'
  with pytest.raises(RuntimeError,match='출력 언어가 같습니다'):record_auto_language({'language':'ja-JP'},{'source_language_auto':True,'language':'ja'})
  # An explicitly multilingual recording is not rejected as a single-language match.
  record_auto_language({'language':'ja','segments':[{'language':'ko'}]},{'source_language_auto':True,'language':'ja'})
 assert detected_languages({'segments':[{'language':'en'}]})==('en',{'en'})
 assert detected_languages({'language':'unknown'})==(None,set())
