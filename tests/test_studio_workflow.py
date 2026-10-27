import json,uuid,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np,soundfile as sf,pytest
from fastapi.testclient import TestClient
import app as api
from core import SR
from storage import Store
from queue_worker import VideoQueue
from studio_workflow import checkpoint,load,save,row_key,clip_file,clip_info,edited_audio,validate_edit,paths

def fixture(tmp_path):
 video_id=str(uuid.uuid4());store=Store(tmp_path);store.add(video_id,'sample.mp4',source_language='ja',target_language='ko');store.update(video_id,state='ready',phase='편집 준비 완료')
 folder=tmp_path/'work/videos'/video_id;folder.mkdir(parents=True)
 t=np.arange(SR)/SR;speech=(.2*np.sin(t*2*np.pi*220)).astype(np.float32)
 values=np.concatenate([speech,np.zeros(SR//4,np.float32),speech,np.zeros(SR//4,np.float32),speech]);values=np.column_stack([values,values])
 clean=folder/'dialogue.wav';sf.write(clean,values,SR,subtype='FLOAT')
 transcript={'segments':[{'start':0,'end':1,'original_text':'はい！','translated_text':'네!','speaker_id':'0'}, {'start':1.3,'end':2.3,'original_text':'父上。','translated_text':'아버지.','speaker_id':'1'}],'target_language':'ko','reference_windows':None}
 checkpoint(video_id,transcript,{'clean':str(clean),'original':str(clean),'source':str(folder/'source.mp4'),'background':None,'duration':len(values)/SR,'language':'ko'},tmp_path)
 p=clip_file(video_id,0,tmp_path);p.parent.mkdir();sf.write(p,values,SR,subtype='FLOAT');doc,ctx=load(video_id,tmp_path)
 from studio_workflow import audio_processing_policy
 doc['generated']['0']={'input_hash':row_key(doc,0),'seconds':len(values)/SR,'processing_policy':audio_processing_policy()}
 from core import atomic_json
 atomic_json(paths(video_id,tmp_path)[1],doc)
 return video_id,store,doc,ctx,values

def body(doc):return {'revision':doc['revision'],'speaker_names':doc.get('speaker_names',{}),'segments':[{'id':i,**{k:r[k] for k in ('original_text','translated_text','speaker_id','keep_original','audio_edit')}} for i,r in enumerate(doc['transcript']['segments'])]}


def test_generation_publishes_each_completed_clip_before_the_whole_job_finishes(tmp_path,monkeypatch):
 import studio_workflow as workflow
 import render
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 doc['generated']['0']['processing_policy']={}
 atomic_json(paths(id,tmp_path)[1],doc)
 original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store)
 monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','tts_reference_mode':'individual','tts_mode':'individual','speaker_embeddings':False})
 monkeypatch.setattr(render,'generate_speech',lambda i,*a,**kw:{'index':i})
 seen=[]
 def fit(prepared,cfg):
  i=prepared['index'];sf.write(clip_file(id,i,tmp_path),x[:SR//2],SR,subtype='FLOAT')
  return {'index':i}
 monkeypatch.setattr(render,'fit_speech',fit)
 def queue(items,generator,fitter,progress):
  reports=[]
  for i,row in enumerate(items):
   reports.append(fitter(generator(i,row)))
   current,context=load(id,tmp_path)
   snapshot=workflow.public(current,context,id,tmp_path)
   seen.append(snapshot)
  return reports
 monkeypatch.setattr(render,'pipelined_queue',queue)
 assert workflow.generate(id,'tts')==0
 assert seen[0]['segments'][0]['audio_ready'] and not seen[0]['segments'][0]['audio_processing']
 assert seen[0]['segments'][0]['audio_key']
 assert not seen[0]['segments'][1]['audio_ready'] and seen[0]['segments'][1]['audio_processing']
 assert seen[1]['segments'][1]['audio_ready'] and not seen[1]['segments'][1]['audio_processing']
 final,_=load(id,tmp_path);assert '_generation_status' not in final


def test_an_unpublished_partial_clip_is_not_exposed_as_ready(tmp_path):
 import studio_workflow as workflow
 id,store,doc,ctx,x=fixture(tmp_path)
 doc['_generation_status']={'1':'pending'}
 sf.write(clip_file(id,1,tmp_path),x[:100],SR)
 row=workflow.public(doc,ctx,id,tmp_path)['segments'][1]
 assert row['audio_processing'] and not row['audio_ready'] and row['audio_key'] is None

def test_nicknames_persist_without_invalidating_tts_and_stale_revision_is_rejected(tmp_path):
 id,store,doc,ctx,x=fixture(tmp_path);before=row_key(doc,0);payload=body(doc);payload['speaker_names']={'0':'아냐','1':'로이드'}
 result,_=save(id,payload,tmp_path);assert result['revision']==2 and row_key(result,0)==before
 assert load(id,tmp_path)[0]['speaker_names']['0']=='아냐'
 with pytest.raises(RuntimeError):save(id,payload,tmp_path)
 payload=body(result);payload['segments'][0]['speaker_id']='1';result,_=save(id,payload,tmp_path)
 assert result['transcript']['segments'][0]['_speaker_locked'] is True and row_key(result,0)!=before


def test_ungenerated_dialogue_can_save_position_and_speaker_without_audio(tmp_path):
 id,store,doc,ctx,x=fixture(tmp_path)
 assert not clip_file(id,1,tmp_path).exists()
 original=(doc['transcript']['segments'][1]['start'],doc['transcript']['segments'][1]['end'])
 payload=body(doc);payload['segments'][1].update(audio_edit={'position':.5},speaker_id='0')
 result,_=save(id,payload,tmp_path);row=result['transcript']['segments'][1]
 assert row['audio_edit']=={'position':.5} and row['speaker_id']=='0'
 assert (row['start'],row['end'])==original
 for edit in [{'position':-1},{'position':float('nan')},{'position':True},{'trim_start':0,'trim_end':1}]:
  payload=body(result);payload['segments'][1]['audio_edit']=edit
  with pytest.raises(ValueError):save(id,payload,tmp_path)


def test_ungenerated_range_is_saved_without_changing_source_reference_and_resolves_after_tts(tmp_path):
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 payload=body(doc);payload['segments'][1]['audio_edit']={'position':.2,'target_duration':2.}
 doc,_=save(id,payload,tmp_path)
 assert doc['transcript']['segments'][1]['start']==1.3
 assert doc['transcript']['segments'][1]['audio_edit']=={'position':.2,'target_duration':2.}
 sf.write(clip_file(id,1,tmp_path),x[:SR],SR,subtype='FLOAT')
 path=edited_audio(id,1,doc,ctx,tmp_path)
 assert sf.info(path).frames==2*SR
 for edit in [{'position':3.,'target_duration':1.},{'position':0,'target_duration':0},{'position':0,'target_duration':True}]:
  clip_file(id,1,tmp_path).unlink(missing_ok=True)
  payload=body(doc);payload['segments'][1]['audio_edit']=edit
  with pytest.raises(ValueError):save(id,payload,tmp_path)


def test_generation_reference_uses_ungenerated_timeline_range_but_existing_audio_keeps_its_reference(tmp_path):
 from studio_workflow import materialize_ungenerated_ranges,editor_segments
 id,store,doc,ctx,x=fixture(tmp_path)
 # Existing audio at index 0 must retain its original source reference.
 doc['transcript']['segments'][0]['audio_edit']={'position':.5,'target_duration':2.}
 doc['transcript']['segments'][1]['audio_edit']={'position':.2,'target_duration':2.5}
 doc['transcript']['reference_windows']=[{'start':0.,'end':1.,'start_frame':0,'end_frame':SR,'fade_in_frames':0,'fade_out_frames':0},
  {'start':1.3,'end':2.3,'start_frame':round(1.3*SR),'end_frame':round(2.3*SR)}]
 assert materialize_ungenerated_ranges(id,doc,ctx,[0,1],tmp_path)
 row=doc['transcript']['segments'][1]
 assert (row['start'],row['end'])==(.2,2.7)
 assert row['_original_reference_range']=={'start':1.3,'end':2.3}
 prepared=editor_segments(doc['transcript'],len(x))
 assert (prepared[1]['reference_lo'],prepared[1]['reference_hi'])==(round(.2*SR),round(2.7*SR))
 from render import reference_audio
 reference=tmp_path/'selected-reference.wav';reference_audio(Path(ctx['clean']),prepared[1],reference)
 assert sf.info(reference).frames==round(2.5*SR)
 assert doc['transcript']['segments'][0]['start']==0
 assert doc['transcript']['reference_windows'][0]['end']==1.

def test_pause_edits_preserve_speech_and_work_in_reverse_selection_order(tmp_path):
 id,store,doc,ctx,x=fixture(tmp_path);info=clip_info(id,0,tmp_path);assert len(info['pauses'])==2
 original=len(x);deltas=[];changes=[]
 for p in reversed(info['pauses']):
  amount=p['end']-p['start']+.2;changes.append({'id':p['id'],'seconds':amount});deltas.append(.2)
 edit={'trim_start':0,'trim_end':info['duration'],'speed':1,'position':0,'pauses':changes,'audio_sha256':info['sha256']}
 doc['transcript']['segments'][0]['audio_edit']=edit;path=edited_audio(id,0,doc,ctx,tmp_path);out,sr=sf.read(path,dtype='float32',always_2d=True)
 assert sr==SR and len(out)==original+round(.4*SR)
 assert np.array_equal(out[:SR],x[:SR]) and np.array_equal(out[-SR:],x[-SR:])
 with pytest.raises(ValueError):validate_edit({**edit,'speed':.9},info,ctx['duration'])
 with pytest.raises(ValueError):validate_edit({**edit,'pauses':[{'id':999,'seconds':1}]},info,ctx['duration'])
 with pytest.raises(ValueError):validate_edit({**edit,'audio_sha256':'old'},info,ctx['duration'])

def test_manual_speed_uses_google_speedy_and_does_not_regenerate_tts(tmp_path):
 id,store,doc,ctx,x=fixture(tmp_path);info=clip_info(id,0,tmp_path);doc['transcript']['segments'][0]['audio_edit']={'speed':1.5,'audio_sha256':info['sha256']}
 def stretch(source,dest,frames):
  data,_=sf.read(source,always_2d=True);sf.write(dest,data[:frames],SR,subtype='FLOAT');return data[:frames]
 with patch('speedy.stretch',side_effect=stretch) as speedy,patch('render.request_audio',side_effect=AssertionError('No TTS for waveform edits')):
  path=edited_audio(id,0,doc,ctx,tmp_path);assert abs(sf.info(path).duration-info['duration']/1.5)<.001;speedy.assert_called_once()

def test_editor_api_validates_revision_before_queuing_generation(tmp_path):
 id,store,doc,ctx,x=fixture(tmp_path);queue=VideoQueue(store)
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start') as start,TestClient(api.app) as client:
  snap=client.get('/api/studio/'+id);assert snap.status_code==200 and len(snap.json()['document']['segments'])==2
  assert 'dubbing_review' not in snap.json()['document']
  assert all('needs_review' not in row for row in snap.json()['document']['segments'])
  p='/api/studio/'+id+'/segments/0/generate'
  assert client.post(p,json={'revision':1}).status_code==400
  assert client.post(p,json={'confirmed':True,'revision':0}).status_code==409;start.assert_not_called()
  assert client.post(p,json={'confirmed':True,'revision':1}).status_code==200
  job=store.get(id);assert job['workflow_action']=='tts' and job['editor_index']==0 and job['state']=='queued'
  assert client.put('/api/studio/'+id+'/draft',json=body(doc)).status_code==200
  start.assert_called_once_with(editor_only=True,automatic=True)
  assert client.post('/api/studio/history',json={'ids':[id,str(uuid.uuid4())]}).json()['videos'][0]['id']==id

def test_editor_queue_does_not_claim_legacy_automatic_tts_jobs(tmp_path):
 id,store,doc,ctx,x=fixture(tmp_path);store.add('old','old.mp4');store.schedule_editor(id,'analyze');job=store.claim(editor_only=True);assert job['id']==id and store.get('old')['state']=='queued'

def test_moved_clip_preserves_reference_removal_and_uses_output_position(tmp_path):
 from render import speech_placements,replacement_intervals
 segments=[{'lo':SR,'hi':2*SR,'output_start_frame':round(.5*SR)}];sf.write(tmp_path/'0000-fitted.wav',np.zeros((SR,2),np.float32),SR)
 placement=speech_placements(segments,tmp_path);assert placement[0]['lo']==round(.5*SR)
 intervals=replacement_intervals(segments,3*SR,placement);assert intervals[0]['lo']==round(.5*SR) and intervals[0]['hi']>=2*SR


def test_windowed_preview_and_zoom_waveform_are_bounded_read_only_and_preserve_samples(tmp_path):
 import io
 id,store,doc,ctx,x=fixture(tmp_path);queue=VideoQueue(store)
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start') as start,TestClient(api.app) as client:
  base='/api/studio/'+id
  response=client.get(base+'/preview-source?kind=original&start=0.5&end=1.5');assert response.status_code==200
  values,rate=sf.read(io.BytesIO(response.content),dtype='float32',always_2d=True)
  assert rate==SR and np.array_equal(values,x[SR//2:SR+SR//2])
  assert client.get(base+'/preview-source?kind=secret&start=0&end=1').status_code==400
  assert client.get(base+'/preview-source?kind=clean&start=-1&end=1').status_code==400
  assert client.get(base+'/preview-source?kind=clean&start=0&end=100').status_code==400
  wave=client.get(base+'/waveform?start=0.5&end=1.5&points=1000').json()
  assert wave['start']==.5 and wave['end']==1.5 and 900<=len(wave['peaks'])<=1000
  assert client.get(base+'/waveform?start=0&end=1&points=100000000').status_code==400
  info=client.get(base+'/segments/0/waveform').json();assert all('cut' in p for p in info['pauses']) and info['reference_rms']>0
  assert load(id,tmp_path)[0]['revision']==1;start.assert_not_called()


def test_edge_extension_padding_preserves_original_audio_and_never_generates_tts(tmp_path):
 id,store,doc,ctx,x=fixture(tmp_path);info=clip_info(id,0,tmp_path)
 doc['transcript']['segments'][0]['audio_edit']={'pad_start':.2,'pad_end':.3,'audio_sha256':info['sha256']}
 with patch('render.request_audio',side_effect=AssertionError('No TTS for edge resize')):
  path=edited_audio(id,0,doc,ctx,tmp_path);out,rate=sf.read(path,dtype='float32',always_2d=True)
 assert rate==SR and len(out)==len(x)+round(.5*SR)
 assert np.array_equal(out[round(.2*SR):round(.2*SR)+len(x)],x)
 assert not np.any(out[:round(.2*SR)]) and not np.any(out[-round(.3*SR):])


def test_added_and_deleted_dialogue_preserve_stable_cache_ids_and_can_be_restored(tmp_path):
 from studio_workflow import editor_segments,digest
 id,store,doc,ctx,x=fixture(tmp_path);cache=clip_file(id,0,tmp_path);sha=digest(cache);before=row_key(doc,0)
 payload=body(doc);payload['segments'].append({'id':2,'start':.4,'end':.9,'original_text':'はい。','translated_text':'네.','speaker_id':'0','keep_original':False,'audio_edit':{},'deleted':False})
 result,_=save(id,payload,tmp_path);assert len(result['transcript']['segments'])==3 and row_key(result,0)==before and digest(cache)==sha
 prepared=editor_segments(result['transcript'],len(x));assert len(prepared)==3 and prepared[2]['lo']==round(.4*SR)
 payload=body(result);payload['segments'][0]['deleted']=True;deleted,_=save(id,payload,tmp_path);assert deleted['transcript']['segments'][0]['deleted'] and digest(cache)==sha
 payload=body(deleted);payload['segments'][0]['deleted']=False;restored,_=save(id,payload,tmp_path);assert not restored['transcript']['segments'][0]['deleted'] and row_key(restored,0)==before and digest(cache)==sha
 invalid=body(restored);invalid['segments'][0]['start']=.5
 with pytest.raises(ValueError):save(id,invalid,tmp_path)


def test_editor_mix_reuses_stable_caches_after_manual_insertion_without_paid_requests(tmp_path,monkeypatch):
 import studio_workflow as workflow
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 payload=body(doc);payload['segments'].append({'id':2,'start':.4,'end':.9,'original_text':'はい。','translated_text':'네.','speaker_id':'0','audio_edit':{},'keep_original':False})
 doc,_=save(id,payload,tmp_path)
 for index,seconds in [(1,.5),(2,.2)]:sf.write(clip_file(id,index,tmp_path),x[:round(seconds*SR)],SR,subtype='FLOAT')
 before={i:workflow.digest(clip_file(id,i,tmp_path)) for i in range(3)}
 original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'ROOT',tmp_path);monkeypatch.setattr(workflow,'Store',lambda:store);monkeypatch.setattr(workflow,'config',lambda:{'language':'ko'})
 with patch('render.generate_speech',side_effect=AssertionError('Mix must not generate TTS')),patch('reference_bank_guard.prepare',side_effect=AssertionError('Mix must not infer speakers')):
  assert workflow.generate(id,'mix')==0
 folder=original_paths(id,tmp_path)[0];assert (folder/'preview.mp3').stat().st_size>1000
 assert sf.info(folder/'mixed.wav').frames==len(x)
 assert {i:workflow.digest(clip_file(id,i,tmp_path)) for i in range(3)}==before
 assert store.get(id)['state']=='ready'


def test_audio_versions_restore_undo_redo_without_generation(tmp_path):
 from studio_workflow import archive_generation,audio_versions,restore_generation,digest,public
 id,store,doc,ctx,x=fixture(tmp_path);base=clip_file(id,0,tmp_path);old=digest(base)
 archive_generation(id,0,doc,tmp_path)
 sf.write(base,x[:SR],SR,subtype='FLOAT');new=digest(base)
 assert audio_versions(id,0,tmp_path)[0]['audio_key']==old
 with patch('render.request_audio',side_effect=AssertionError('Restore must never generate TTS')):
  restored,_=restore_generation(id,0,old,tmp_path);assert digest(base)==old
  assert restored['generated']['0']['input_hash']==row_key(restored,0)
  restored,_=restore_generation(id,0,new,tmp_path);assert digest(base)==new
  hidden,_=restore_generation(id,0,None,tmp_path);assert hidden['generated']['0']['hidden']
  restored,_=restore_generation(id,0,new,tmp_path);assert not restored['generated']['0'].get('hidden')
 assert {v['audio_key'] for v in audio_versions(id,0,tmp_path)}=={old,new}
 with pytest.raises(ValueError):restore_generation(id,0,'../secret',tmp_path)


def test_verified_speaker_corrections_reach_editor_and_preserve_user_selection(tmp_path):
 from studio_workflow import apply_verified_speakers
 id,store,doc,ctx,x=fixture(tmp_path);segments=copy.deepcopy(doc['transcript']['segments'])
 audit={'changes':[{'index':1,'original_speaker':'1','bank_speaker':'0','action':'reassign_reference_bank'}]}
 assert apply_verified_speakers(doc,segments,audit)==[1]
 assert doc['transcript']['segments'][1]['speaker_id']==segments[1]['speaker_id']=='0'
 assert doc['transcript']['segments'][1]['_asr_speaker_id']=='1'
 assert apply_verified_speakers(doc,segments,audit)==[]
 doc['transcript']['segments'][1].update(speaker_id='1',_speaker_locked=True)
 assert apply_verified_speakers(doc,segments,audit)==[]


def test_empty_speaker_tracks_persist_without_touching_tts_inputs_and_can_be_undone(tmp_path):
 from studio_workflow import public
 id,store,doc,ctx,x=fixture(tmp_path);key=row_key(doc,0)
 payload=body(doc);payload['speaker_ids']=['0','1','7']
 result,_=save(id,payload,tmp_path)
 assert public(result,ctx,id,tmp_path)['speaker_ids']==['0','1','7']
 assert row_key(result,0)==key
 payload=body(result);payload['speaker_ids']=['0','1'];result,_=save(id,payload,tmp_path)
 assert load(id,tmp_path)[0]['speaker_ids']==['0','1']
 for ids in [['0','0'],[''],[1],'wrong']:
  payload=body(result);payload['speaker_ids']=ids
  with pytest.raises(ValueError):save(id,payload,tmp_path)


def test_audio_restore_keeps_generated_translation_paired_through_undo_and_redo(tmp_path):
 from studio_workflow import archive_generation,restore_generation,digest,public,generation_input
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path);base=clip_file(id,0,tmp_path);old_key=digest(base)
 old_input=generation_input(doc,0);doc['generated']['0']['input']=old_input
 atomic_json(paths(id,tmp_path)[1],doc)
 payload=body(doc);payload['segments'][0]['translated_text']='Changed translation'
 doc,_=save(id,payload,tmp_path);archive=archive_generation(id,0,doc,tmp_path)
 assert json.loads((archive/'editor-state.json').read_text('utf8'))['generation']['input']==old_input
 sf.write(base,x[:SR],SR,subtype='FLOAT');new_key=digest(base)
 new_input={**old_input,'translated_text':'Changed translation'}
 doc['generated']['0']={'input_hash':row_key(doc,0),'sha256':new_key,'input':new_input}
 atomic_json(paths(id,tmp_path)[1],doc)
 # The history applies the saved script before switching the corresponding audio.
 payload=body(doc);payload['segments'][0]['translated_text']=old_input['translated_text'];doc,_=save(id,payload,tmp_path)
 doc,_=restore_generation(id,0,old_key,tmp_path)
 row=public(doc,ctx,id,tmp_path)['segments'][0]
 assert digest(base)==old_key and row['translated_text']==row['audio_input']['translated_text']==old_input['translated_text']
 assert not row['audio_stale']
 payload=body(doc);payload['segments'][0]['translated_text']=new_input['translated_text'];doc,_=save(id,payload,tmp_path)
 doc,_=restore_generation(id,0,new_key,tmp_path);row=public(doc,ctx,id,tmp_path)['segments'][0]
 assert digest(base)==new_key and row['translated_text']==row['audio_input']['translated_text']==new_input['translated_text']
 assert not row['audio_stale']

def test_studio_has_40_local_fit_slots_separate_from_20_tts_slots(tmp_path,monkeypatch):
 import threading
 import studio_workflow as workflow
 import render
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 rows=[{**copy.deepcopy(doc['transcript']['segments'][0]),'start':i*.05,'end':i*.05+.04} for i in range(45)]
 doc['transcript']['segments']=rows;doc['generated']={}
 atomic_json(paths(id,tmp_path)[1],doc)
 original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store)
 monkeypatch.setattr(workflow,'config',lambda:{'language':'ko'})
 monkeypatch.setattr('reference_bank_guard.prepare',lambda *a,**k:{})
 monkeypatch.setattr('shared_reference.prepare',lambda *a,**k:None)
 lock=threading.Lock();tts_barrier=threading.Barrier(20);fit_barrier=threading.Barrier(40)
 active={'tts':0,'fit':0};peak={'tts':0,'fit':0}
 def enter(kind):
  with lock:active[kind]+=1;peak[kind]=max(peak[kind],active[kind])
 def leave(kind):
  with lock:active[kind]-=1
 def generate(i,row,*args):
  enter('tts')
  try:
   if i<20:tts_barrier.wait(10)
   return {'index':i}
  finally:leave('tts')
 def fit(prepared,cfg):
  i=prepared['index'];enter('fit')
  try:
   if i<40:fit_barrier.wait(10)
   if i==44:raise RuntimeError('simulated fitting failure')
   sf.write(clip_file(id,i,tmp_path),x[:100],SR,subtype='FLOAT')
   return {'index':i}
  finally:leave('fit')
 monkeypatch.setattr(render,'generate_speech',generate);monkeypatch.setattr(render,'fit_speech',fit)
 assert workflow.generate(id,'tts')==0
 assert peak=={'tts':20,'fit':40}
 saved,_=load(id,tmp_path)
 assert len(saved['generated'])==45
 assert saved['generated']['44']['error']=='simulated fitting failure'
 assert saved['generated']['0']['input']['translated_text']==rows[0]['translated_text']
 assert store.get(id)['state']=='ready'

def test_cached_studio_runs_new_acoustic_policy_and_publishes_new_speaker(tmp_path,monkeypatch):
 import studio_workflow as workflow
 import render
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 sf.write(clip_file(id,1,tmp_path),x[:100],SR,subtype='FLOAT')
 doc['generated']['1']={'input_hash':row_key(doc,1),'seconds':100/SR}
 atomic_json(paths(id,tmp_path)[1],doc)
 original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store);monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','speaker_embeddings':True})
 audit={'changes':[{'index':1,'original_speaker':'1','bank_speaker':'2','action':'acoustic_recluster'}]}
 calls=[]
 monkeypatch.setattr('reference_bank_guard.prepare',lambda *a,**k:calls.append('guard') or audit)
 monkeypatch.setattr('shared_reference.prepare',lambda *a,**k:None)
 monkeypatch.setattr(render,'generate_speech',lambda i,row,*a:{'index':i,'speaker':row['speaker_id']})
 def fit(p,cfg):
  sf.write(clip_file(id,p['index'],tmp_path),x[:100],SR,subtype='FLOAT');return {'index':p['index']}
 monkeypatch.setattr(render,'fit_speech',fit)
 assert workflow.generate(id,'tts')==0 and calls==['guard']
 saved,_=load(id,tmp_path)
 assert saved['transcript']['segments'][1]['speaker_id']=='2'
 assert saved['generated']['1']['input']['speaker_id']=='2'

def test_embeddings_disabled_keep_qwen_ids_and_ignore_old_acoustic_donors(tmp_path,monkeypatch):
 import studio_workflow as workflow
 import render
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 doc['transcript']['preserve_source_speaker_ids']=True
 doc['transcript']['segments'][1].update(_reference_bank_speaker='99',_reference_donor_indices=[0])
 atomic_json(paths(id,tmp_path)[1],doc)
 original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store)
 monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','speaker_embeddings':False})
 monkeypatch.setattr('reference_bank_guard.prepare',lambda *a,**k:pytest.fail('Embeddings must stay off'))
 def shared(segments,*a,**k):
  assert segments[1]['speaker_id']=='1'
  assert '_reference_bank_speaker' not in segments[1] and '_reference_donor_indices' not in segments[1]
 monkeypatch.setattr('shared_reference.prepare',shared)
 monkeypatch.setattr(render,'generate_speech',lambda i,row,*a:{'index':i})
 def fit(p,cfg):
  sf.write(clip_file(id,p['index'],tmp_path),x[:100],SR,subtype='FLOAT');return {'index':p['index']}
 monkeypatch.setattr(render,'fit_speech',fit)
 assert workflow.generate(id,'tts',1)==0
 saved,_=load(id,tmp_path);assert saved['transcript']['segments'][1]['speaker_id']=='1'

@pytest.mark.parametrize('batch_mode',[False,'context','tagged_context'])
def test_individual_generation_reference_follows_generation_mode(tmp_path,monkeypatch,batch_mode):
 import studio_workflow as workflow
 import render
 id,store,doc,ctx,x=fixture(tmp_path)
 original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store)
 monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','speaker_embeddings':False,'tts_reference_mode':'same_speaker_tagged_context' if batch_mode=='tagged_context' else 'same_speaker_context' if batch_mode=='context' else 'individual','tts_mode':'same_speaker_batches' if batch_mode is True else 'individual'})
 monkeypatch.setattr('shared_reference.prepare',lambda *a,**k:pytest.fail('Individual reference must not use donor clips'))
 calls=[]
 def prepare(i,row,*a):
  assert ('_shared_reference_path' in row)==bool(batch_mode)
  if batch_mode:
   assert row['_shared_reference_indices']==[1]
   assert sf.info(row['_shared_reference_path']).duration==pytest.approx(1.)
  calls.append(i);return {'index':i}
 monkeypatch.setattr(render,'generate_speech',prepare)
 def fit(p,cfg):
  sf.write(clip_file(id,p['index'],tmp_path),x[:100],SR,subtype='FLOAT');return {'index':p['index']}
 monkeypatch.setattr(render,'fit_speech',fit)
 assert workflow.generate(id,'tts',1)==0 and calls==[1]


def test_studio_batch_generation_preserves_clean_clips_and_maps_subset_back_to_ids(tmp_path,monkeypatch):
 import studio_workflow as workflow
 id,store,doc,ctx,x=fixture(tmp_path)
 from core import atomic_json
 doc['transcript']['segments'][1]['audio_edit']={'position':.2,'target_duration':.5}
 atomic_json(paths(id,tmp_path)[1],doc)
 before=workflow.digest(clip_file(id,0,tmp_path));original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store)
 monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','tts_mode':'same_speaker_batches','tts_reference_mode':'individual'})
 calls=[]
 def batch(rows,clean,folder,cfg,progress,status=None):
  assert len(rows)==1 and rows[0]['translated_text']=='아버지.'
  assert rows[0]['next_dialogue_start_frame']==len(x)
  folder.mkdir(parents=True,exist_ok=True);sf.write(folder/'0000-fitted.wav',x[:100],SR,subtype='FLOAT')
  calls.append('batch');return [{'index':0,'tts_generation_mode':'same_speaker_batches'}]
 monkeypatch.setattr('batch_tts.synthesize_speaker_batches',batch)
 assert workflow.generate(id,'tts')==0 and calls==['batch']
 saved,_=load(id,tmp_path)
 assert workflow.digest(clip_file(id,0,tmp_path))==before
 assert saved['generated']['1']['input']['translated_text']=='아버지.'
 assert saved['transcript']['segments'][1]['audio_edit']['position']==.2
 assert 'target_duration' not in saved['transcript']['segments'][1]['audio_edit']
 assert sf.info(edited_audio(id,1,saved,ctx,tmp_path)).frames==round(.5*SR)


@pytest.mark.parametrize('single',[False,True])
def test_batch_regeneration_uses_clean_same_voice_context_but_replaces_only_requested_clips(tmp_path,monkeypatch,single):
 import studio_workflow as workflow
 import render
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 doc['transcript']['segments'][0]['speaker_id']='1'
 doc['generated']['0']['input_hash']=row_key(doc,0)
 atomic_json(paths(id,tmp_path)[1],doc)
 before=workflow.digest(clip_file(id,0,tmp_path));original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store)
 monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','tts_mode':'same_speaker_batches','tts_reference_mode':'individual'})
 monkeypatch.setattr(render,'generate_speech',lambda *a:pytest.fail('Must generate the full batch'))
 calls=[]
 def batch(rows,clean,folder,cfg,progress,status=None):
  assert [r['translated_text'] for r in rows]==['네!','아버지.']
  assert bool(cfg.get('batch_regeneration_nonce'))==single
  folder.mkdir(parents=True,exist_ok=True)
  for local in range(2):sf.write(folder/f'{local:04d}-fitted.wav',x[:100+local],SR,subtype='FLOAT')
  calls.append('batch');return [{'index':i} for i in range(2)]
 monkeypatch.setattr('batch_tts.synthesize_speaker_batches',batch)
 assert workflow.generate(id,'tts',1 if single else None)==0 and calls==['batch']
 saved,_=load(id,tmp_path)
 assert workflow.digest(clip_file(id,0,tmp_path))==before
 assert sf.info(clip_file(id,1,tmp_path)).frames==101
 assert saved['generated']['1']['input']['translated_text']=='아버지.'


def test_old_processing_policy_refits_unchanged_text_instead_of_reusing_cut_audio(tmp_path,monkeypatch):
 import studio_workflow as workflow
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 doc['generated']['0']['processing_policy']={'fit':1}
 atomic_json(paths(id,tmp_path)[1],doc)
 before=workflow.digest(clip_file(id,0,tmp_path));original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store)
 monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','tts_mode':'same_speaker_batches','tts_reference_mode':'individual'})
 def batch(rows,clean,folder,cfg,progress,status=None):
  assert len(rows)==2
  folder.mkdir(parents=True,exist_ok=True)
  for i in range(2):sf.write(folder/f'{i:04d}-fitted.wav',x[:100+i],SR,subtype='FLOAT')
  return [{'index':i} for i in range(2)]
 monkeypatch.setattr('batch_tts.synthesize_speaker_batches',batch)
 assert workflow.generate(id,'tts')==0
 saved,_=load(id,tmp_path)
 assert workflow.digest(clip_file(id,0,tmp_path))!=before
 assert saved['generated']['0']['processing_policy']==workflow.audio_processing_policy()
 assert saved['generated']['0']['input_hash']==row_key(saved,0)
 assert saved['revision']>doc['revision']


def test_cached_last_utterance_is_refitted_to_video_end_with_reference_preparation(tmp_path,monkeypatch):
 import studio_workflow as workflow
 import render
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 row=doc['transcript']['segments'][1];row.update(start=3.2,end=3.5)
 sf.write(clip_file(id,1,tmp_path),x[:round(.32*SR)],SR,subtype='FLOAT')
 doc['generated']['1']={'input_hash':row_key(doc,1),'seconds':.32}
 atomic_json(paths(id,tmp_path)[1],doc)
 original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store);monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','speaker_embeddings':False,'tts_reference_mode':'same_speaker_shared_donor_tag'})
 def shared(segments,*a,**k):
  for s in segments:s['_test_reference_ready']=True
 monkeypatch.setattr('shared_reference.prepare',shared)
 calls=[]
 def prepare(i,row,*a):
  assert row['_test_reference_ready'] and row['next_dialogue_start_frame']==len(x)
  calls.append(i);return {'index':i,'frames':len(x)-row['lo']}
 monkeypatch.setattr(render,'generate_speech',prepare)
 def fit(p,cfg):
  sf.write(clip_file(id,p['index'],tmp_path),x[:p['frames']],SR,subtype='FLOAT');return {'index':p['index']}
 monkeypatch.setattr(render,'fit_speech',fit)
 assert workflow.generate(id,'tts')==0 and calls==[1]
 assert sf.info(clip_file(id,1,tmp_path)).frames+round(3.2*SR)==len(x)

def test_bulk_tts_only_regenerates_stale_audio_and_api_schedules_one_job(tmp_path,monkeypatch):
 import studio_workflow as workflow
 import render
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 sf.write(clip_file(id,1,tmp_path),x[:100],SR,subtype='FLOAT')
 doc['generated']['1']={'input_hash':row_key(doc,1),'seconds':100/SR}
 doc['transcript']['segments'][1]['translated_text']='바뀐 대사!'
 atomic_json(paths(id,tmp_path)[1],doc)
 queue=VideoQueue(store)
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start') as start,TestClient(api.app) as client:
  assert client.post('/api/studio/'+id+'/generate-stale',json={'confirmed':True,'revision':0}).status_code==409
  start.assert_not_called()
  assert client.post('/api/studio/'+id+'/generate-stale',json={'confirmed':True,'revision':doc['revision']}).status_code==200
  assert store.get(id)['workflow_action']=='tts' and store.get(id)['editor_index'] is None
  start.assert_called_once()
 original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store);monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','speaker_embeddings':False})
 monkeypatch.setattr('shared_reference.prepare',lambda *a,**k:None)
 calls=[];before=workflow.digest(clip_file(id,0,tmp_path))
 monkeypatch.setattr(render,'generate_speech',lambda i,*a:calls.append(i) or {'index':i})
 def fit(p,cfg):
  sf.write(clip_file(id,p['index'],tmp_path),x[:200],SR,subtype='FLOAT');return {'index':p['index']}
 monkeypatch.setattr(render,'fit_speech',fit)
 assert workflow.generate(id,'tts')==0 and calls==[1]
 assert workflow.digest(clip_file(id,0,tmp_path))==before
 assert workflow.audio_versions(id,1,tmp_path)


def test_legacy_batch_regeneration_sends_full_untagged_batch_with_instructions(tmp_path,monkeypatch):
 import io,base64
 import studio_workflow as workflow
 import render
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 doc['transcript']['segments'][0]['speaker_id']='1'
 atomic_json(paths(id,tmp_path)[1],doc)
 before=workflow.digest(clip_file(id,0,tmp_path));original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store)
 monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','key':'test','tts_mode':'same_speaker_batches','tts_reference_mode':'same_speaker_batch'})
 payloads=[]
 class Reply(io.BytesIO):headers={}
 def reply(request,**kwargs):
  payload=json.loads(request.data);payloads.append(payload)
  assert payload['instructions']==render.tts_instructions('ko')
  assert payload['input'].splitlines()==['네!','아버지.']
  assert payload['input_references'][1]['text'].splitlines()==['はい！','父上。']
  audio,rate=sf.read(io.BytesIO(base64.b64decode(payload['input_references'][0]['input_audio']['data'].split(',',1)[1])))
  assert rate==SR and len(audio)==2*SR
  assert payload['provider']['options']['fish-audio']['temperature']==1.
  assert payload['provider']['options']['fish-audio']['top_p']==1.
  return Reply(b'audio')
 monkeypatch.setattr(render.urllib.request,'urlopen',reply)
 monkeypatch.setattr(render,'probe',lambda *a:{'format':{'duration':'2'}})
 def split(audio,segments,indices,folder,cfg):
  assert len(indices)==2
  return x[:2*SR],[(0,SR),(SR,2*SR)],{'status':'test'}
 monkeypatch.setattr('batch_tts.split_generation',split)
 def fit(item,cfg):
  sf.write(item['fitted'],x[:101+item['index']],SR,subtype='FLOAT')
  return {'index':item['index']}
 monkeypatch.setattr(render,'fit_speech',fit)
 monkeypatch.setattr(render,'generate_speech',lambda *a:pytest.fail('Individual generation must not be used'))
 for attempt in range(2):
  assert workflow.generate(id,'tts',1)==0
  assert len(payloads)==attempt+1  # A regenerate click must not reuse the prior batch.
  assert workflow.digest(clip_file(id,0,tmp_path))==before
  assert sf.info(clip_file(id,1,tmp_path)).frames==102


def test_legacy_short_batch_tail_regeneration_keeps_sixty_second_group_ownership(tmp_path,monkeypatch):
 import studio_workflow as workflow
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 doc['transcript']['segments']=[{'start':i*.1,'end':(i+1)*.1,'speaker_id':'A','original_text':f'source {i}','translated_text':f'target {i}','keep_original':False,'audio_edit':{}} for i in range(17)]
 atomic_json(paths(id,tmp_path)[1],doc)
 before=workflow.digest(clip_file(id,0,tmp_path));original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store)
 monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','tts_mode':'same_speaker_batches','tts_reference_mode':'same_speaker_batch'})
 def batch(rows,clean,folder,cfg,progress,status=None):
  assert [r['translated_text'] for r in rows]==[f'target {i}' for i in range(17)]
  assert cfg['_speaker_batch_groups']==[list(range(17))]
  assert cfg['_speaker_batch_requests']==[list(range(17))]
  folder.mkdir(parents=True,exist_ok=True);sf.write(folder/'0016-fitted.wav',x[:100],SR,subtype='FLOAT')
  return [{'index':i,'context_only':i!=16} for i in range(17)]
 monkeypatch.setattr('batch_tts.synthesize_speaker_batches',batch)
 assert workflow.generate(id,'tts',16)==0
 assert workflow.digest(clip_file(id,0,tmp_path))==before
 assert sf.info(clip_file(id,16,tmp_path)).frames==100



def test_reference_listen_cache_changes_with_source_window(tmp_path):
 from studio_workflow import source_clip,paths
 id,store,doc,ctx,x=fixture(tmp_path)
 old=source_clip(id,0,doc,ctx,tmp_path)
 old_values,_=sf.read(old)
 assert source_clip(id,0,doc,ctx,tmp_path)==old
 doc['transcript']['segments'][0].update(start=.3,end=.6)
 windows=doc['transcript'].get('reference_windows')
 if windows:
  windows[0].update(start=.3,end=.6,start_frame=round(.3*SR),end_frame=round(.6*SR))
 new=source_clip(id,0,doc,ctx,tmp_path)
 assert new!=old and old.exists()
 assert sf.info(new).duration==pytest.approx(.3)
 values,_=sf.read(new);assert len(values)!=len(old_values)



def test_watch_uses_export_mix_and_mux_without_generating_or_overwriting_export(tmp_path,monkeypatch):
 import studio_workflow as workflow
 import render
 id,store,doc,ctx,x=fixture(tmp_path)
 sf.write(clip_file(id,0,tmp_path),x[:SR//2],SR,subtype='FLOAT')
 sf.write(clip_file(id,1,tmp_path),x[:SR//2],SR,subtype='FLOAT')
 before={i:workflow.digest(clip_file(id,i,tmp_path)) for i in range(2)}
 published=tmp_path/'outputs'/id/'final.mp4';published.parent.mkdir(parents=True);published.write_bytes(b'unchanged export')
 original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'ROOT',tmp_path);monkeypatch.setattr(workflow,'Store',lambda:store);monkeypatch.setattr(workflow,'config',lambda:{'language':'ko'})
 calls=[]
 def mux(source,mixed,destination,language,duration):
  calls.append((source,mixed,destination,language,duration));destination.write_bytes(b'player video')
 with patch('render.generate_speech',side_effect=AssertionError('Watching cannot generate paid TTS')),patch('reference_bank_guard.prepare',side_effect=AssertionError('Watching cannot change voices')),patch.object(render,'mux',side_effect=mux):
  assert workflow.generate(id,'watch')==0
 saved,_=load(id,tmp_path)
 assert saved['player_revision']==saved['revision'] and saved.get('render_revision') is None
 assert calls[0][1].name=='mixed.wav' and calls[0][2].parent.name=='player'
 assert published.read_bytes()==b'unchanged export'
 assert {i:workflow.digest(clip_file(id,i,tmp_path)) for i in range(2)}==before
 assert store.get(id)['state']=='ready'


def test_watch_api_checks_revision_and_reuses_current_player(tmp_path):
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path);queue=VideoQueue(store)
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start') as start,TestClient(api.app) as client:
  endpoint='/api/studio/'+id+'/watch'
  assert client.post(endpoint,json={'confirmed':True,'revision':0}).status_code==409
  assert client.get('/api/studio/'+id+'/watch-video').status_code==409
  assert client.post(endpoint,json={'confirmed':True,'revision':1}).status_code==200
  assert store.get(id)['workflow_action']=='watch'
  store.update(id,state='ready')
  doc['player_revision']=1;doc['player_volume_version']=__import__('studio_workflow').VOLUME_VERSION;doc['player_mix_version']=__import__('studio_workflow').transition_policy()['version'];atomic_json(paths(id,tmp_path)[1],doc)
  target=paths(id,tmp_path)[0]/'player/video.mp4';target.parent.mkdir();target.write_bytes(b'video')
  assert client.get('/api/studio/'+id+'/watch-video',headers={'Range':'bytes=0-1'}).status_code==206
  response=client.post(endpoint,json={'confirmed':True,'revision':1})
  assert response.json()['ready'] is True and start.call_count==1

def test_waveform_is_available_before_draft_and_switches_after_separation(tmp_path):
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path);queue=VideoQueue(store)
 folder=tmp_path/'work/videos'/id
 paths(id,tmp_path)[1].unlink()
 sf.write(folder/'original.wav',x*.5,SR,subtype='FLOAT')
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),TestClient(api.app) as client:
  base='/api/studio/'+id
  initial=client.get(base).json()
  assert initial['document'] is None and initial['source_waveform']['kind']=='original'
  raw=client.get(base+'/waveform').json()
  assert raw['kind']=='original' and max(raw['peaks'])>0
  atomic_json(folder/'waveform-source.json',{'path':ctx['clean'],'kind':'clean'})
  ready=client.get(base).json()
  assert ready['document'] is None and ready['source_waveform']['kind']=='clean'
  clean=client.get(base+'/waveform').json()
  assert clean['key']!=raw['key'] and max(clean['peaks'])>max(raw['peaks'])*1.9
  assert client.get(base+'/waveform?start=0&end=1&points=1000').status_code==200

def test_retry_reuses_successful_clips_even_after_reference_policy_changes(tmp_path,monkeypatch):
 import studio_workflow as workflow,render
 from core import atomic_json
 id,store,doc,ctx,x=fixture(tmp_path)
 doc['generated']['0']['processing_policy']={}
 atomic_json(paths(id,tmp_path)[1],doc)
 atomic_json(paths(id,tmp_path)[0]/'resume-completed.json',{'preserve_completed':True})
 original_paths=workflow.paths
 monkeypatch.setattr(workflow,'paths',lambda video_id,root=tmp_path:original_paths(video_id,tmp_path))
 monkeypatch.setattr(workflow,'Store',lambda:store)
 monkeypatch.setattr(workflow,'config',lambda:{'language':'ko','tts_reference_mode':'individual','tts_mode':'individual','speaker_embeddings':False})
 generated=[]
 def generate(i,*a,**kw):generated.append(i);return {'index':i}
 def fit(prepared,cfg):
  i=prepared['index'];sf.write(clip_file(id,i,tmp_path),x[:SR//2],SR,subtype='FLOAT');return {'index':i}
 monkeypatch.setattr(render,'generate_speech',generate);monkeypatch.setattr(render,'fit_speech',fit)
 monkeypatch.setattr(render,'pipelined_queue',lambda items,g,f,progress:[f(g(i,row)) for i,row in enumerate(items)])
 assert workflow.generate(id,'tts')==0
 assert generated==[1]
 assert not (paths(id,tmp_path)[0]/'resume-completed.json').exists()
 assert load(id,tmp_path)[0]['generated']['0']['processing_policy']=={}



@pytest.mark.parametrize('failed_same_video',[True,False])
def test_explicit_editor_generation_resumes_only_its_own_failure(tmp_path,failed_same_video):
 id,store,doc,ctx,x=fixture(tmp_path);queue=VideoQueue(store)
 queue.failure_id=id if failed_same_video else 'another-video'
 with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start') as start,TestClient(api.app) as client:
  response=client.post('/api/studio/'+id+'/segments/0/generate',json={'confirmed':True,'revision':1})
  assert response.status_code==200
  start.assert_called_once_with(editor_only=True,automatic=not failed_same_video)
