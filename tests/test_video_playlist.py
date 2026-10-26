import json,uuid
from unittest.mock import patch
from fastapi.testclient import TestClient
import app as api
from core import atomic_json
from storage import Store
from queue_worker import VideoQueue
from studio_workflow import checkpoint,paths,clip_file,load
from video_playlist import origin,members


def group_fixture(root):
    store=Store(root);ids=[str(uuid.uuid4()) for _ in range(3)]
    for n,identifier in enumerate(ids,1):
        store.add(identifier,f'Long ({n}).mp4',source_language='ja',target_language='ko')
        folder=root/'work/videos'/identifier;folder.mkdir(parents=True)
        atomic_json(folder/'split-origin.json',{'group_id':ids[0],'original_filename':'Long.mp4','part':n,'parts':3,'start':(n-1)*1200,'end':n*1200})
        atomic_json(folder/'split-group.json',{'ids':[ids[2],ids[0],ids[1],ids[0],'../../outside']})
        if n!=2:
            checkpoint(identifier,{'target_language':'ko','segments':[{'start':0,'end':1,'original_text':'a','translated_text':'b','speaker_id':0}]},{'duration':1200},root)
            clip=clip_file(identifier,0,root);clip.parent.mkdir();clip.write_bytes(b'cached audio')
    return store,ids


def test_provenance_orders_siblings_and_ignores_duplicate_or_unsafe_ids(tmp_path):
    store,ids=group_fixture(tmp_path)
    group,items=members(ids[1],store,tmp_path)
    assert group['title']=='Long.mp4'
    assert [job['id'] for job,_ in items]==ids
    assert origin('../../outside',tmp_path) is None
    # A similar title from a separate upload never joins this group.
    stranger=str(uuid.uuid4());store.add(stranger,'Long (4).mp4')
    assert members(stranger,store,tmp_path)==(None,[])


def test_deleted_first_part_does_not_break_siblings_and_mismatched_group_is_rejected(tmp_path):
    store,ids=group_fixture(tmp_path)
    with store.connection() as db:db.execute('DELETE FROM videos WHERE id=?',(ids[0],))
    assert [job['id'] for job,_ in members(ids[1],store,tmp_path)[1]]==ids[1:]
    path=tmp_path/'work/videos'/ids[2]/'split-origin.json';data=json.loads(path.read_text('utf8'));data['group_id']=str(uuid.uuid4());atomic_json(path,data)
    assert [job['id'] for job,_ in members(ids[1],store,tmp_path)[1]]==[ids[1]]


def test_playlist_api_exposes_ready_and_pending_parts_without_generating(tmp_path):
    store,ids=group_fixture(tmp_path);queue=VideoQueue(store)
    with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start') as start,TestClient(api.app) as client:
        response=client.get(f'/api/studio/{ids[0]}/playlist')
        assert response.status_code==200
        playlist=response.json()['playlist'];assert playlist['parts']==3
        assert [item['id'] for item in playlist['items']]==ids
        assert [item['playable'] for item in playlist['items']]==[True,False,True]
        assert playlist['items'][1]['watch_url']==f'/videos/{ids[1]}/watch'
        history=client.post('/api/studio/history',json={'ids':[ids[2],ids[0]]}).json()['videos']
        assert [v['split']['part'] for v in history]==[3,1]
        assert all(v['split']['group_id']==ids[0] and v['duration']==1200 for v in history)
        assert len(history)==2 # Do not unhide sibling history entries.
        start.assert_not_called()
        doc,_=load(ids[0],tmp_path);doc['generated']['0']={'hidden':True};atomic_json(paths(ids[0],tmp_path)[1],doc)
        assert client.get(f'/api/studio/{ids[0]}/playlist').json()['playlist']['items'][0]['playable'] is False


def test_standalone_and_corrupt_provenance_remain_single_videos(tmp_path):
    store,ids=group_fixture(tmp_path)
    path=tmp_path/'work/videos'/ids[0]/'split-origin.json';path.write_text('not json','utf8')
    assert members(ids[0],store,tmp_path)==(None,[])
    queue=VideoQueue(store)
    with patch.object(api,'ROOT',tmp_path),patch.object(api,'store',store),patch.object(api,'queue',queue),TestClient(api.app) as client:
        assert client.get(f'/api/studio/{ids[0]}/playlist').json()=={'playlist':None}
