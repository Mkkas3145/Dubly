import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from fastapi.testclient import TestClient
import app as api
from storage import Store
from queue_worker import VideoQueue
from project_sharing import SharedProjects


def videos(store,count):
    ids=[str(uuid.uuid4()) for _ in range(count)]
    for id in ids:store.add(id,'video.mp4')
    return ids


def test_code_import_and_bidirectional_additions_reuse_server_ids_without_exposing_other_projects(tmp_path):
    store=Store(tmp_path);a,b,private=videos(store,3);share=SharedProjects(store)
    created=share.create([a]);code=created['code']
    assert created['ids']==[a]
    assert SharedProjects(store).sync(code,[])['ids']==[a]
    assert set(share.sync(code,[b])['ids'])=={a,b}
    assert set(share.sync(code,[])['ids'])=={a,b}
    assert private not in share.sync(code,[])['ids']
    other=share.create([private]);assert other['code']!=code and other['ids']==[private]
    with store.connection() as db:
        assert code not in str([tuple(r) for r in db.execute('SELECT * FROM shared_projects')])
    assert all(store.get(id)['state']=='queued' for id in (a,b,private))


def test_parallel_connections_do_not_lose_new_projects(tmp_path):
    store=Store(tmp_path);ids=videos(store,20);share=SharedProjects(store);code=share.create([])['code']
    with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(lambda id:share.sync(code,[id]),ids))
    assert set(share.sync(code,[])['ids'])==set(ids)
    assert len(share.sync(code,ids*2)['ids'])==20


def test_deleted_or_unknown_projects_are_not_resurrected(tmp_path):
    store=Store(tmp_path);a,b=videos(store,2);share=SharedProjects(store);code=share.create([a,b])['code']
    with store.connection() as db:db.execute('DELETE FROM videos WHERE id=?',(a,))
    assert share.sync(code,[a,str(uuid.uuid4())])['ids']==[b]


def test_sharing_api_checks_codes_and_ids_without_scheduling_work(tmp_path):
    store=Store(tmp_path);a=videos(store,1)[0];queue=VideoQueue(store)
    with patch.object(api,'store',store),patch.object(api,'queue',queue),patch.object(queue,'start') as start,TestClient(api.app) as client:
        assert client.post('/api/project-share/create',json={'ids':['../../x']}).status_code==400
        result=client.post('/api/project-share/create',json={'ids':[a]});assert result.status_code==200
        code=result.json()['code']
        assert client.post('/api/project-share/sync',json={'code':code,'ids':[]}).json()['ids']==[a]
        assert client.post('/api/project-share/sync',json={'code':'DUBLY-'+'x'*32,'ids':[]}).status_code==404
        assert client.post('/api/project-share/sync',json={'code':'../x','ids':[]}).status_code==400
        assert client.post('/api/project-share/sync',json={'code':code,'ids':[a]*201}).status_code==400
        assert client.post('/api/project-share/sync',json={'code':None}).status_code==400
        start.assert_not_called()
