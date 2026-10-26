import copy
from unittest.mock import patch
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
import app as api
import studio_workflow as workflow
from core import SR, atomic_json
from queue_worker import VideoQueue
from studio_concurrency import commit, pending, advance
from test_studio_workflow import fixture, body


def test_worker_commit_preserves_concurrent_text_timing_aliases_and_new_rows(tmp_path):
    video, store, doc, ctx, x = fixture(tmp_path); baseline = copy.deepcopy(doc)
    edited = copy.deepcopy(doc)
    edited['transcript']['segments'][0].update(translated_text='아니요!', audio_edit={'position': .2})
    edited['speaker_names'] = {'0': '아냐'}
    edited['transcript']['segments'].append(dict(edited['transcript']['segments'][1], start=2.5, end=3))
    edited['revision'] += 1; atomic_json(workflow.paths(video, tmp_path)[1], edited)
    doc['generated']['0'] = dict(doc['generated']['0'], sha256='new-audio')
    doc['transcript']['segments'][0]['audio_edit'] = {}
    doc['revision'] += 1
    commit(workflow.paths(video, tmp_path)[0], doc, baseline, 0)
    saved, _ = workflow.load(video, tmp_path)
    assert saved['transcript'] == edited['transcript']
    assert saved['speaker_names'] == edited['speaker_names']
    assert saved['generated']['0']['sha256'] == 'new-audio'
    assert saved['revision'] > edited['revision']


def test_edit_api_works_during_regeneration_and_duplicate_rows_queue_once(tmp_path):
    video, store, doc, ctx, x = fixture(tmp_path); queue = VideoQueue(store)
    store.schedule_editor(video, 'tts', 0); store.update(video, state='running')
    with patch.object(api, 'ROOT', tmp_path), patch.object(api, 'store', store), patch.object(api, 'queue', queue), patch.object(queue, 'start'), TestClient(api.app) as client:
        data = body(doc); data['segments'][1]['translated_text'] = '수정한 대사.'
        edited = client.put(f'/api/studio/{video}/draft', json=data)
        assert edited.status_code == 200
        revision = edited.json()['revision']; payload = {'revision': revision, 'confirmed': True}
        assert client.post(f'/api/studio/{video}/segments/0/generate', json=payload).json()['duplicate']
        assert client.post(f'/api/studio/{video}/segments/1/generate', json=payload).json()['queued']
        assert client.post(f'/api/studio/{video}/segments/1/generate', json=payload).json()['duplicate']
        assert pending(workflow.paths(video, tmp_path)[0]) == [1]
        assert client.get(f'/api/studio/{video}').json()['video']['regenerating_indices'] == [0, 1]
    store.update(video, state='ready'); advance(store, store.get(video))
    assert store.get(video)['editor_index'] == 1 and store.get(video)['state'] == 'queued'
    assert pending(workflow.paths(video, tmp_path)[0]) == []


def test_failure_holds_queued_rows_and_cancel_clears_them(tmp_path):
    from studio_concurrency import enqueue
    video, store, doc, ctx, x = fixture(tmp_path); folder = workflow.paths(video, tmp_path)[0]
    enqueue(folder, 1, 0); store.update(video, state='error')
    advance(store, store.get(video)); assert pending(folder) == [1]
    store.update(video, state='cancelled'); advance(store, store.get(video)); assert pending(folder) == []


def test_real_single_generation_keeps_mid_request_human_edits_and_marks_old_input_stale(tmp_path, monkeypatch):
    import render
    video, store, doc, ctx, x = fixture(tmp_path); original_paths = workflow.paths
    monkeypatch.setattr(workflow, 'paths', lambda video_id, root=tmp_path: original_paths(video_id, tmp_path))
    monkeypatch.setattr(workflow, 'Store', lambda: store)
    monkeypatch.setattr(workflow, 'config', lambda: {'language': 'ko', 'tts_reference_mode': 'individual', 'tts_mode': 'individual', 'speaker_embeddings': False})
    monkeypatch.setattr(render, 'generate_speech', lambda i, *a: {'index': i})
    def fit(item, cfg):
        # An actual save lands after the worker has sent its immutable request.
        latest, _ = workflow.load(video, tmp_path); data = body(latest)
        data['segments'][0].update(translated_text='새로 편집한 대사.', audio_edit={'position': .1})
        data['segments'][1]['translated_text'] = '다른 대사도 편집.'
        workflow.save(video, data, tmp_path)
        sf.write(workflow.clip_file(video, 0, tmp_path), x[:SR // 2], SR, subtype='FLOAT')
        return {'index': 0}
    monkeypatch.setattr(render, 'fit_speech', fit)
    assert workflow.generate(video, 'tts', 0) == 0
    saved, _ = workflow.load(video, tmp_path)
    assert saved['transcript']['segments'][0]['translated_text'] == '새로 편집한 대사.'
    assert saved['transcript']['segments'][0]['audio_edit']['position'] == .1
    assert saved['transcript']['segments'][1]['translated_text'] == '다른 대사도 편집.'
    assert saved['generated']['0']['input']['translated_text'] == '네!'
    assert saved['generated']['0']['input_hash'] != workflow.row_key(saved, 0)
