"""Short cross-process draft locks and lossless single-line generation commits."""
from pathlib import Path
import copy, json, time
from filelock import FileLock
from core import atomic_json


def draft_lock(folder):
    return FileLock(str(Path(folder) / 'draft.lock'), timeout=30)


def merge(base, generated, edited):
    """Three-way merge; a concurrent human edit wins a same-field conflict."""
    if generated == base: return copy.deepcopy(edited)
    if edited == base: return copy.deepcopy(generated)
    if isinstance(base, dict) and isinstance(generated, dict) and isinstance(edited, dict):
        result = {}
        for k in base.keys() | generated.keys() | edited.keys():
            if k not in generated:
                if k in edited and edited[k] != base.get(k): result[k] = copy.deepcopy(edited[k])
            elif k not in edited:
                if k not in base: result[k] = copy.deepcopy(generated[k])
            elif k not in base:
                result[k] = copy.deepcopy(edited[k])
            else: result[k] = merge(base[k], generated[k], edited[k])
        return result
    if isinstance(base, list) and isinstance(generated, list) and isinstance(edited, list):
        # Dialogue IDs are stable append-only indices, including deleted rows.
        result = []
        for i in range(max(len(generated), len(edited))):
            if i >= len(generated): result.append(copy.deepcopy(edited[i]))
            elif i >= len(edited): result.append(copy.deepcopy(generated[i]))
            elif i >= len(base): result.append(copy.deepcopy(edited[i]))
            else: result.append(merge(base[i], generated[i], edited[i]))
        return result
    return copy.deepcopy(edited)


def commit(folder, doc, baseline, index):
    path = Path(folder) / 'draft.json'
    with draft_lock(folder):
        latest = json.loads(path.read_text('utf8'))
        result = merge(baseline, doc, latest)
        # The worker owns the generated artifact, not the current editable words.
        key = str(index)
        if doc.get('generated', {}).get(key) != baseline.get('generated', {}).get(key):
            result.setdefault('generated', {})[key] = copy.deepcopy(doc['generated'][key])
        if '_generation_status' in doc: result['_generation_status'] = copy.deepcopy(doc['_generation_status'])
        else: result.pop('_generation_status', None)
        result['revision'] = max(doc['revision'], latest['revision']) + (latest != baseline)
        result['updated'] = time.time()
        atomic_json(path, result)
    doc.clear(); doc.update(result)
    return copy.deepcopy(result)


def pending(folder):
    try:
        value = json.loads((Path(folder) / 'regeneration-queue.json').read_text('utf8'))
        return list(dict.fromkeys(i for i in value if type(i) is int and i >= 0)) if isinstance(value, list) else []
    except (OSError, ValueError): return []


def enqueue(folder, index, active=None):
    with draft_lock(folder):
        ids = pending(folder)
        if index == active or index in ids: return False
        ids.append(index); atomic_json(Path(folder) / 'regeneration-queue.json', ids)
        return True


def advance(store, job):
    """Called after the old worker has exited; preserve the global queue limit."""
    folder = store.root / 'work/videos' / job['id'] / 'editor'
    if not folder.exists(): return
    with draft_lock(folder):
        ids = pending(folder)
        if job['state'] == 'cancelled':
            if ids: atomic_json(folder / 'regeneration-queue.json', [])
            return
        if job['state'] not in {'ready', 'done'} or not ids: return
        doc = json.loads((folder / 'draft.json').read_text('utf8')); rows = doc['transcript']['segments']
        while ids:
            index = ids.pop(0)
            if index < len(rows) and not rows[index].get('deleted') and not rows[index].get('keep_original'):
                store.schedule_editor(job['id'], 'tts', index)
                break
        atomic_json(folder / 'regeneration-queue.json', ids)
