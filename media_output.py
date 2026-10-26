"""Publish validated media even when a Windows player holds the previous file."""
from pathlib import Path
import json,uuid
from core import atomic_json

def publish(temporary,destination):
    temporary,destination=Path(temporary),Path(destination)
    try:
        temporary.replace(destination)
        actual=destination
    except PermissionError:
        actual=destination.with_name(destination.stem+'-'+uuid.uuid4().hex+destination.suffix)
        temporary.replace(actual)
    atomic_json(destination.with_name('current-output.json'),{'filename':actual.name})
    return actual

def resolve(destination):
    destination=Path(destination);pointer=destination.with_name('current-output.json')
    if pointer.exists():
        data=json.loads(pointer.read_text('utf8'));name=data.get('filename','')
        if not name or Path(name).name!=name or '/' in name or chr(92) in name or not name.endswith('.mp4'):
            raise RuntimeError('Invalid published MP4 filename')
        actual=destination.with_name(name)
        if actual.is_file():return actual
        raise RuntimeError('Published MP4 is missing')
    return destination
