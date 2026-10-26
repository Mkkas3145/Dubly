from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import json
import pytest
import core


def test_concurrent_json_writers_never_share_temporary_paths(tmp_path):
 destination=tmp_path/'draft.json'
 def write(i):core.atomic_json(destination,{'writer':i,'data':'가'*1000})
 with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(write,range(100)))
 assert json.loads(destination.read_text('utf8'))['writer'] in range(100)
 assert not list(tmp_path.glob('*.tmp'))


def test_atomic_json_retries_brief_windows_destination_lock(tmp_path):
 replace=core.os.replace;calls=[]
 def locked(source,target):
  calls.append(source)
  if len(calls)<3:raise PermissionError('sharing violation')
  return replace(source,target)
 with patch.object(core.os,'replace',side_effect=locked):core.atomic_json(tmp_path/'draft.json',{'ok':True})
 assert len(calls)==3 and json.loads((tmp_path/'draft.json').read_text())=={'ok':True}


def test_failed_write_preserves_prior_json_and_cleans_temp(tmp_path):
 destination=tmp_path/'draft.json';core.atomic_json(destination,{'original':True})
 with patch.object(core.os,'replace',side_effect=PermissionError('locked')),patch('time.sleep'),pytest.raises(PermissionError):
  core.atomic_json(destination,{'new':True})
 assert json.loads(destination.read_text())=={'original':True} and not list(tmp_path.glob('*.tmp'))


def test_tts_error_preserves_actual_failure_and_stage():
 import render
 def fit(value):raise PermissionError('locked draft')
 with pytest.raises(RuntimeError,match='파일 접근') as failure:
  render.pipelined_queue([None],lambda i,v:i,fit,lambda *a:None)
 assert isinstance(failure.value.__cause__,PermissionError)
 assert '후처리' in str(failure.value)
