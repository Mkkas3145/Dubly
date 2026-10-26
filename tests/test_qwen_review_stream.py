import json
from unittest.mock import patch
import pytest
import qwen_performance_turns as q
from qwen_review_stream import ReviewProgress
from qwen_delivery_cues import request


def rows():
 return q.fixed_rows([{'id':i,'speaker':'0','start':i*2.,'end':i*2.+1.,'text':'Hello',
  'words':[{'word':'Hello','start':i*2.+.1,'end':i*2.+.8}]} for i in range(2)])


def value():
 return {'segments':[{'id':i,'parts':[{'first_word_id':i,'last_word_id':i,'speaker':0,'emotion':'sad, then \\"happy\\"'}]} for i in range(2)],'speakers':[{'id':0,'nickname':'인물'}]}


def monitor(events):
 return ReviewProgress(rows(),lambda row,entry:q.apply([row],{'segments':[entry]}),lambda p,d:events.append((p,d)))


@pytest.mark.parametrize('style',['object','array','fence','names_first'])
def test_partial_tokens_are_not_errors_and_only_complete_valid_regions_count(style):
 v=value()
 if style=='array':v=v['segments']
 if style=='names_first':v={'speakers':v['speakers'],'segments':v['segments']}
 text=json.dumps(v,ensure_ascii=False)
 if style=='fence':text='```json\n'+text+'\n```'
 events=[];m=monitor(events)
 for char in text:m.feed(char)
 m.finish()
 assert len(events)==2
 assert '1/2개' in events[0][1] and '2/2개' in events[1][1]
 assert events[0][0]==50 and events[-1][0]==99


@pytest.mark.parametrize('text',[
 'Here is my answer',
 '{"segments":[{"id":0,"parts":[]',
 '{"segments":[{"id":0,"parts":[{"first_word_id":999,"last_word_id":999,"speaker":0}]}',
 '{"segments":[{"id":0,"id":1',
 '{"segments": [}',
])
def test_invalid_format_is_rejected_before_response_finishes(text):
 with pytest.raises(ValueError):monitor([]).feed(text)


def test_duplicate_region_is_rejected_without_counting_it_twice():
 v=value();v['segments'][1]=v['segments'][0]
 events=[];m=monitor(events)
 with pytest.raises(ValueError,match='duplicate_region'):m.feed(json.dumps(v))
 assert len(events)==1


class Reply:
 def __init__(self,parts):self.parts=parts;self.closed=False;self.read=0
 def __enter__(self):return self
 def __exit__(self,*args):self.closed=True
 def __iter__(self):
  for text in self.parts:
   self.read+=1
   yield ('data: '+json.dumps({'id':'request','model':q.QWEN,'choices':[{'index':0,'delta':{'content':text}}]})+'\n').encode()
   yield b'\n'
  yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"output_tokens":42}}\n'
  yield b'\n'
  yield b'data: [DONE]\n'
  yield b'\n'


def test_sse_reports_progress_and_preserves_receipt(tmp_path):
 text=json.dumps(value());reply=Reply([text[:12],text[12:80],text[80:]])
 events=[]
 with patch('urllib.request.urlopen',return_value=reply):
  result,receipt=request({'stream':True},'test',tmp_path,['0','1'],lambda *a:None,
   normalizer=lambda v:q.normalize(v,rows()),content_parser=q.decode,stream_validator=monitor(events))
 assert reply.closed and len(events)==2
 assert len(result['segments'])==2 and receipt['usage']['output_tokens']==42
 assert json.loads((tmp_path/'raw.json').read_text())['choices'][0]['finish_reason']=='stop'
 assert (tmp_path/'stream-content.txt').read_text()==text


def test_bad_sse_format_closes_connection_without_consuming_later_chunks(tmp_path):
 reply=Reply(['not JSON','this must not be consumed'])
 with patch('urllib.request.urlopen',return_value=reply),pytest.raises(ValueError):
  request({'stream':True},'test',tmp_path,['0','1'],lambda *a:None,
   normalizer=lambda v:v,stream_validator=monitor([]))
 assert reply.closed and reply.read==1
 raw=json.loads((tmp_path/'raw.json').read_text())
 assert raw['stream_aborted'] and raw['choices'][0]['message']['content']=='not JSON'


def test_stream_that_ends_without_stop_is_not_accepted(tmp_path):
 class Truncated(Reply):
  def __iter__(self):yield b'data: {"choices":[{"delta":{"content":"{\\"segments\\":"}}]}\n\n'
 with patch('urllib.request.urlopen',return_value=Truncated([])),pytest.raises(RuntimeError,match='response is incomplete'):
  request({'stream':True},'test',tmp_path,['0'],lambda *a:None,stream_validator=monitor([]))

def test_unrecognized_root_shape_is_rejected_as_soon_as_key_finishes():
 with pytest.raises(ValueError,match='unexpected_root_field'):
  monitor([]).feed('{"transcription"')


def test_whitespace_before_json_fence_is_valid():
 m=monitor([])
 for char in '\n  ```json\n'+json.dumps(value())+'\n```':m.feed(char)
 m.finish()
 assert len(m.seen)==2


def test_early_stream_rejection_retries_and_only_caches_complete_valid_response(tmp_path):
 import numpy as np,soundfile as sf
 clean=tmp_path/'clean.wav';sf.write(clean,np.zeros(4*16000),16000)
 audio=tmp_path/'audio.mp3';audio.write_bytes(b'audio')
 replies=[Reply(['Wrong format','not read']),Reply([json.dumps(value())])];progress=[];sent=[]
 def open_url(req,**kwargs):
  sent.append(json.loads(req.data));return replies[len(sent)-1]
 with patch('qwen_audio_pipeline.encode_audio',return_value=(audio,{'sha256':'test'})),patch('urllib.request.urlopen',side_effect=open_url):
  result,report=q.review(clean,rows(),tmp_path,{'key':'test','language':'ko'},lambda p,d:progress.append(d))
 assert report['requests']==2 and len(result)==2
 assert replies[0].closed and replies[0].read==1
 assert all(p['stream'] and p['reasoning']=={'enabled':False} and p['provider']['require_parameters'] for p in sent)
 assert any('다시 시도' in text for text in progress) and any('2/2개' in text for text in progress)
 assert len(list(tmp_path.rglob('result.json')))==1

def test_unfinished_runaway_emotion_string_is_aborted_without_waiting_for_quote():
 m=monitor([])
 with pytest.raises(ValueError,match='string_too_long:emotion'):
  m.feed('{"segments":[{"id":0,"parts":[{"first_word_id":0,"last_word_id":0,"speaker":0,"emotion":"'+'a'*501)
 assert not m.seen

def test_bad_later_region_retries_only_remaining_regions_with_full_audio(tmp_path):
 import numpy as np,soundfile as sf
 clean=tmp_path/'clean.wav';sf.write(clean,np.zeros(4*16000),16000)
 audio=tmp_path/'audio.mp3';audio.write_bytes(b'audio')
 first='{"segments":['+json.dumps(value()['segments'][0])+',{"id":1,"parts":[{"first_word_id":1,"last_word_id":1,"speaker":0,"emotion":"'+'x'*81
 second={'segments':[value()['segments'][1]],'speakers':value()['speakers']}
 replies=[Reply([first,'never consumed']),Reply([json.dumps(second)])];sent=[];events=[]
 def send(req,**kwargs):sent.append(json.loads(req.data));return replies[len(sent)-1]
 with patch('qwen_audio_pipeline.encode_audio',return_value=(audio,{'sha256':'test'})),patch('urllib.request.urlopen',side_effect=send):
  result,report=q.review(clean,rows(),tmp_path,{'key':'test','language':'ko'},lambda p,d:events.append(d))
 assert report['requests']==2 and len(result)==2 and replies[0].read==1
 context=json.loads(sent[1]['messages'][1]['content'][0]['text'])
 assert [r['id'] for r in context['pending_regions']]==[1]
 assert [r['id'] for r in context['accepted_regions']]==[0]
 assert sent[0]['messages'][1]['content'][1]==sent[1]['messages'][1]['content'][1]
 assert all(m['role']!='assistant' for m in sent[1]['messages'])
 assert 'x'*81 not in json.dumps(sent[1])
 assert any('2/2개' in message for message in events)


def test_manual_retry_resumes_validated_regions_from_disk_after_all_attempts_fail(tmp_path):
 import numpy as np,soundfile as sf
 clean=tmp_path/'clean.wav';sf.write(clean,np.zeros(4*16000),16000)
 audio=tmp_path/'audio.mp3';audio.write_bytes(b'audio')
 first='{"segments":['+json.dumps(value()['segments'][0])+',INVALID'
 replies=[Reply([first]),Reply(['bad']),Reply(['bad'])]
 with patch('qwen_audio_pipeline.encode_audio',return_value=(audio,{'sha256':'test'})),patch('urllib.request.urlopen',side_effect=replies),pytest.raises(RuntimeError,match='1/2'):
  q.review(clean,rows(),tmp_path,{'key':'test','language':'ko'})
 partial=json.loads(next(tmp_path.rglob('partial.json')).read_text('utf8'))
 assert [r['id'] for r in partial['segments']]==[0]
 sent=[]
 def send(req,**kwargs):
  sent.append(json.loads(req.data));return Reply([json.dumps({'segments':[value()['segments'][1]],'speakers':value()['speakers']})])
 with patch('qwen_audio_pipeline.encode_audio',return_value=(audio,{'sha256':'test'})),patch('urllib.request.urlopen',side_effect=send):
  result,report=q.review(clean,rows(),tmp_path,{'key':'test','language':'ko'})
 assert len(sent)==1 and report['resumed_regions']==1 and len(result)==2
 assert [r['id'] for r in json.loads(sent[0]['messages'][1]['content'][0]['text'])['pending_regions']]==[1]


def test_word_rewrite_does_not_abort_other_valid_review_regions(tmp_path):
 import numpy as np,soundfile as sf
 clean=tmp_path/'clean.wav';sf.write(clean,np.zeros(4*16000),16000)
 audio=tmp_path/'audio.mp3';audio.write_bytes(b'audio')
 reply=value();reply['segments'][0]['parts'][0]['text']='Invented words!'
 sent=[]
 def send(req,**kwargs):sent.append(json.loads(req.data));return Reply([json.dumps(reply)])
 with patch('qwen_audio_pipeline.encode_audio',return_value=(audio,{'sha256':'test'})),patch('urllib.request.urlopen',side_effect=send):
  result,report=q.review(clean,rows(),tmp_path,{'key':'test','language':'ko'})
 assert len(sent)==1 and len(result)==2
 assert result[0]['text']=='Hello' and result[0]['qwen_speaker_reviewed']
 assert report['changes'][0]['rejected_punctuation'][0]['rejected']=='Invented words!'


def test_retry_recovers_valid_ranges_from_old_punctuation_aborted_stream(tmp_path):
 import numpy as np,soundfile as sf
 from core import atomic_json
 clean=tmp_path/'clean.wav';sf.write(clean,np.zeros(4*16000),16000)
 audio=tmp_path/'audio.mp3';audio.write_bytes(b'audio')
 replies=[Reply(['bad']) for _ in range(3)]
 with patch.object(q,'PROMPT',q.LEGACY_PROMPT),patch('qwen_audio_pipeline.encode_audio',return_value=(audio,{'sha256':'test'})),patch('urllib.request.urlopen',side_effect=replies),pytest.raises(RuntimeError):
  q.review(clean,rows(),tmp_path,{'key':'test','language':'ko'})
 run=next(tmp_path.rglob('input.json')).parent
 old=run/'attempt-legacy';old.mkdir()
 entry=value()['segments'][0];entry['parts'][0]['text']='Invented words'
 atomic_json(old/'raw.json',{'stream_error':'review_punctuation_changed_spoken_text'})
 (old/'stream-content.txt').write_text('['+json.dumps(entry)+',',encoding='utf8')
 sent=[]
 def send(req,**kwargs):
  sent.append(json.loads(req.data));return Reply([json.dumps({'segments':[value()['segments'][1]],'speakers':value()['speakers']})])
 with patch('qwen_audio_pipeline.encode_audio',return_value=(audio,{'sha256':'test'})),patch('urllib.request.urlopen',side_effect=send):
  result,report=q.review(clean,rows(),tmp_path,{'key':'test','language':'ko'})
 assert len(sent)==1 and report['resumed_regions']==1 and result[0]['text']=='Hello'
 assert [r['id'] for r in json.loads(sent[0]['messages'][1]['content'][0]['text'])['pending_regions']]==[1]

 assert 'NOT transcription' in sent[0]['messages'][0]['content']
 assert json.loads(sent[0]['messages'][1]['content'][0]['text'])['pending_regions'][0]['source_text']=='Hello'


def test_provider_stream_error_body_is_preserved(tmp_path):
 class ProviderFailure(Reply):
  def __iter__(self):
   yield b'data: {"error":{"code":503,"message":"upstream disconnected"}}\n'
   yield b'\n'
 with patch('urllib.request.urlopen',return_value=ProviderFailure([])),pytest.raises(RuntimeError,match='upstream disconnected'):
  request({'stream':True},'test',tmp_path,['0'],lambda *a:None,normalizer=lambda v:v)
 raw=json.loads((tmp_path/'raw.json').read_text('utf8'))
 assert raw['provider_error']=={'code':503,'message':'upstream disconnected'}
