"""Resident, serialized Qwen timestamp inference; never edits generated audio."""
from pathlib import Path
import json,os,sys,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from core import ROOT,atomic_json

VERSION=1
MODEL='Qwen/Qwen3-ForcedAligner-0.6B'

def main(state,token):
    import torch
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ForcedAligner
    path=snapshot_download(MODEL,local_files_only=True)
    model=Qwen3ForcedAligner.from_pretrained(path,dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='cuda:0' if torch.cuda.is_available() else 'cpu',attn_implementation='sdpa')
    lock=threading.Lock();calls=0
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def authorized(self):return self.headers.get('Authorization')=='Bearer '+token
        def do_GET(self):
            if not self.authorized():self.send_error(403);return
            body=json.dumps({'version':VERSION,'model':MODEL,'model_loads':1,'calls':calls}).encode()
            self.send_response(200);self.end_headers();self.wfile.write(body)
        def do_POST(self):
            nonlocal calls
            if not self.authorized():self.send_error(403);return
            try:
                size=int(self.headers.get('Content-Length',0))
                if size>1024**2:raise ValueError('Alignment request too large')
                data=json.loads(self.rfile.read(size));audio=Path(data['audio']).resolve()
                if not audio.is_relative_to((ROOT/'work').resolve()):raise ValueError('Audio outside workspace work directory')
                with lock:
                    started=time.perf_counter()
                    result=model.align(audio=str(audio),text=data['text'],language=data['language'])[0]
                    output={'words':[{'word':w.text,'start':w.start_time,'end':w.end_time} for w in result],
                        'seconds':time.perf_counter()-started,'model':MODEL};calls+=1
                body=json.dumps(output,ensure_ascii=False).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except Exception as e:self.send_error(500,type(e).__name__+': '+str(e))
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    atomic_json(state,{'pid':os.getpid(),'port':server.server_port,'token':token,'version':VERSION})
    server.serve_forever()

if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2])
