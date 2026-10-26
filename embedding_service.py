"""Resident local LaBSE encoder, shared by successive video processes."""
from pathlib import Path
import io,json,os,sys,threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
from core import ROOT,atomic_json

VERSION=1


class Encoder:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer,AutoModel
        from safetensors.torch import load_file
        self.torch=torch;self.lock=threading.Lock();self.cache=OrderedDict();self.calls=0
        path=ROOT/'models/labse';self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        self.model=AutoModel.from_pretrained(str(path),local_files_only=True).eval().to(self.device)
        self.dense=torch.nn.Linear(768,768)
        self.dense.load_state_dict({k.removeprefix('linear.'):v for k,v in load_file(str(path/'2_Dense/model.safetensors')).items()})
        self.dense=self.dense.eval().to(self.device)

    def encode(self,texts):
        with self.lock,self.torch.inference_mode():
            wanted=list(dict.fromkeys(t for t in texts if t not in self.cache))
            for start in range(0,len(wanted),32):
                chunk=wanted[start:start+32]
                batch=self.tokenizer(chunk,padding=True,truncation=True,max_length=256,return_tensors='pt').to(self.device)
                hidden=self.model(**batch).last_hidden_state[:,0]
                vectors=self.torch.nn.functional.normalize(self.torch.tanh(self.dense(hidden)),p=2,dim=1).cpu().numpy()
                self.cache.update(zip(chunk,vectors))
            result=np.asarray([self.cache[t] for t in texts],dtype=np.float32) if texts else np.empty((0,768),np.float32)
            while len(self.cache)>50000:self.cache.popitem(last=False)
            self.calls+=1;return result


def main(state,token):
    engine=Encoder()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def authorized(self):return self.headers.get('Authorization')=='Bearer '+token
        def do_GET(self):
            if not self.authorized():self.send_error(403);return
            body=json.dumps(dict(pid=os.getpid(),version=VERSION,model_loads=1,calls=engine.calls,
                cache_texts=len(engine.cache),device=engine.device)).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
        def do_POST(self):
            if not self.authorized():self.send_error(403);return
            try:
                size=int(self.headers.get('Content-Length',0))
                if size>32*1024**2:raise ValueError('oversized request')
                texts=json.loads(self.rfile.read(size))['texts']
                if not isinstance(texts,list) or any(not isinstance(t,str) for t in texts):raise ValueError('invalid texts')
                output=io.BytesIO();np.save(output,engine.encode(texts),allow_pickle=False)
                data=output.getvalue();self.send_response(200);self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            except Exception as e:
                self.send_error(500,type(e).__name__+': '+str(e))
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    atomic_json(state,dict(pid=os.getpid(),port=server.server_port,token=token,version=VERSION))
    server.serve_forever()


if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2])
