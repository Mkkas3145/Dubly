"""Short-lived batched GPU speaker encoder. No ASR, no padded batch pooling."""
from pathlib import Path
import sys,json,hashlib,math,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/redimnet2'))


def run(manifest,destination):
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from redimnet2.redimnet2 import ReDimNet2Wrap
    from reference_bank_guard import MODEL_SHA,BASE_SHA
    torch.set_num_threads(4);started=time.monotonic()
    data=json.loads(Path(manifest).read_text('utf8'));rows=data['rows'];cache=Path(data['cache']);cache.mkdir(parents=True,exist_ok=True)
    vectors={};pending={}
    for row in rows:
        path=Path(row['path']);identity=hashlib.sha256(path.read_bytes()+json.dumps(data['policy'],sort_keys=True).encode()).hexdigest()
        saved=cache/f'{identity}.npy'
        if saved.exists():vectors[row['index']]=np.load(saved,allow_pickle=False);continue
        x,sr=sf.read(path,dtype='float32',always_2d=True);x=x.mean(1);g=math.gcd(sr,16000)
        x=resample_poly(x,16000//g,sr//g)
        if len(x)<3200:
            raise RuntimeError('Speaker verification input is shorter than 0.2 seconds')
        size=min(64000,len(x));starts=sorted(set([0,max(0,(len(x)-size)//2),max(0,len(x)-size)]))
        chunks=np.stack([x[a:a+size] for a in starts])
        pending[row['index']]=(saved,chunks)
    if pending:
        base_path=ROOT/'models/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt';plus_path=ROOT/'models/redimnet2-plus/pytorch_model_fsdp.bin'
        for path,expected in ((base_path,BASE_SHA),(plus_path,MODEL_SHA)):
            with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected:raise RuntimeError('Speaker checkpoint hash mismatch')
        base=torch.load(base_path,map_location='cpu',weights_only=True)
        cfg={**base['model_config'],'agg_gnorm':False};del base
        plus=torch.load(plus_path,map_location='cpu',weights_only=True)
        state={k.replace('_orig_mod.','').replace('module.','').removeprefix('encoder.model.'):v for k,v in plus.items() if 'encoder.model.' in k}
        model=ReDimNet2Wrap(**cfg);model.load_state_dict(state,strict=True);del plus,state
        device='cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cuda':torch.cuda.set_per_process_memory_fraction(min(1.,2*1024**3/torch.cuda.get_device_properties(0).total_memory))
        model=model.float().eval().to(device)
        # Match exact lengths; padding changes attentive statistics and identity.
        buckets={}
        for index,(saved,chunks) in pending.items():buckets.setdefault(chunks.shape[1],[]).append(index)
        for size,indices in buckets.items():
            for first in range(0,len(indices),4):
                batch=indices[first:first+4];chunks=np.concatenate([pending[i][1] for i in batch])
                with torch.inference_mode():z=torch.nn.functional.normalize(model(torch.from_numpy(chunks).to(device)).float(),dim=-1).cpu().numpy()
                pos=0
                for index in batch:
                    saved,parts=pending[index];v=z[pos:pos+len(parts)].mean(0);pos+=len(parts)
                    v=v/np.linalg.norm(v)
                    if not np.isfinite(v).all():raise RuntimeError('Invalid speaker embedding')
                    vectors[index]=v;temp=saved.with_suffix('.partial.npy');np.save(temp,v);temp.replace(saved)
                print(json.dumps({'done':len(vectors),'seconds':time.monotonic()-started}),flush=True)
    np.save(destination,np.stack([vectors[r['index']] for r in rows]))
    print(json.dumps({'done':len(vectors),'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':run(*sys.argv[1:])
