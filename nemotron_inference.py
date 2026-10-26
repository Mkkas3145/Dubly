"""Bounded batched Nemotron inference with content-addressed exact reuse."""
from collections import OrderedDict
import hashlib
import math
from pathlib import Path
import numpy as np


def forward_batch(model,processor,waves,device):
    import torch
    # Extract each clip independently: padding raw waveforms before STFT would
    # change the last real mel frames at a short clip's boundary.
    inputs=[processor(w,sampling_rate=16000).to(device,dtype=model.dtype) for w in waves]
    features=torch.nn.utils.rnn.pad_sequence([i['input_features'][0] for i in inputs],batch_first=True)
    masks=torch.nn.utils.rnn.pad_sequence([i['attention_mask'][0] for i in inputs],batch_first=True)
    with torch.inference_mode():
        logits=model(input_features=features,attention_mask=masks).logits
        values=logits.float().sigmoid().cpu().numpy()
    return [v[:math.ceil(len(w)/160)].copy() for v,w in zip(values,waves)]


class CachedInference:
    def __init__(self,folder,revision,dtype,forward,*,batch_size=4,source_identity=None,on_oom=lambda:None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.salt=revision.encode()+str(dtype).encode();self.forward=forward
        self.batch_size=batch_size;self.source_identity=source_identity;self.on_oom=on_oom
        self.memory=OrderedDict()
        self.stats={'requests':0,'memory_hits':0,'disk_hits':0,'deduplicated':0,'model_calls':0,'model_items':0,'oom_splits':0}

    def __call__(self,wave):return self.batch([wave])[0]

    def _remember(self,key,value):
        self.memory[key]=value;self.memory.move_to_end(key)
        if len(self.memory)>512:self.memory.popitem(last=False)

    def _run(self,items):
        try:
            values=self.forward([wave for _,wave in items])
        except Exception as error:
            # Retry the same comparisons in smaller batches only for a real OOM.
            if len(items)==1 or error.__class__.__name__ not in ('OutOfMemoryError','CUDAOutOfMemoryError'):raise
            error.__traceback__=None
            self.batch_size=min(self.batch_size,max(1,len(items)//2))
            self.stats['oom_splits']+=1;self.on_oom()
            mid=len(items)//2
            self._run(items[:mid]);self._run(items[mid:]);return
        if len(values)!=len(items):raise RuntimeError('Nemotron batch result count mismatch')
        self.stats['model_calls']+=1;self.stats['model_items']+=len(items)
        for (key,wave),value in zip(items,values):
            active=np.asarray(value)>.5
            if active.ndim!=2 or len(active)!=math.ceil(len(wave)/160):raise RuntimeError('Nemotron batch frame extent mismatch')
            np.savez_compressed(self.folder/f'{key}.npz',active=active)
            self._remember(key,active)

    def batch(self,waves):
        keys=[];pending={};results={}
        for wave in waves:
            self.stats['requests']+=1
            key=hashlib.sha256(self.salt+wave.tobytes()).hexdigest();keys.append(key)
            if key in results or key in pending:self.stats['deduplicated']+=1;continue
            if key in self.memory:
                self.stats['memory_hits']+=1;results[key]=self.memory[key];self.memory.move_to_end(key);continue
            path=self.folder/f'{key}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as saved:results[key]=saved['active']
                self.stats['disk_hits']+=1;self._remember(key,results[key])
            else:pending[key]=wave
        # Similar lengths share a batch, with a fixed padded-audio memory budget.
        items=sorted(pending.items(),key=lambda item:len(item[1]));batch=[]
        for item in items:
            if batch and (len(batch)>=self.batch_size or len(item[1])*(len(batch)+1)>480*16000):
                self._run(batch)
                for key,_ in batch:results[key]=self.memory[key]
                batch=[]
            batch.append(item)
        if batch:
            self._run(batch)
            for key,_ in batch:results[key]=self.memory[key]
        return [results[key] for key in keys]
