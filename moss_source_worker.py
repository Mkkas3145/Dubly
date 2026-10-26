"""Isolated local MOSS transcription; source timestamps are never adjusted."""
from pathlib import Path
import argparse,gc,json,sys,time

ROOT=Path(__file__).resolve().parent


def run(source,folder):
    import torch
    from transformers import AutoModelForCausalLM,AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages,generate_transcription,DEFAULT_PROMPT
    from moss_prompt import transcription_prompt
    prompt=transcription_prompt(DEFAULT_PROMPT)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    def event(stage,**data):print(json.dumps({'stage':stage,**data},ensure_ascii=False),flush=True)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    event('loading');started=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/moss-transcribe-diarize',trust_remote_code=True,
        local_files_only=True,dtype='auto',attn_implementation='sdpa').to(device=device,dtype=dtype).eval()
    processor=AutoProcessor.from_pretrained(ROOT/'models/moss-transcribe-diarize',trust_remote_code=True,local_files_only=True)
    load=time.perf_counter()-started;event('loaded',load_seconds=load)
    last=0
    def tokens(n):
        nonlocal last
        if n-last>=64:last=n;event('generating',generated_tokens=n)
    try:
        started=time.perf_counter()
        result=generate_transcription(model,processor,build_transcription_messages(Path(source),prompt=prompt),
            max_length=131072,max_new_tokens=32768,do_sample=False,device=device,dtype=dtype,
            input_callback=lambda n:event('input_ready',prompt_tokens=n),token_callback=tokens)
        (folder/'raw.txt').write_text(result['text'],'utf8')
        rows=[{'start':s.start,'end':s.end,'speaker':s.speaker,'text':s.text} for s in parse_transcript(result['text'])]
        data={**result,'segments':rows,'load_seconds':load,'inference_seconds':time.perf_counter()-started,
            'max_new_tokens':32768,'prompt':prompt,
            'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0}
        (folder/'result.partial.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),'utf8')
        (folder/'result.partial.json').replace(folder/'result.json')
        event('done',segments=len(rows),inference_seconds=data['inference_seconds'])
    finally:
        del model,processor;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('folder');args=parser.parse_args()
    run(args.source,args.folder)
