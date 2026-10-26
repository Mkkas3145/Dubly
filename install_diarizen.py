"""Install pinned public DiariZen v2 weights for comparison only."""
import hashlib,json,urllib.request,time
from concurrent.futures import ThreadPoolExecutor
from core import ROOT,atomic_json
MODEL='BUT-FIT/diarizen-wavlm-large-s80-md-v2'
REVISION='027b3e221b9d81dce2be794084f1dbd3ba2da403'
EMBEDDING='pyannote/wespeaker-voxceleb-resnet34-LM'
EMBEDDING_REVISION='837717ddb9ff5507820346191109dc79c958d614'
ASSETS=ROOT/'models/diarizen-large-v2'

def install():
    jobs=[]
    for repo,revision,sub in [(MODEL,REVISION,'segmentation'),(EMBEDDING,EMBEDDING_REVISION,'embedding')]:
        items=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true'))
        for item in items:
            name=item['path']
            if item['type']!='file' or name=='.gitattributes':continue
            jobs.append((repo,revision,name,item,ASSETS/sub/name))
    def fetch(job):
        repo,rev,name,item,path=job;path.parent.mkdir(parents=True,exist_ok=True)
        sha=item.get('lfs',{}).get('oid')
        def digest(p):
            with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
        if path.exists() and path.stat().st_size==item['size'] and (not sha or digest(path)==sha):return
        temp=path.with_suffix(path.suffix+'.download');size=0;last=-1
        with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',timeout=120) as response,temp.open('wb') as target:
            while block:=response.read(1024**2):
                target.write(block);size+=len(block);percent=int(size/item['size']*100)
                if percent//10!=last and item['size']>1024**2:
                    last=percent//10;print(json.dumps(dict(file=str(path.relative_to(ASSETS)),percent=percent,MB=round(size/1e6,1),total_MB=round(item['size']/1e6,1))),flush=True)
        if size!=item['size'] or (sha and digest(temp)!=sha):raise RuntimeError('download checksum mismatch: '+name)
        temp.replace(path)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,jobs))
    atomic_json(ASSETS/'installation.json',dict(model=MODEL,revision=REVISION,embedding_revision=EMBEDDING_REVISION,bytes=sum(j[3]['size'] for j in jobs)))
    print('DiariZen weights ready',flush=True)

if __name__=='__main__':install()
