"""Install pinned upstream BandIt v2 and its official multilingual checkpoint."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib
import math
import urllib.request
from core import ROOT,command,atomic_json

COMMIT='d5563d9031e95fdaa3e5a73d5020b9a0df61adb6'
URL='https://zenodo.org/api/records/12701995/files/checkpoint-multi.ckpt/content'
MD5='fea2868787551b0cff36cfcf7c3622a3'
SIZE=446680129
CHECKPOINT=ROOT/'models/bandit-v2-multi.ckpt'
UPSTREAM=ROOT/'vendor/bandit-v2'


def checksum(path,algorithm='md5'):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,algorithm).hexdigest()


def install():
    if not (UPSTREAM/'.git').is_dir():
        command(['git','clone','--no-checkout','https://github.com/kwatcharasupat/bandit-v2.git',str(UPSTREAM)])
    command(['git','-C',str(UPSTREAM),'checkout',COMMIT])
    CHECKPOINT.parent.mkdir(parents=True,exist_ok=True)
    if not CHECKPOINT.is_file() or checksum(CHECKPOINT)!=MD5:
        # Small resumable parts avoid restarting a 447 MB download after interruption.
        parts=ROOT/'work/bandit-download';parts.mkdir(parents=True,exist_ok=True)
        part_size=8*1024*1024;count=math.ceil(SIZE/part_size)
        def download(index):
            start=index*part_size;end=min(SIZE,start+part_size)-1
            path=parts/f'{index:04d}.part'
            if path.is_file() and path.stat().st_size==end-start+1:return
            request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=120) as response,path.open('wb') as out:
                if response.status!=206 or response.headers.get('Content-Range','').split('/')[0]!=f'bytes {start}-{end}':
                    raise RuntimeError('Invalid BandIt download range.')
                while block:=response.read(256*1024):out.write(block)
            if path.stat().st_size!=end-start+1:raise RuntimeError('BandIt download is incomplete.')
        with ThreadPoolExecutor(max_workers=8) as pool:
            tasks=[pool.submit(download,n) for n in range(count)]
            for done,future in enumerate(as_completed(tasks),1):
                future.result();print(f'BandIt download: {done}/{count}',flush=True)
        temporary=CHECKPOINT.with_suffix('.partial')
        with temporary.open('wb') as out:
            for n in range(count):
                with (parts/f'{n:04d}.part').open('rb') as source:
                    while block:=source.read(4*1024*1024):out.write(block)
        if checksum(temporary)!=MD5:raise RuntimeError('BandIt official weights checksum mismatch.')
        temporary.replace(CHECKPOINT)
    atomic_json(CHECKPOINT.with_suffix('.json'),{'model':'BandIt v2 Multi','source':URL,'upstream_commit':COMMIT,
        'published_md5':MD5,'sha256':checksum(CHECKPOINT,'sha256'),'bytes':SIZE})
    print('Installed and verified official BandIt v2 Multi.',flush=True)


if __name__=='__main__':install()
