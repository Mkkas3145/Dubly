"""Assemble an offline Windows payload from the installed, tested environments.

No credentials, projects, results, caches or unselected models are copied.
Run with the production Python interpreter after building the desktop launcher.
"""
from pathlib import Path
import argparse,ast,concurrent.futures,hashlib,importlib.metadata as metadata,json,os,re,shutil,struct,sys,zipfile
from packaging.requirements import Requirement
from packaging.markers import default_environment

APP=Path(__file__).resolve().parents[1]
ROOTS=['torch','torchaudio','pytorch-lightning','fastapi','uvicorn','pydantic','python-multipart','python-dotenv','soundfile','librosa','PyYAML','beartype','rotary-embedding-torch','einops','huggingface-hub','faster-whisper','scipy','numba','transformers','uroman','panphon','SudachiPy','sudachidict_core','filelock']


def dependencies():
    result={};environment=default_environment();environment['extra']=''
    def visit(name):
        dist=metadata.distribution(name);key=dist.metadata['Name'].lower().replace('_','-')
        if key in result:return
        result[key]=dist
        for value in dist.requires or []:
            req=Requirement(value)
            if not req.marker or req.marker.evaluate(environment):visit(req.name)
    for name in ROOTS:visit(name)
    return result


def pe_imports(path):
    """Read the import tables without requiring an installed compiler or SDK."""
    import mmap
    with Path(path).open('rb') as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as data:
        if data[:2]!=b'MZ':return []
        pe=struct.unpack_from('<I',data,0x3c)[0]
        if data[pe:pe+4]!=b'PE\0\0':return []
        sections=struct.unpack_from('<H',data,pe+6)[0];size=struct.unpack_from('<H',data,pe+20)[0];optional=pe+24
        magic=struct.unpack_from('<H',data,optional)[0];directory=optional+(112 if magic==0x20b else 96)
        table=optional+size
        def offset(rva):
            for n in range(sections):
                at=table+n*40;virtual,address,raw_size,raw=struct.unpack_from('<IIII',data,at+8)
                if address<=rva<address+max(virtual,raw_size):return raw+rva-address
            return rva
        result=[]
        for index,step,name_offset in ((1,20,12),(13,32,4)):
            rva,_=struct.unpack_from('<II',data,directory+index*8)
            if not rva:continue
            at=offset(rva)
            while any(data[at:at+step]):
                name=struct.unpack_from('<I',data,at+name_offset)[0]
                if name:
                    begin=offset(name);end=data.find(b'\0',begin);result.append(data[begin:end].decode('ascii').lower())
                at+=step
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--python-embed',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);server=out/'server';server.mkdir(exist_ok=True)
    jobs=[];sources=set()
    def add(source,destination):
        source,destination=Path(source),Path(destination)
        if not source.is_file():raise FileNotFoundError(source)
        if destination in sources:return
        sources.add(destination);jobs.append((source,destination))
    def tree(source,destination,skip=()):
        for p in Path(source).rglob('*'):
            if p.is_file() and not any(part in ('__pycache__','.git','.pytest_cache','tests','test') for part in p.relative_to(source).parts) and p.suffix not in skip:add(p,Path(destination)/p.relative_to(source))
    for p in APP.glob('*.py'):add(p,server/p.name)
    for name in ('static','assets'):tree(APP/name,server/name)
    add(APP/'requirements.txt',server/'requirements.txt')
    tree(APP/'vendor/mel-band-roformer',server/'vendor/mel-band-roformer',('.ckpt','.pth'))
    tree(APP/'vendor/mel-band-roformer-deux',server/'vendor/mel-band-roformer-deux')
    add(APP/'models/mel-band-roformer-deux.ckpt',server/'models/mel-band-roformer-deux.ckpt')
    tree(APP/'models/mfa/pretrained_models',server/'models/mfa/pretrained_models')
    for name in ('worker.py','build.json','windows/speedy.dll'):add(APP/'tools/google-speedy'/name,server/'tools/google-speedy'/name)
    for name in ('speedy','sonic','kissfft'):tree(APP/'tools/google-speedy'/name,server/'tools/google-speedy'/name,('.obj','.o','.so','.dll','.exe'))
    python=out/'runtime/python';python.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.python_embed) as archive:archive.extractall(python)
    (python/'python311._pth').write_text('python311.zip\n.\nLib/site-packages\n../../server\nimport site\n','ascii')
    packages=dependencies();site=python/'Lib/site-packages'
    for name,dist in packages.items():
        for relative in dist.files or []:
            relative=Path(relative)
            if relative.is_absolute() or '..' in relative.parts or '__pycache__' in relative.parts or relative.suffix in ('.pyc','.pdb'):continue
            source=dist.locate_file(relative)
            if source.is_file():add(source,site/relative)
    # Runtime binaries are app-local; the target PC needs no separate VC installer.
    redist=Path(os.environ.get('DUBLY_VC_REDIST',r'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT'))
    for p in redist.glob('*.dll'):add(p,python/p.name)
    for name in ('ffmpeg','ffprobe'):
        binary=Path(shutil.which(name)).resolve();add(binary,out/f'runtime/bin/{name}.exe')
    # Keep MFA's Python and only DLLs referenced by its extensions. Reuse the
    # compatible CUDA 12 DLLs already shipped with PyTorch instead of duplicating them.
    original=APP/'.mfa-env';mfa=server/'.mfa-env'
    tree(original/'Lib',mfa/'Lib',('.pyc','.pdb'))
    tree(original/'DLLs',mfa/'DLLs',('.pdb',))
    for p in original.iterdir():
        if p.is_file() and (p.suffix=='.dll' or p.name in ('python.exe','LICENSE_PYTHON.txt')):add(p,mfa/p.name)
    available={p.name.lower():p for p in (original/'Library/bin').glob('*.dll')}
    available.update({p.name.lower():p for p in original.glob('*.dll')})
    torch_lib=Path(metadata.distribution('torch').locate_file('torch/lib'))
    shared={p.name.lower() for p in torch_lib.glob('*.dll')}
    queue=[src for src,dst in jobs if dst.is_relative_to(mfa) and src.suffix.lower() in ('.pyd','.dll','.exe')];seen=set();reused=set()
    for p in (original/'Library/bin').glob('*sndfile*.dll'):queue.append(p)
    for p in (original/'Library/bin').glob('mkl*.dll'):queue.append(p)
    for p in (original/'Library/bin').glob('libiomp*.dll'):queue.append(p)
    for binary in queue:
        key=str(binary).lower()
        if key in seen:continue
        seen.add(key)
        if binary.parent==original/'Library/bin':add(binary,mfa/'Library/bin'/binary.name)
        for name in pe_imports(binary):
            if name.startswith(('cublas','cufft','curand','cusolver','cusparse','cudart','nvrtc','nvjitlink')) and name in shared:reused.add(name);continue
            if name in available and str(available[name]).lower() not in seen:queue.append(available[name])
    for name in ('LICENSE','README.md'):
        if (APP/name).is_file():add(APP/name,out/name)
    for p in (APP/'work/desktop-publish').glob('*'):
        if p.is_file() and p.suffix!='.pdb':add(p,out/p.name)
    print('Copying',len(jobs),'files',round(sum(p.stat().st_size for p,_ in jobs)/2**30,2),'GiB',flush=True)
    def copy(pair):
        source,destination=pair;destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists() or destination.stat().st_size!=source.stat().st_size or destination.stat().st_mtime_ns!=source.stat().st_mtime_ns:shutil.copy2(source,destination)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(copy,jobs))
    manifest={'packages':{k:v.version for k,v in packages.items()},'mfa_shared_cuda_dlls':sorted(reused),'files':len(jobs),'credentials_included':False,'projects_included':False,'offline_runtime':True}
    (out/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2),'utf8')
    print('BUNDLE COMPLETE',out,flush=True)


if __name__=='__main__':main()
