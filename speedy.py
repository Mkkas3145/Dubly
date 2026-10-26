"""Google Speedy nonlinear time scaling, using the evaluated upstream sources."""
from pathlib import Path
import json,math,shutil,subprocess,tempfile,sys
import numpy as np
import soundfile as sf
from core import ROOT,SR

ENGINE='google_speedy'
VERSION='e05c9b6-sonic-b93885d'
TOOLS=ROOT/'tools/google-speedy'
WINDOWS_BINARY=TOOLS/'windows/speedy.dll'
BINARY=WINDOWS_BINARY if WINDOWS_BINARY.is_file() else TOOLS/'libspeedy.so'
WORKER=TOOLS/'worker.py'
DISTRO='Ubuntu'
PREFERRED_LENGTH_ERROR_FRAMES=round(.003*SR)
MAX_LENGTH_ERROR_FRAMES=round(.020*SR)

def wslpath(path):
    path=Path(path).resolve()
    if len(path.drive)!=2 or path.drive[1]!=':':raise RuntimeError('Speedy working paths must be on a local Windows drive.')
    return '/mnt/'+path.drive[0].lower()+str(path)[2:].replace('\\','/')

def ready():
    return BINARY.is_file() and WORKER.is_file() and (BINARY.suffix=='.dll' or shutil.which('wsl') is not None)

def stretch(source,destination,frames):
    if not ready():raise RuntimeError('Google Speedy is not installed. Run install_speedy.py.')
    if not isinstance(frames,(int,np.integer)) or frames<1:raise RuntimeError('Invalid Speedy target duration.')
    source,destination=Path(source),Path(destination)
    values,rate=sf.read(source,dtype='float32',always_2d=True)
    if rate!=SR or values.shape[1]!=2 or not len(values) or not np.isfinite(values).all():raise RuntimeError('Invalid Speedy input audio format.')
    if frames==len(values):
        sf.write(destination,values,SR,subtype='FLOAT');return values
    # Sonic's upstream nonlinear path uses short samples internally, even when
    # called via its float API. Normalize only if necessary to avoid saturation.
    gain=max(1.,float(np.max(np.abs(values)))/.9999)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='speedy-',dir=destination.parent) as temporary:
        folder=Path(temporary);input_file=folder/'input.wav';output_file=folder/'output.wav'
        sf.write(input_file,values/gain,SR,subtype='PCM_16')
        request=folder/'request.json';response=folder/'response.json'
        native=BINARY.suffix=='.dll'
        runtime_path=(lambda p:str(Path(p).resolve())) if native else wslpath
        request.write_text(json.dumps({'input':runtime_path(input_file),'output':runtime_path(output_file),
          'library':runtime_path(BINARY),'target_frames':int(frames),'maximum_under_frames':MAX_LENGTH_ERROR_FRAMES,
          'preferred_under_frames':PREFERRED_LENGTH_ERROR_FRAMES}),encoding='utf-8')
        args=[sys.executable,str(WORKER),str(request),str(response)] if native else ['wsl','-d',DISTRO,'--','python3',wslpath(WORKER),wslpath(request),wslpath(response)]
        result=subprocess.run(args,
          capture_output=True,timeout=120)
        if result.returncode or not output_file.exists() or not response.exists():
            log=ROOT/'logs/speedy-errors.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('ab') as diagnostics:diagnostics.write(result.stderr+b'\n')
            raise RuntimeError('Google Speedy processing or duration validation failed. Previous output was preserved. See logs/speedy-errors.log.')
        metadata=json.loads(response.read_text(encoding='utf-8'))
        output,sr=sf.read(output_file,dtype='float32',always_2d=True);output*=gain
        if (sr!=SR or output.shape[1]!=2 or not np.isfinite(output).all() or
            not 0<=frames-len(output)<=MAX_LENGTH_ERROR_FRAMES or metadata['output_frames']!=len(output)):
            raise RuntimeError('Speedy target duration validation failed. Speech was not trimmed or processed with a different engine.')
        # Prefer 3ms, allow at most 20ms early when pitch-synchronous output
        # length is discrete. Never overrun adjacent dialogue. Do not pad,
        # crop, or run a second stretching algorithm to force an exact sample count.
        final=folder/'validated.wav';sf.write(final,output,SR,subtype='FLOAT')
        final.replace(destination)
        return output
