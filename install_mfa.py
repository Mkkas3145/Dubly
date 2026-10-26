"""Optional isolated MFA installation/recovery, independent of the app venv."""
from pathlib import Path
import hashlib,shutil,subprocess
from core import atomic_json
from mfa_hint import ENVIRONMENT,MODEL_ROOT,MODELS,model_path,runtime_environment


def main():
    executable=ENVIRONMENT/'Scripts/mfa.exe'
    if not executable.is_file():
        candidates=[shutil.which('conda'),str(Path.home()/'miniforge3/Scripts/conda.exe'),str(Path.home()/'miniconda3/Scripts/conda.exe')]
        conda=next((p for p in candidates if p and Path(p).is_file()),None)
        if conda is None:raise RuntimeError('Miniforge is required: https://conda-forge.org/download/')
        subprocess.run([conda,'create','--prefix',str(ENVIRONMENT),'--override-channels','-c','conda-forge','montreal-forced-aligner=3.4.3','-y'],check=True)
    MODEL_ROOT.mkdir(parents=True,exist_ok=True);env=runtime_environment()
    for kind in MODELS:
        if not model_path(kind).is_file():subprocess.run([str(executable),'model','download',kind,'korean_mfa'],env=env,check=True)
    version=subprocess.check_output([str(executable),'version'],env=env,encoding='utf-8').strip()
    manifest={'mfa_version':version,'language':'ko','role':'advisory_only',
        'models':{kind:{'sha256':hashlib.sha256(model_path(kind).read_bytes()).hexdigest()} for kind in MODELS}}
    atomic_json(MODEL_ROOT/'installed.json',manifest)
    print('MFA ready:',version)


if __name__=='__main__':main()
