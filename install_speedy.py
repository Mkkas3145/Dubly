"""Build the pinned, evaluated Speedy sources for Windows or WSL Ubuntu."""
from pathlib import Path
import json,shlex,subprocess,os,shutil
from core import ROOT
from speedy import wslpath

PINS={'speedy':('https://github.com/google/speedy.git','e05c9b6fa473881e9c2b5ddaa323435b5513012c'),
 'sonic':('https://github.com/waywardgeek/sonic.git','b93885dcb70aae50c6f76b0fe4e0868f029a077e'),
 'kissfft':('https://github.com/mborgerding/kissfft.git','45b3ad4c0830abae6b2ea857864fe60a676dd896')}

def main():
    tools=ROOT/'tools/google-speedy';tools.mkdir(parents=True,exist_ok=True)
    for name,(url,commit) in PINS.items():
        path=tools/name
        if not path.exists():subprocess.run(['git','clone',url,str(path)],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',commit],check=True)
    worker=ROOT/'packaging/speedy/worker.py'
    shutil.copy2(worker,tools/'worker.py')
    if os.name=='nt':
        destination=tools/'windows';destination.mkdir(exist_ok=True)
        for name in ('build.cmd','exports.def'):shutil.copy2(ROOT/'packaging/speedy'/name,destination/name)
        subprocess.run(['cmd','/c',str(destination/'build.cmd')],check=True)
        (tools/'build.json').write_text(json.dumps({'sources':PINS,'runtime':'Windows native / Python stdlib','algorithm_modified':False},indent=2),encoding='utf8')
        print('Google Speedy ready.');return
    commands=[
      'gcc -O3 -fPIC -DSONIC_INTERNAL -Isonic -c sonic/sonic.c -o sonic_internal.o',
      'gcc -O3 -fPIC -DKISS_FFT -Isonic -Ikissfft -Ispeedy -c speedy/speedy.c -o speedy.o',
      'gcc -O3 -fPIC -Isonic -Ispeedy -c speedy/soniclib.c -o soniclib.o',
      'gcc -O3 -fPIC -Ikissfft -c kissfft/kiss_fft.c -o kiss_fft.o',
      'gcc -shared sonic_internal.o speedy.o soniclib.o kiss_fft.o -lm -o libspeedy.so']
    subprocess.run(['wsl','-d','Ubuntu','--','bash','-lc','cd '+shlex.quote(wslpath(tools))+' && '+' && '.join(commands)],check=True)
    (tools/'build.json').write_text(json.dumps({'sources':PINS,'flags':'-O3 -fPIC -DKISS_FFT',
      'runtime':'WSL Ubuntu gcc / Python 3 stdlib','algorithm_modified':False},indent=2),encoding='utf-8')
    print('Google Speedy ready.')

if __name__=='__main__':main()
