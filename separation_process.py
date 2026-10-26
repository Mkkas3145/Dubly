"""Run GPU separation in a child so API waits retain no CUDA context."""
from pathlib import Path
import json,os,subprocess,sys
from core import ROOT

def prepare(original,stage_work,work,cfg,progress):
    args=[sys.executable,'-u',str(ROOT/'separation_process.py'),str(original),str(stage_work),str(work),cfg['separator'],cfg['background']]
    env=dict(os.environ);env.pop('DUBLY_VIDEO_ID',None);env['PYTHONIOENCODING']='utf-8'
    result=None
    with (Path(stage_work)/'separation-process.log').open('a',encoding='utf8') as log,subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace') as process:
        for line in process.stdout:
            try:event=json.loads(line)
            except ValueError:
                log.write(line);log.flush();continue
            logged={key:value for key,value in event.items() if key!='detail'} if 'separation_progress' in event else event
            log.write(json.dumps(logged,ensure_ascii=False)+'\n');log.flush()
            if 'separation_progress' in event:progress(event['separation_progress'],event.get('detail',''))
            if 'separation_result' in event:result=event['separation_result']
        if process.wait() or result is None:raise RuntimeError('Audio separation failed. See separation-process.log.')
    return Path(result['clean']),Path(result['background']) if result['background'] else None

def run(original,stage_work,work,separator,background_model):
    from separation import separate
    original,stage_work,work=map(Path,(original,stage_work,work));clean=stage_work/'dialogue.wav';background=None
    def progress(percent,detail=''):print(json.dumps({'separation_progress':percent,'detail':detail},ensure_ascii=False),flush=True)
    dedicated=background_model!='residual'
    if separator==background_model=='leap-xe-vocals':
        from leap_xe import prepare_pair
        clean,background=prepare_pair(original,stage_work,progress)
    elif separator==background_model=='mel-band-roformer-deux':
        from deux import prepare_pair
        clean,background=prepare_pair(original,stage_work,progress)
    else:
        if not clean.is_file():separate(original,clean,lambda p,d:progress(p*(.5 if dedicated else 1),d),model_name=separator)
        if dedicated:
            from background import prepare_background
            background=prepare_background(original,work,lambda p,d:progress(50+p*.5,d),model_name=background_model)
    print(json.dumps({'separation_result':{'clean':str(clean),'background':str(background) if background else None}}),flush=True)

if __name__=='__main__':run(*sys.argv[1:])
