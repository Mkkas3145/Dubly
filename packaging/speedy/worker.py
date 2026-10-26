"""Linux ctypes worker. Independent process per call: upstream globals are isolated."""
import ctypes as C,json,sys,wave
from pathlib import Path

def run(request):
    lib=C.CDLL(request['library']);ptr=C.c_void_p
    for name,args,ret in [
      ('sonicCreateStream',[C.c_int,C.c_int],ptr),('sonicDestroyStream',[ptr],None),
      ('sonicSetSpeed',[ptr,C.c_float],None),('sonicEnableNonlinearSpeedup',[ptr,C.c_float],None),
      ('sonicSetDurationFeedbackStrength',[ptr,C.c_float],None),
      ('sonicWriteShortToStream',[ptr,C.POINTER(C.c_short),C.c_int],C.c_int),
      ('sonicReadShortFromStream',[ptr,C.POINTER(C.c_short),C.c_int],C.c_int),('sonicFlushStream',[ptr],C.c_int)]:
        f=getattr(lib,name);f.argtypes=args;f.restype=ret
    with wave.open(request['input'],'rb') as src:
        rate=src.getframerate();channels=src.getnchannels();count=src.getnframes()
        assert src.getsampwidth()==2;data=src.readframes(count)
    def render(speed):
        stream=lib.sonicCreateStream(rate,channels)
        if not stream:raise RuntimeError('Speedy stream allocation failed')
        try:
            lib.sonicEnableNonlinearSpeedup(stream,1.)
            lib.sonicSetDurationFeedbackStrength(stream,0.)
            lib.sonicSetSpeed(stream,speed)
            output=[];buf=(C.c_short*(4096*channels))()
            def drain():
                while True:
                    n=lib.sonicReadShortFromStream(stream,buf,4096)
                    if not n:break
                    output.append(C.string_at(buf,n*channels*2))
            for start in range(0,len(data),1000*channels*2):
                b=data[start:start+1000*channels*2];frames=len(b)//(2*channels)
                samples=(C.c_short*(frames*channels)).from_buffer_copy(b)
                if lib.sonicWriteShortToStream(stream,samples,frames)<=0:raise RuntimeError('Speedy input failed')
                drain()
            if not lib.sonicFlushStream(stream):raise RuntimeError('Speedy flush failed')
            drain();return b''.join(output)
        finally:lib.sonicDestroyStream(stream)
    target=request['target_frames'];nominal=count/target
    low,high=(1.,max(4.,nominal*3)) if nominal>1 else (.001,1.)
    speed=nominal;best=None;history=[]
    # First establish an upper speed yielding a non-overrunning result.
    for _ in range(8):
        audio=render(high);n=len(audio)//(channels*2)
        history.append({'speed':high,'frames':n})
        if n<=target:best=(n,audio,high);break
        high*=2
    if best is None:raise RuntimeError('No non-overlapping Speedy result')
    for _ in range(18):
        audio=render(speed);n=len(audio)//(channels*2)
        history.append({'speed':speed,'frames':n})
        if n<=target and (best is None or n>best[0]):best=(n,audio,speed)
        if 0<=target-best[0]<=request.get('preferred_under_frames',request['maximum_under_frames']):break
        if n>target:low=speed
        else:high=speed
        speed=(low+high)/2
    n,audio,speed=best
    if not 0<=target-n<=request['maximum_under_frames']:
        raise RuntimeError(f'Cannot meet target without trimming: target={target}, output={n}, under={target-n}')
    with wave.open(request['output'],'wb') as out:
        out.setnchannels(channels);out.setsampwidth(2);out.setframerate(rate);out.writeframes(audio)
    return {'output_frames':n,'target_frames':target,'global_speed':speed,'history':history,
        'nonlinear_factor':1.,'duration_feedback_strength':0.}

if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(json.dumps(run(request)))
