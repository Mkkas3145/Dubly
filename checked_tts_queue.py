"""Parallel generation -> ordered inspection tasks -> parallel fitting."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from core import TTS_CONCURRENCY,FIT_CONCURRENCY

def run(items,generator,fitter,checker,progress,*,batch_size=16):
    if type(batch_size) is not int or batch_size<1:raise ValueError('Invalid inspection batch size')
    size=batch_size;groups=[list(range(i,min(i+size,len(items)))) for i in range(0,len(items),size)]
    generated={};checking={};generating={};fitting={};results={};ready=deque();submitted=0;sent=set()
    errors=[];cap=2*TTS_CONCURRENCY+size
    with ThreadPoolExecutor(max_workers=TTS_CONCURRENCY,thread_name_prefix='fish') as gp, \
         ThreadPoolExecutor(max_workers=2,thread_name_prefix='output-check') as cp, \
         ThreadPoolExecutor(max_workers=FIT_CONCURRENCY,thread_name_prefix='timing') as fp:
        def fill():
            nonlocal submitted
            while ready and len(fitting)<FIT_CONCURRENCY:
                index,p=ready.popleft();fitting[fp.submit(fitter,p)]=index
            for n,indices in enumerate(groups):
                if len(checking)>=2:break
                if n not in sent and all(i in generated for i in indices):
                    values=[generated.pop(i) for i in indices];sent.add(n)
                    checking[cp.submit(checker,values)]=indices
            while not errors and submitted<len(items) and len(generating)<TTS_CONCURRENCY and \
                  len(generating)+len(generated)+len(ready)+sum(len(v) for v in checking.values())<cap:
                index=submitted;generating[gp.submit(generator,index,items[index])]=index;submitted+=1
        def update():
            progress(len(results),len(generating),len(items)-submitted,len(fitting),
                len(ready)+len(generated),sum(len(v) for v in checking.values()))
        fill();update()
        while generating or checking or fitting or generated or ready:
            active=[*generating,*checking,*fitting]
            if not active:raise RuntimeError('TTS validation queue progression failed.')
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                if future in generating:
                    index=generating.pop(future)
                    try:generated[index]=future.result()
                    except Exception as error:errors.append((index,error))
                elif future in checking:
                    indices=checking.pop(future)
                    try:
                        values=future.result()
                        if len(values)!=len(indices):raise RuntimeError('TTS validation result count does not match.')
                        ready.extend(zip(indices,values))
                    except Exception as error:errors.append((indices[0],error))
                else:
                    index=fitting.pop(future)
                    try:results[index]=future.result()
                    except Exception as error:errors.append((index,error))
            if errors:break
            fill();update()
    if errors:
        index,error=errors[0];raise RuntimeError(f'TTS dialogue {index+1} validation or correction failed: {error}')
    return [results[i] for i in range(len(items))]
