"""Dialogue playback has no edge envelope; reference preprocessing is separate."""
import numpy as np

FADE_SECONDS=0.
REFERENCE_FADE_SECONDS=0.
PADDING_SECONDS=0.


def fade_edges(values,rate,*,fade_in=True,fade_out=True):
    # Keep compatibility with playback/fitting callers without attenuating any
    # dialogue samples. Transitions belong to render.transition_weights only.
    return values.copy()


def edited_edge_fades(edit,duration,base_edges=None):
    return {'start':(edit.get('fade_in') or 0)>0,'end':(edit.get('fade_out') or 0)>0}


def reference_fade_edges(values,rate):
    """Reference audio retains the full voice; no automatic edge attenuation."""
    return values.copy()


def fade_lengths(fade_in,fade_out,duration):
    a=max(0.,float(fade_in or 0));b=max(0.,float(fade_out or 0));duration=max(0.,duration)
    if a+b>duration and a+b>0:
        factor=duration/(a+b);a*=factor;b*=factor
    return a,b


def apply_dialogue_fades(values,rate,fade_in=0.,fade_out=0.):
    """Explicit user envelope in OUTPUT seconds; sample count never changes."""
    result=values.copy();a,b=fade_lengths(fade_in,fade_out,len(result)/rate)
    for seconds,reverse in [(a,False),(b,True)]:
        count=min(len(result),round(seconds*rate))
        if count<1:continue
        curve=np.sin(np.linspace(0,np.pi/2,count,dtype=np.float64)).astype(np.float32) if count>1 else np.zeros(1,np.float32)
        if reverse:curve=curve[::-1]
        if result.ndim>1:curve=curve[:,None]
        if reverse:result[-count:]*=curve
        else:result[:count]*=curve
    return result
