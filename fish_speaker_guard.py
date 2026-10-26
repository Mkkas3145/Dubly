"""Fish identities can veto a Nemotron merge; never merge different Nemotron IDs."""
import numpy as np
import soundfile as sf
POLICY={'version':3,'identity':'nemotron_and_fish_intersection',
    'conflict':'split_at_quiet_word_gap_or_keep_native_speaker_reference_group'}
def quiet_boundary(source,a,b):
    if b<=a:return None
    info=sf.info(source);lo=round(a*info.samplerate);hi=round(b*info.samplerate)
    if hi-lo<2:return None
    with sf.SoundFile(source) as reader:
        reader.seek(lo);audio=reader.read(hi-lo,dtype='float32',always_2d=True).mean(axis=1)
    width=max(1,min(len(audio),round(info.samplerate*.01)));step=max(1,width//2)
    starts=np.arange(0,max(1,len(audio)-width+1),step)
    energy=np.array([np.mean(audio[i:i+width]**2) for i in starts])
    # Require a quiet boundary; a confident speaker label is not a safe cut.
    if np.sqrt(energy.min())>max(.001,float(np.sqrt(energy.max()))*.1):return None
    quiet=np.flatnonzero(energy<=np.quantile(energy,.4));middle=(len(audio)-width)/2
    i=int(min(quiet,key=lambda q:abs(starts[q]-middle)))
    return (lo+starts[i]+width//2)/info.samplerate
def split_regions(rows,source,clean):
    result=[]
    for row in rows:
        words=[w for w in source.get('words',[]) if w.get('fish_speaker') and
            min(w['end'],row['end'])>max(w['start'],row['start'])]
        changes=[(left,right) for left,right in zip(words,words[1:]) if left['fish_speaker']!=right['fish_speaker']]
        if not changes:result.append(dict(row));continue
        cuts=[];unsafe=False
        for left,right in changes:
            cut=quiet_boundary(clean,max(row['start'],left['end']),min(row['end'],right['start']))
            if cut is None:unsafe=True;break
            cuts.append(cut)
        if unsafe:result.append({**row,'fish_conflict_unresolved':True});continue
        points=[row['start'],*cuts,row['end']]
        result.extend({**row,'start':a,'end':b,'fish_split':bool(cuts)} for a,b in zip(points,points[1:]) if b>a)
    return [{**r,'id':i} for i,r in enumerate(result)]
