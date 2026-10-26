import numpy as np
from nemotron_inference import CachedInference


def wave(length,value):return np.full(length*160,value,np.float32)


def test_batch_deduplicates_exact_audio_restores_order_and_reuses_disk(tmp_path):
    calls=[]
    def forward(waves):
        calls.append([len(w) for w in waves])
        return [np.full((len(w)//160,8),w[0]>.5) for w in waves]
    infer=CachedInference(tmp_path,'revision','float32',forward,batch_size=4)
    values=[wave(8,.8),wave(4,.2),wave(8,.8),wave(6,.7)]
    results=infer.batch(values)
    assert len(calls)==1 and len(calls[0])==3
    assert [len(r) for r in results]==[8,4,8,6]
    assert [bool(r.any()) for r in results]==[True,False,True,True]
    assert infer.stats['deduplicated']==1
    other=CachedInference(tmp_path,'revision','float32',lambda _:(_ for _ in ()).throw(AssertionError('cache miss')))
    assert all(np.array_equal(a,b) for a,b in zip(results,other.batch(values)))
    assert other.stats['disk_hits']==3


def test_oom_splits_same_inputs_without_dropping_comparisons(tmp_path):
    class OutOfMemoryError(Exception):pass
    def forward(waves):
        if len(waves)>1:raise OutOfMemoryError()
        return [np.ones((len(waves[0])//160,8))]
    infer=CachedInference(tmp_path,'revision','float32',forward,batch_size=4)
    results=infer.batch([wave(i,.7) for i in range(1,5)])
    assert len(results)==4 and all(r.all() for r in results)
    assert infer.stats['oom_splits']==3 and infer.stats['model_items']==4


def test_non_memory_errors_are_not_hidden(tmp_path):
    import pytest
    def forward(waves):raise ValueError('bad input')
    with pytest.raises(ValueError,match='bad input'):
        CachedInference(tmp_path,'revision','float32',forward).batch([wave(2,.8),wave(3,.9)])
