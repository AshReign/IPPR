import numpy as np
import pytest
from p2.cache_io import VideoCacheReader,VideoCacheWriter

def test_roundtrip_empty_frames_resume_and_config_guard(tmp_path):
    p=tmp_path/'v.sqlite'
    spec={'work_shape_hw':[2,4],'model_sha256':'test-only'}
    boxes=np.array([[0,0,3,2]],dtype=np.float32)
    masks=np.array([[[0,.1,.5,.501],[1,.6,.4,0]]],dtype=np.float16)
    w=VideoCacheWriter(p,spec)
    w.append(0,0,0,boxes,np.array([.1],dtype=np.float32),masks)
    w.close()
    w=VideoCacheWriter(p,spec)
    assert w.next_frame==1
    w.append(1,.04,1,np.empty((0,4)),np.empty(0),np.empty((0,2,4)))
    w.finish(2);w.close()
    r=VideoCacheReader(p)
    assert r.complete and len(r)==2
    assert np.array_equal(r.read(0)['soft_mask'],masks)
    assert r.read(1)['soft_mask'].shape==(0,2,4)
    r.close()
    with pytest.raises(ValueError,match='different'):
        VideoCacheWriter(p,{**spec,'model_sha256':'other'})

def test_reject_missing_order_and_nonmonotonic_time(tmp_path):
    w=VideoCacheWriter(tmp_path/'v.sqlite',{'work_shape_hw':[2,4]})
    empty=(np.empty((0,4)),np.empty(0),np.empty((0,2,4)))
    with pytest.raises(ValueError,match='contiguous'):
        w.append(1,0,0,*empty)
    w.append(0,0,0,*empty)
    with pytest.raises(ValueError,match='increasing'):
        w.append(1,0,1,*empty)
    w.close()
