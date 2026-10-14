"""Exact frozen estimator comparison including RNG and stateless squared loss."""
import numpy as np

def equal_state(a,b):
    assert type(a) is type(b), (type(a),type(b))
    if isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,np.random.RandomState):
        equal_state(a.get_state(),b.get_state())
    elif isinstance(a,np.random.Generator):
        equal_state(a.bit_generator.state,b.bit_generator.state)
    elif type(a).__module__=='sklearn._loss._loss' and type(a).__name__=='CyHalfSquaredError':
        # This Cython loss implementation is stateless; tree/bin state is
        # compared separately through the estimator's full Python state.
        assert a.__getstate__() is None and b.__getstate__() is None
    elif isinstance(a,np.dtype):
        assert a==b
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal_state(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_state(x,y)
    elif isinstance(a,(str,int,float,bool,type(None),np.generic)):
        np.testing.assert_array_equal(a,b)
    elif hasattr(a,'children_left'):
        equal_state(a.__getstate__(),b.__getstate__())
    elif hasattr(a,'__dict__'):
        equal_state(vars(a),vars(b))
    else:
        raise AssertionError('unsupported model state '+str(type(a)))
