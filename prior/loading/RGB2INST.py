import numpy as np

def RGB2INST(x:np.ndarray):
    assert x.shape[-1] == 3
    y = x.astype(np.int32)
    y = y[..., 0] + y[..., 1] * 256 + y[..., 2] * 256**2
    return y