import numpy as np

def null_space(a, rtol=1e-5):
    _, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()