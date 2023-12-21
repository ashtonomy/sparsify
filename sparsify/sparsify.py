import os
import sys
import numpy as np

# Add hybrid/l2 directories to path and import 
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(parent_dir, "hybrid"))
sys.path.append(os.path.join(parent_dir, "l2"))

from hybrid import hybrid
from l2 import l2

def sparsify(
    A: np.ndarray, 
    eps: float=0.05, 
    delta: float=0.1, 
    method: str="hybrid"
) -> np.ndarray:
    """
    Return a sparse sketch of input matrix A via element-wise 
    sparsification. Methods include hybrid, l1, and l2 sampling.

    Args
        A: Input matrix (m x n) to sparsify
        eps: error bound epsilon
        delta: failure probability bound
        method: One of hybrid, l1
    
    Returns: 
        Sparse sketch of A, same size as input matrix A
    """

    if method == "hybrid" or method == "l1":
        return hybrid(A, eps, delta, method)
    elif method == "l2":
        raise ValueError("l2-Sparsification not supported")
        # return l2(A, eps)
    else:
        raise ValueError(f"Method <{method}> not supported.")
    
