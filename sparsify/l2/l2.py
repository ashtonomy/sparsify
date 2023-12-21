import numpy as np
# from numba import jit, njit
import logging

logger = logging.getLogger(__name__)

# @njit
def compute_p(sq_fro_norm, A_ij):
    """
    Compute probability of A_ij
    """
    if np.isclose(A_ij, 0.0):
        p_ij = 0.0
    else:
        p_ij = (A_ij**2) / sq_fro_norm
    return p_ij

# @njit
def l2(A, eps):
    """
    Implements algorithm 1 from Drineas & Zouzias 2011:
    A note on element-wise matrix sparsification via a 
    matrix-valued Bernstein inequality.
    
    Args
        A: data matrix (n x n) # Currently only supports square matrices
        eps: epsilon 
    
    Returns
        Sparse sketch A_tilde, same size as input matrix A
    """

    raise DeprecationWarning

    n = A.shape[0]

    logger.debug(f"n = {n}")

    # Truncate small values
    trunc_A = np.where(np.abs(A) > (eps / (2 * n)), A, 0.0)

    sq_fro_norm = np.linalg.norm(trunc_A, "fro") ** 2

    # Compute s
    s = 28 * n * np.log(np.sqrt(2)*n) * sq_fro_norm
    s = s / (eps ** 2)
    s = int(np.ceil(s))

    logger.debug(f"s = {s}")

    # Compute probability distribution
    compute_p_vec = np.vectorize(compute_p)
    P = compute_p_vec(sq_fro_norm, trunc_A)

    # Sample and scale
    idxs = [*range(n*n)]
    sample_idxs = np.random.choice(idxs, s, replace=True, p=P.flatten()/P.sum())
    A_tilde = np.zeros((n*n,))
    for i in sample_idxs:
        A_tilde[i] += A.flatten()[i] / P.flatten()[i]
    A_tilde = A_tilde.reshape(n,n) / s
    return A_tilde

