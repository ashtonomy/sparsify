import numpy as np
# from numba import jit, njit
import scipy
from scipy.optimize import fminbound

#@njit
def compute_p(l1_norm, sq_fro_norm, A_ij, alpha):
    """
    Equation 11 from Kundu et al. 2017
    Computes sampling probabilities
    """
    if np.isclose(A_ij, 0.0):
        p_ij = 0.0
    else:
        p_ij = alpha * np.abs(A_ij) / l1_norm
        p_ij = p_ij + (1-alpha) * (A_ij**2) / sq_fro_norm
    
    return p_ij

#@njit
def compute_s(A, f_a, eps, delta, l2_norm):
    """
    Equation 11 from Kundu et al. 2017
    Computes an optimal s given a data matrix A, 
    parameter alpha, eps, and failure-bound delta
    """
    m, n = A.shape
    s = 2 / (((eps * l2_norm)**2))
    s = s * np.log((m+n) / delta)
    return s * f_a

#@njit
def compute_gamma(A, alpha, l1_norm, l2_norm, sq_fro_norm):
    """
    Equation 6 from Kundu et al. 2017
        Computes value of gamma
    """
    m,n = A.shape

    A_g = np.zeros_like(A)
    partial_prod = (1-alpha) * (l1_norm / sq_fro_norm)

    for i in range(m):
        for j in range(n):
            if A[i,j] != 0:
                A_g[i,j] = l1_norm / (alpha + partial_prod*np.abs(A[i,j]))

    return A_g.max() + l2_norm

#@njit
def compute_ksi(sq_fro_norm, l1_norm, A_ij, alpha):
    """
    Equation 8 from Kundu et al. 2017
        Computes ksi of A at i,j
    """
    if A_ij == 0:
        ksi_ij = 0
    else:
        denom = (alpha * sq_fro_norm) / ((np.abs(A_ij) * l1_norm)) + (1-alpha)
        ksi_ij = sq_fro_norm / denom
    
    return ksi_ij

#@njit
def compute_rho_squared(A, alpha, l1_norm, sq_fro_norm):
    """
    Equation 7 from Kundu et al. 2017
        Computes rho squared
    """
    smallest_sv = scipy.linalg.svdvals(A)[-1]
    compute_ksi_vec = np.vectorize(compute_ksi)
    ksi_A = compute_ksi_vec(sq_fro_norm, l1_norm, A, alpha)
    max_col_sum = ksi_A.sum(0).max()
    max_row_sum = ksi_A.sum(1).max()

    rho_2 = max(max_row_sum, max_col_sum) - (smallest_sv**2)
    return rho_2

#@njit
def compute_f(A, alpha, eps, l1_norm, l2_norm, sq_fro_norm):
    """
    Parameterization for s from Kundu et al. 2017
    
    Returns f(alpha)
    """
    rho_squared = compute_rho_squared(A, alpha, l1_norm, sq_fro_norm)
    gamma = compute_gamma(A, alpha, l1_norm, l2_norm, sq_fro_norm)
    f_alpha = rho_squared + gamma * eps * l2_norm / 3.0
    return f_alpha

def compute_alpha(A, eps, l1_norm, l2_norm, sq_fro_norm, method):
    """
    Equation 10 from Kundu et al. 2017: Computes an optimal alpha 
        given matrix A and error bound eps
    """

    if method == "l1":
        alpha = 1.0
        f_a = compute_f(A, alpha, eps, l1_norm, l2_norm, sq_fro_norm)
    else:
        min_func = lambda a: compute_f(A, a, eps, l1_norm, l2_norm, sq_fro_norm)
        res = fminbound(min_func, 0, 1, full_output=True)
        alpha, f_a, _, _ = res
    
    return f_a, alpha

def hybrid(A, eps, delta, method):
    """
    Return a sparse sketch (or low-rank approximation) of input matrix A. 
        Algorithm derived from from Kundu et al. 2017: Recovering PCA and 
        sparse-PCA via Hybrid-(l1,l2) Sparse Sampling of Data Elements
    
    Args
        A: data matrix (n x m)
        method: One of "hybrid" or "l1": l1 is same as 
                Achlioptas et al 2013b
        eps: epsilon 
        delta: Failure probability bound
        n_alphas: Number of alphas to test (over range (0,1]
                Default is 10   
    
    Returns
        Sparse sketch A_tilde, same size as input matrix A
    """
    m, n = A.shape

    # Precompute for faster computation
    l1_norm = np.linalg.norm(A, 1)
    l2_norm = np.linalg.norm(A, 2)
    sq_fro_norm = np.linalg.norm(A, "fro") ** 2

    # Get alpha value and compute optimal s
    f_a, alpha = compute_alpha(A, eps, l1_norm, l2_norm, sq_fro_norm, method)
    s = compute_s(A, f_a, eps, delta, l2_norm)
    s = np.ceil(s)
    s = int(s)

    # Compute sampling probability distribution
    compute_p_vec = np.vectorize(compute_p)
    P = compute_p_vec(l1_norm, sq_fro_norm, A, alpha)

    # Sample and scale
    idxs = [*range(m*n)]
    sample_idxs = np.random.choice(idxs, s, replace=True, p=P.flatten()/P.sum())
    A_tilde = np.zeros((m*n,))
    for i in sample_idxs:
        A_tilde[i] += A.flatten()[i] / P.flatten()[i]
    A_tilde = A_tilde.reshape(m,n) / s
    return A_tilde