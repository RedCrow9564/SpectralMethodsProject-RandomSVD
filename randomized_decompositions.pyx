# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
"""
randomized_decompositions.pyx -
================================
dffdgdg

"""
import numpy as np
cimport numpy as np
from numpy.linalg import svd
from scipy.linalg.interpolative import interp_decomp, reconstruct_interp_matrix
from Infrastructure.utils import RowVector, Matrix

def random_svd(const double[:, ::1] A, const int k, const int increment) -> (Matrix, RowVector, Matrix):
    """

    Args:
        A(Matrix): ffff
        k(int): Approximation rank.
        increment(int): The extra sampled columns in the approximation (beyond the first ``k`` columns.

    Returns:
        Matrices :math:`U, V` and :math:`\sigma` for which the SVD approximation is :math:`U\sigmaV^{T}
    """
    cdef Py_ssize_t m = A.shape[0]
    cdef double[::1] sigma
    cdef double[::1, :] Q, VT
    cdef double[:, ::1] H, U

    _, _, H = svd(np.dot(np.random.randn(k + increment, m), A), full_matrices=False, compute_uv=True)
    Q = H[:k, :].T
    U, sigma, H = svd(np.dot(A, Q), full_matrices=False, compute_uv=True)
    VT = np.dot(Q, H.T).T
    return U, sigma, VT


def random_id(np.ndarray[double, ndim=2, mode='c'] A, const int k, const int increment) -> (Matrix, Matrix):
    cdef Py_ssize_t m = A.shape[0]
    cdef np.ndarray[int, ndim=1, mode='c'] idx
    cdef np.ndarray[double, ndim=2, mode='fortran'] proj
    cdef double[::1, :] B, P

    idx, proj = interp_decomp(np.dot(np.random.randn(k + increment, m), A), k, rand=False)
    P = reconstruct_interp_matrix(idx, proj)
    B = A[:, idx[:k]]
    return B, P
