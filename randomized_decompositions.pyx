# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from numpy.linalg import svd
from scipy.linalg.interpolative import interp_decomp, reconstruct_interp_matrix
from Infrastructure.utils import RowVector, Matrix

def random_svd(np.ndarray[double, ndim=2] A, const int k, const int increment) -> (Matrix, RowVector, Matrix):
    cdef Py_ssize_t m = A.shape[0]
    cdef np.ndarray[double, ndim=1] sigma
    cdef np.ndarray[double, ndim=2] Q, VT, U

    _, _, Q = svd(np.random.randn(k + increment, m).dot(A), full_matrices=False, compute_uv=True)
    Q = Q[:k, :].T
    U, sigma, VT = svd(A.dot(Q), full_matrices=False, compute_uv=True)
    VT = Q.dot(VT.T).T
    return U, sigma, VT


def random_id(np.ndarray[double, ndim=2] A, const int k, const int increment) -> (Matrix, Matrix):
    cdef Py_ssize_t m = A.shape[0]
    cdef np.ndarray[int, ndim=1] idx
    cdef np.ndarray[double, ndim=2] B, P, proj

    idx, proj = interp_decomp(np.random.randn(k + increment, m).dot(A), k, rand=False)
    P = reconstruct_interp_matrix(idx, proj)
    B = A[:, idx[:k]]
    return B, P
