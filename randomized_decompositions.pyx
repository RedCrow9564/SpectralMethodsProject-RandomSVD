# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
"""
randomized_decompositions.pyx - Randomized algorithms module
============================================================
This module contains the two randomized algorithms for matrix decompositions: the randomized SVD and randomized ID.
"""
import numpy as np
cimport numpy as np
from numpy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
from matrices_classes cimport GeneralMat, MatInSVDForm, MatInIDForm
from Infrastructure.utils import Matrix


def random_svd(GeneralMat A, const int k, const int increment) -> Matrix:
    """
    The function for Random SVD algorithm.

    Args:
        A(GeneralMat): The matrix to decompose.
        k(const int): Approximation rank.
        increment(const int): The extra sampled columns in the approximation (beyond the first ``k`` columns).

    Returns:
        Matrices :math:`U, V` and :math:`\Sigma` for which the SVD approximation is :math:`U\SigmaV^{T}`
    """
    cdef ssize_t m = A.shape[0]
    cdef const double[::1] sigma
    cdef const double[:, ::1] Q, U, H

    Q = svd(A.transpose_dot(np.random.randn(m, k + increment)), full_matrices=False, compute_uv=True)[0]
    Q = np.ascontiguousarray(Q[:, :k])
    U, sigma, H = svd(A.dot(Q), full_matrices=False, compute_uv=True)
    return MatInSVDForm(U, sigma, np.dot(Q, H.T))


def random_id(GeneralMat A, const int k, const int increment) -> Matrix:
    """
    The function for Random ID algorithm.

    Args:
        A(GeneralMat): The matrix to decompose.
        k(const int): Approximation rank.
        increment(const int): The extra sampled columns in the approximation (beyond the first ``k`` columns.

    Returns:
        Matrices :math:`U, V` and :math:`\Sigma` for which the SVD approximation is :math:`U\SigmaV^{T}`
    """
    cdef ssize_t m = A.shape[0]
    cdef const int[::1] idx
    cdef const double[::1, :] P, proj
    cdef const double[:, ::1] B

    idx, proj = interp_decomp(A.left_dot(np.random.randn(k + increment, m)).base, k, rand=False)[:2]
    P = np.hstack([np.eye(k), proj])[:, np.argsort(idx)]
    B = A.slice_columns(idx[:k])
    return MatInIDForm(B, P)
