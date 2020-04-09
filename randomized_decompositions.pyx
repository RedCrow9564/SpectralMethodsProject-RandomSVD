# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
# distutils: extra_compile_args = /openmp
# distutils: extra_link_args = /openmp
"""
randomized_decompositions.pyx -
================================
dffdgdg

"""
import numpy as np
cimport numpy as np
from numpy.linalg import svd
from libc.math cimport sqrt
from scipy.linalg.interpolative import interp_decomp
from Infrastructure.utils import Matrix


cdef inline double[:, ::1] mat_scalar_product(const double[:, ::1] mat, const double scalar):
    cdef double[:, ::1] result = np.empty_like(mat)
    cdef Py_ssize_t i, j

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = scalar * mat[i, j]
    return result

cdef inline void vector_scalar_div(double[::1] vec, const double scalar):
    cdef Py_ssize_t i, j
    for j in range(vec.shape[0]):
        vec[j] /= scalar

cdef inline double[:, :] multiply_by_diagonal_mat(const double[:, :] mat, const double[::1] vec):
    cdef ssize_t i,j
    cdef double[:, ::1] result = mat.copy()

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            result[i, j] = mat[i, j] * vec[i]
    return result

cdef class ExperimentNo5Form:
    cdef double sigma
    cdef readonly Py_ssize_t shape[2]
    cdef readonly np.dtype dtype
    cdef double[:, ::1] common_vector

    def __init__(self, const (Py_ssize_t, Py_ssize_t) mat_shape, const double sigma):
        self.shape[0] = mat_shape[0]
        self.shape[1] = mat_shape[1]
        self.sigma = sigma
        self.dtype = np.dtype(np.double)


    # cdef inline double[:, ::1] dot(self, const double[:, ::1] other_vector):
    def dot(self, other_vector):
        cdef double[::1] column_sums = np.sum(other_vector, axis=0)
        cdef double[:, ::1] result = mat_scalar_product(other_vector, self.sigma)

        for j in range(other_vector.shape[1]):
            result[0, j] += column_sums[j] / sqrt(<double>self.shape[1])
        return result

    # cdef inline double[:, ::1] transpose_dot(self, const double[:, ::1] other_vector):
    def transpose_dot(self, other_vector):
        cdef double[:, ::1] result = mat_scalar_product(other_vector, self.sigma)
        cdef double[::1] first_row = other_vector[0, :].copy()
        vector_scalar_div(first_row, sqrt(<double>self.shape[1]))
        return np.tile(first_row, (self.shape[0], 1)) + result

    # cdef inline double[:, ::1] left_dot(self, const double[:, ::1] other_vector):
    def left_dot(self, other_vector):
        cdef double[:, ::1] result = mat_scalar_product(other_vector, self.sigma)
        cdef double[::1] first_col = other_vector[:, 0].copy()
        vector_scalar_div(first_col, sqrt(<double>self.shape[1]))
        return np.tile(np.reshape(first_col, (-1, 1)), (1, self.shape[1])) + result

    cdef inline double[:, ::1] slice_columns(self, const int[::1] idx):
        cdef double[:, ::1] result = np.zeros((self.shape[0], len(idx)), dtype=np.double)
        cdef Py_ssize_t i

        for i in range(len(idx)):
            result[0, i] += 1 / sqrt(<double>self.shape[1])
            result[idx[i], i] += self.sigma
        return result

    def as_numpy_arr(self) -> Matrix:
        cdef double[:, ::1] result = self.sigma * np.eye(self.shape[0], self.shape[1])
        cdef Py_ssize_t j

        for j in range(self.shape[1]):
            result[0, j] += 1 / sqrt(<double>self.shape[1])
        return result.base

    def matmat(self, other):
        return self.dot(other)

    def matvec(self, other):
        return self.dot(other[:, None])

    def rmatvec(self, other):
        return self.transpose_dot(other[:, None])



cdef class MatInSVDForm:
    cdef readonly const double[:, ::1] U, V
    cdef readonly const double[::1] sigma
    cdef readonly ssize_t shape[2]
    cdef readonly np.dtype dtype

    def __init__(self, const double[:, ::1] mat_U, const double[::1] sigmas, const double[:, ::1] mat_V):
        self.U = mat_U
        self.sigma = sigmas
        self.V = mat_V
        self.shape = (mat_U.shape[0], mat_V.shape[0])
        self.dtype = np.dtype(np.double)

    cdef inline double[:, ::1] dot(self, const double[:, ::1] other_vector):
        return np.dot(self.U, multiply_by_diagonal_mat(np.dot(self.V.T, other_vector), self.sigma))

    cdef inline double[:, ::1] transpose_dot(self, const double[:, ::1] other_vector):
        return np.dot(self.V, multiply_by_diagonal_mat(np.dot(self.U.T, other_vector), self.sigma))

    cdef inline double[:, ::1] left_dot(self, const double[:, ::1] other_vector):
        return np.dot(np.dot(other_vector, self.U), multiply_by_diagonal_mat(self.V.T, self.sigma))

    cdef inline double[:, ::1] slice_columns(self, const int[::1] idx):
        return np.dot(self.U, multiply_by_diagonal_mat(self.V.base[idx, :].T, self.sigma))

    def as_numpy_arr(self) -> Matrix:
        return np.array(np.dot(self.U, np.multiply(self.V, self.sigma).T))

    def matmat(self, other):
        return self.dot(other)

    def matvec(self, other):
        return self.dot(other[:, None])

    def rmatvec(self, other):
        return self.transpose_dot(other[:, None])


cdef class MatInIDForm:
    cdef readonly const double[:, ::1] B
    cdef readonly const double[::1, :] P
    cdef readonly size_t shape[2]
    cdef readonly np.dtype dtype

    def __init__(self, const double[:, ::1] mat_B, const double[::1, :] mat_P):
        self.B = mat_B
        self.P = mat_P
        self.shape = (mat_B.shape[0], mat_P.shape[1])
        self.dtype = np.dtype(np.float64)

    cdef inline double[:, ::1] dot(self, const double[:, ::1] other_vector):
        return np.dot(self.B, np.dot(self.P, other_vector))

    cdef inline double[:, ::1] transpose_dot(self, const double[:, ::1] other_vector):
        return np.dot(self.P.T, np.dot(self.B.T, other_vector))

    cdef inline double[:, ::1] left_dot(self, const double[:, ::1] other_vector):
        return np.dot(np.dot(other_vector, self.B), self.P)

    cdef inline double[:, ::1] slice_columns(self, const int[::1] idx):
        return np.dot(self.B, self.P.base[idx, :].T)

    def as_numpy_arr(self) -> Matrix:
        return np.array(np.dot(self.B, self.P))

    def matmat(self, other):
        return self.dot(other)

    def matvec(self, other):
        return self.dot(other[:, None])

    def rmatvec(self, other):
        return self.transpose_dot(other[:, None])


ctypedef fused GeneralMat:
    MatInSVDForm
    MatInIDForm
    ExperimentNo5Form


def random_svd(GeneralMat A, const int k, const int increment) -> Matrix:
    """

    Args:
        A(Matrix): ffff
        k(int): Approximation rank.
        increment(int): The extra sampled columns in the approximation (beyond the first ``k`` columns.

    Returns:
        Matrices :math:`U, V` and :math:`\sigma` for which the SVD approximation is :math:`U\sigmaV^{T}
    """
    cdef ssize_t m = A.shape[0]
    cdef const double[::1] sigma
    cdef const double[:, ::1] Q, U, H

    Q = svd(A.transpose_dot(np.random.randn(m, k + increment)), full_matrices=False, compute_uv=True)[0]
    Q = np.ascontiguousarray(Q[:, :k])
    U, sigma, H = svd(A.dot(Q), full_matrices=False, compute_uv=True)
    return MatInSVDForm(U, sigma, np.dot(Q, H.T))


def random_id(GeneralMat A, const int k, const int increment) -> Matrix:
    cdef ssize_t m = A.shape[0]
    cdef const int[::1] idx
    cdef const double[::1, :] P, proj
    cdef const double[:, ::1] B

    idx, proj = interp_decomp(A.left_dot(np.random.randn(k + increment, m)).base, k, rand=False)[:2]
    P = np.hstack([np.eye(k), proj])[:, np.argsort(idx)]
    B = A.slice_columns(idx[:k])
    return MatInIDForm(B, P)
