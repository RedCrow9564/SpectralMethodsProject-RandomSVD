# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
"""
matrices_classes.pyx - Matrix representations module
====================================================
This module contains classes for convenient representations of matrices.
"""
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from Infrastructure.utils import Matrix

cdef inline double[:, ::1] mat_scalar_product(const double[:, ::1] mat, const double scalar):
    """
    Function for performing matrix-scalar product as a C-level loop.
    
    Args:
        mat(Matrix): The matrix for the product.
        scalar(const double): The scalar...
        
    Returns:
        The product matrix.
    """
    cdef double[:, ::1] result = mat.copy()
    cdef Py_ssize_t i, j

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] *= scalar
    return result

cdef inline void vector_scalar_div(double[::1] vec, const double scalar):
    """
    In-place vector-scalar division as a C-level loop.
    
    Args:
        vec(Vector): The vector.
        scalar(const double): The scalar...
    """
    cdef Py_ssize_t i, j
    for j in range(vec.shape[0]):
        vec[j] /= scalar

cdef inline double[:, :] multiply_by_diagonal_mat(const double[:, :] mat, const double[::1] vec):
    """
    Function for performing diagonal matrix product with another matrix as a C-level loop.
    
    Args:
        mat(const Matrix): The general matrix for the product.
        vec(const Vector): The diagonal elements as a vector.
        
    Returns:
        The product matrix.
    """
    cdef ssize_t i,j
    cdef double[:, ::1] result = mat.copy()

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            result[i, j] = mat[i, j] * vec[i]
    return result

cdef class ExperimentNo5Form:
    r"""
    A class for storing a matrix :math:`A\in\mathbb{R}^{m \times n}` as :math:`A=uv^{T}+\sigma I_n` where :math:`u=(1,0,...,0)`
    and :math:`v^{T}=\frac{1}{\sqrt n}(1, 1, ..., 0)`
    """
    def __init__(self, const (Py_ssize_t, Py_ssize_t) mat_shape, const double sigma):
        self.shape[0] = mat_shape[0]
        self.shape[1] = mat_shape[1]
        self.sigma = sigma
        self.dtype = np.dtype(np.double)


    cdef inline double[:, ::1] dot(self, const double[:, ::1] other_matrix):
        """
        Computes the product :math:`AG` with a general matrix :math:`G\in\R^{n \times l}`
        
        Args:
            other_matrix(Matrix): The matrix G.
            
        Returns:
            The result of :math:`AG`. 
        """
        cdef double[::1] column_sums = np.sum(other_matrix, axis=0)
        cdef double[:, ::1] result = mat_scalar_product(other_matrix, self.sigma)

        for j in range(other_matrix.shape[1]):
            result[0, j] += column_sums[j] / sqrt(<double>self.shape[1])
        return result

    cdef inline double[:, ::1] transpose_dot(self, const double[:, ::1] other_matrix):
        """
        Computes the product :math:`A^{T}G` with a general matrix :math:`G\in\R^{n \times l}`
        
        Args:
            other_matrix(Matrix): The matrix G.
            
        Returns:
            The result of :math:`A^{T}G`. 
        """
        cdef double[:, ::1] result = mat_scalar_product(other_matrix, self.sigma)
        cdef double[::1] first_row = other_matrix[0, :].copy()
        vector_scalar_div(first_row, sqrt(<double>self.shape[1]))
        return np.tile(first_row, (self.shape[0], 1)) + result

    cdef inline double[:, ::1] left_dot(self, const double[:, ::1] other_matrix):
        """
        Computes the product :math:`GA` with a general matrix :math:`G\in\R^{l \times m}`
        
        Args:
            other_matrix(Matrix): The matrix G.
            
        Returns:
            The result of :math:`GA`. 
        """
        cdef double[:, ::1] result = mat_scalar_product(other_matrix, self.sigma)
        cdef double[::1] first_col = other_matrix[:, 0].copy()
        vector_scalar_div(first_col, sqrt(<double>self.shape[1]))
        return np.tile(np.reshape(first_col, (-1, 1)), (1, self.shape[1])) + result

    cdef inline double[:, ::1] slice_columns(self, const int[::1] idx):
        """
        Picks columns of :math:`A` in requested indices.
        
        Args:
            idx(const int[::1]): The requested column indices.
            
        Returns:
            The columns of :math:`A` in the requested indices.
        """
        cdef double[:, ::1] result = np.zeros((self.shape[0], len(idx)), dtype=np.double)
        cdef Py_ssize_t i

        for i in range(len(idx)):
            result[0, i] += 1 / sqrt(<double>self.shape[1])
            result[idx[i], i] += self.sigma
        return result

    def as_numpy_arr(self) -> Matrix:
        """
        Converts an object matrix into a Numpy array.

        Returns:
            A Numpy array which represents this matrix.
        """
        cdef double[:, ::1] result = self.sigma * np.eye(self.shape[0], self.shape[1])
        cdef Py_ssize_t j

        for j in range(self.shape[1]):
            result[0, j] += 1 / sqrt(<double>self.shape[1])
        return result.base

    def matmat(self, other):
        """
        A Python wrapper for :method:`dot` for matrix-matrix product.
        """
        return self.dot(other)

    def matvec(self, other):
        """
        A Python wrapper for :method:`dot` for matrix-vector product.
        """
        return self.dot(other[:, None])

    def rmatvec(self, other):
        """
        A Python wrapper for :method:`transpose_dot` for matrix-vector product.
        """
        return self.transpose_dot(other[:, None])



cdef class MatInSVDForm:
    r"""
    A class for storing an SVD decomposition for a matrix :math:`A\in\mathbb{R}^{m \times n}`: :math:`A=U\Sigma V^{T}`
    where :math:`U\in\mathbb{R}^{m \times r}` and :math:`V\in\mathbb{R}^{r \times n}` have orthonormal columns and
    :math:`\Sigma\in\mathbb{R}^{r \times r}` is diagonal and positive-definite.
    """
    def __init__(self, const double[:, ::1] mat_U, const double[::1] sigmas, const double[:, ::1] mat_V):
        self.U = mat_U
        self.sigma = sigmas
        self.V = mat_V
        self.shape = (mat_U.shape[0], mat_V.shape[0])
        self.dtype = np.dtype(np.double)

    cdef inline double[:, ::1] dot(self, const double[:, ::1] other_matrix):
        """
        Computes the product :math:`AG` with a general matrix :math:`G\in\R^{n \times l}`
        
        Args:
            other_matrix(Matrix): The matrix G.
            
        Returns:
            The result of :math:`AG`. 
        """
        return np.dot(self.U, multiply_by_diagonal_mat(np.dot(self.V.T, other_matrix), self.sigma))

    cdef inline double[:, ::1] transpose_dot(self, const double[:, ::1] other_matrix):
        """
        Computes the product :math:`A^{T}G` with a general matrix :math:`G\in\R^{n \times l}`
        
        Args:
            other_matrix(Matrix): The matrix G.
            
        Returns:
            The result of :math:`A^{T}G`. 
        """
        return np.dot(self.V, multiply_by_diagonal_mat(np.dot(self.U.T, other_matrix), self.sigma))

    cdef inline double[:, ::1] left_dot(self, const double[:, ::1] other_matrix):
        """
        Computes the product :math:`GA` with a general matrix :math:`G\in\R^{l \times m}`
        
        Args:
            other_matrix(Matrix): The matrix G.
            
        Returns:
            The result of :math:`GA`. 
        """
        return np.dot(np.dot(other_matrix, self.U), multiply_by_diagonal_mat(self.V.T, self.sigma))

    cdef inline double[:, ::1] slice_columns(self, const int[::1] idx):
        """
        Picks columns of :math:`A` in requested indices.
        
        Args:
            idx(const int[::1]): The requested column indices.
            
        Returns:
            The columns of :math:`A` in the requested indices.
        """
        return np.dot(self.U, multiply_by_diagonal_mat(self.V.base[idx, :].T, self.sigma))

    def as_numpy_arr(self) -> Matrix:
        """
        Converts an object matrix into a Numpy array.

        Returns:
            A Numpy array which represents this matrix.
        """
        return np.array(np.dot(self.U, np.multiply(self.V, self.sigma).T))

    def matmat(self, other):
        """
        A Python wrapper for :method:`dot` for matrix-matrix product.
        """
        return self.dot(other)

    def matvec(self, other):
        """
        A Python wrapper for :method:`dot` for matrix-vector product.
        """
        return self.dot(other[:, None])

    def rmatvec(self, other):
        """
        A Python wrapper for :method:`transpose_dot` for matrix-vector product.
        """
        return self.transpose_dot(other[:, None])


cdef class MatInIDForm:
    r"""
    A class for storing an ID decomposition for a matrix :math:`A\in\mathbb{R}^{m \times n}`: :math:`A=BP` where
    :math:`B\in\mathbb{R}^{m \times r}` in the skeleton matrix and :math:`P\in\mathbb{R}^{r \times n}`
    is the interpolatory matrix.
    """
    def __init__(self, const double[:, ::1] mat_B, const double[::1, :] mat_P):
        self.B = mat_B
        self.P = mat_P
        self.shape = (mat_B.shape[0], mat_P.shape[1])
        self.dtype = np.dtype(np.float64)

    cdef inline double[:, ::1] dot(self, const double[:, ::1] other_matrix):
        """
        Computes the product :math:`AG` with a general matrix :math:`G\in\R^{n \times l}`
        
        Args:
            other_matrix(Matrix): The matrix G.
            
        Returns:
            The result of :math:`AG`. 
        """
        return np.dot(self.B, np.dot(self.P, other_matrix))

    cdef inline double[:, ::1] transpose_dot(self, const double[:, ::1] other_matrix):
        """
        Computes the product :math:`A^{T}G` with a general matrix :math:`G\in\R^{n \times l}`
        
        Args:
            other_matrix(Matrix): The matrix G.
            
        Returns:
            The result of :math:`A^{T}G`. 
        """
        return np.dot(self.P.T, np.dot(self.B.T, other_matrix))

    cdef inline double[:, ::1] left_dot(self, const double[:, ::1] other_matrix):
        """
        Computes the product :math:`GA` with a general matrix :math:`G\in\R^{l \times m}`
        
        Args:
            other_matrix(Matrix): The matrix G.
            
        Returns:
            The result of :math:`GA`. 
        """
        return np.dot(np.dot(other_matrix, self.B), self.P)

    cdef inline double[:, ::1] slice_columns(self, const int[::1] idx):
        """
        Picks columns of :math:`A` in requested indices.
        
        Args:
            idx(const int[::1]): The requested column indices.
            
        Returns:
            The columns of :math:`A` in the requested indices.
        """
        return np.dot(self.B, self.P.base[idx, :].T)

    def as_numpy_arr(self) -> Matrix:
        """
        Converts an object matrix into a Numpy array.

        Returns:
            A Numpy array which represents this matrix.
        """
        return np.array(np.dot(self.B, self.P))

    def matmat(self, other):
        """
        A Python wrapper for :method:`dot` for matrix-matrix product.
        """
        return self.dot(other)

    def matvec(self, other):
        """
        A Python wrapper for :method:`dot` for matrix-vector product.
        """
        return self.dot(other[:, None])

    def rmatvec(self, other):
        """
        A Python wrapper for :method:`transpose_dot` for matrix-vector product.
        """
        return self.transpose_dot(other[:, None])
