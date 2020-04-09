# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
"""
matrices_classes.pxd - Matrix representations module
====================================================
This module contains declarations for convenient representations of matrices.
"""
import numpy as np
cimport numpy as np


cdef class ExperimentNo5Form:
    cdef double sigma
    cdef readonly Py_ssize_t shape[2]
    cdef readonly np.dtype dtype

    cdef inline double[:, ::1] dot(self, const double[:, ::1] other_vector)
    cdef inline double[:, ::1] transpose_dot(self, const double[:, ::1] other_vector)
    cdef inline double[:, ::1] left_dot(self, const double[:, ::1] other_vector)
    cdef inline double[:, ::1] slice_columns(self, const int[::1] idx)


cdef class MatInSVDForm:
    cdef readonly const double[:, ::1] U, V
    cdef readonly const double[::1] sigma
    cdef readonly ssize_t shape[2]
    cdef readonly np.dtype dtype

    cdef inline double[:, ::1] dot(self, const double[:, ::1] other_vector)
    cdef inline double[:, ::1] transpose_dot(self, const double[:, ::1] other_vector)
    cdef inline double[:, ::1] left_dot(self, const double[:, ::1] other_vector)
    cdef inline double[:, ::1] slice_columns(self, const int[::1] idx)


cdef class MatInIDForm:
    cdef readonly const double[:, ::1] B
    cdef readonly const double[::1, :] P
    cdef readonly size_t shape[2]
    cdef readonly np.dtype dtype

    cdef inline double[:, ::1] dot(self, const double[:, ::1] other_vector)
    cdef inline double[:, ::1] transpose_dot(self, const double[:, ::1] other_vector)
    cdef inline double[:, ::1] left_dot(self, const double[:, ::1] other_vector)
    cdef inline double[:, ::1] slice_columns(self, const int[::1] idx)

ctypedef fused GeneralMat:
    MatInSVDForm
    MatInIDForm
    ExperimentNo5Form
