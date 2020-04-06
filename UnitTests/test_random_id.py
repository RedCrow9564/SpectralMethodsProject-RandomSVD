# -*- coding: utf-8 -*-
"""
test_random_id.py - tests for Randomized Interpolative Decomposition
====================================================================

This module contains the tests for the implementation of randomized ID algorithm.

"""
import unittest
from numpy.random import randn as _randn
import numpy as np
from scipy.linalg import svdvals
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
from randomized_decompositions import random_id


class TestRandomID(unittest.TestCase):
    """
    A class which contains tests for the validity of the random_id algorithm implementation.
    """
    def setUp(self):
        """
        This method sets the variables for the following tests.
        """
        self._m = 100
        self._n = 30
        self._k = 5
        self._increment = 20
        self._A = _randn(self._m, self._n)
        self._B, self._P = random_id(self._A, self._k, self._increment)
        self._approximation = np.dot(self._B, self._P)
        self._P = self._P.base

    def test_matrices_shapes(self):
        """
        This methods tests the shapes of the matrices :math:`B` and :math:`P` in the decomposition.
        """
        self.assertTrue(self._B.shape, (self._m, self._k))
        self.assertTrue(self._P.shape, (self._k, self._n))

    def test_interpolative_decomposition(self):
        """
        This methods tests if the decomposition satisfies the properties of interpolative-decomposition.
        """
        self.assertTrue(np.all(self._P <= 2))  # Validate entries of P are between -1 and 2.
        self.assertTrue(np.all(self._P >= -2))
        # Validate P's norm is bound by the theoretical bound
        self.assertLessEqual(np.linalg.norm(self._P), np.sqrt(self._k * (self._n - self._k) + 1))
        self.assertGreaterEqual(svdvals(self._P)[-1], 1)  # Validate the least singular value of P is at least 1.

        for unit_vector in np.eye(self._k):  # Validate P has kxk identity matrix as a sub-matrix.
            self.assertIn(unit_vector, self._P.T)

        for col in self._B.T:  # Validate every column of B is also a column of A.
            self.assertIn(col, self._A.T)

    def test_approximation_estimate(self):
        """
        This methods tests if the random ID satisfies the theoretical bound. There is a probability
        of less then :math:`10^{-17}` this bound won't be satisfied...
        """
        real_sigmas = np.linalg.svd(self._A, full_matrices=False, compute_uv=False)
        estimate_error = np.linalg.norm(self._A - self._approximation)
        expected_bound = 10 * np.sqrt(self._n * (self._k + self._increment) * self._m * self._k)
        expected_bound *= real_sigmas[self._k]
        self.assertLessEqual(estimate_error, expected_bound)


if __name__ == '__main__':
    unittest.main()
