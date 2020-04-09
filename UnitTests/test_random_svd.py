# -*- coding: utf-8 -*-
"""
test_random_svd.py - tests for Randomized Singular-Value Decomposition
======================================================================

This module contains the tests for the implementation of randomized SVD algorithm.

"""
import unittest
from numpy.random import randn as _randn
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True, build_in_temp=True, build_dir=r'.')
from randomized_decompositions import random_svd
from data_loader import get_data
from Infrastructure.enums import ExperimentType


class TestRandomSVD(unittest.TestCase):
    """
    A class which contains tests for the validity of the random_svd algorithm implementation.
    """
    def setUp(self):
        """
        This method sets the variables for the following tests.
        """
        self._m = 100
        self._n = 30
        self._k = 5
        self._increment = 20
        self._A = get_data(ExperimentType.ExampleNo2)(self._m, np.arange(2 * self._k).astype(float))
        self._approximation = random_svd(self._A, self._k, self._increment)
        self._U = self._approximation.U
        self._sigma = self._approximation.sigma
        self._VT = self._approximation.V.T
        self._approximation = self._approximation.as_numpy_arr()
        self._A = self._A.as_numpy_arr()

    def test_matrices_shapes(self):
        """
        This methods tests the shapes of the matrices :math:`U` and :math:`V` in the decomposition.
        """
        self.assertTrue(self._U.shape, (self._m, self._k))
        self.assertTrue(self._VT.shape, (self._k, self._n))

    def test_matrices_svd_decomposition(self):
        """
        This methods tests if the output decomposition satisfies the properties of SVD decomposition.
        """
        self.assertTrue(np.allclose(np.dot(self._U.T, self._U), np.eye(self._k)))
        self.assertTrue(np.allclose(np.dot(self._VT, self._VT.T), np.eye(self._k)))
        self.assertTrue(np.all(self._sigma.base > 0))

    def test_decomposition_rank(self):
        """
        This methods tests if the number of positive singular values is equal to the approximation rank.
        """
        self.assertEqual(len(self._sigma), self._k)

    def test_approximation_estimate(self):
        """
        This methods tests if the random SVD satisfies the theoretical bound. There is a probability
        of less then :math:`10^{-17}` this bound won't be satisfied...
        """
        real_sigmas = np.linalg.svd(self._A, full_matrices=False, compute_uv=False)
        estimate_error = np.linalg.norm(self._A - self._approximation)
        expected_bound = 10 * np.sqrt(self._n * (self._k + self._increment))
        expected_bound *= real_sigmas[self._k]
        self.assertLessEqual(estimate_error, expected_bound)


if __name__ == '__main__':
    unittest.main()
