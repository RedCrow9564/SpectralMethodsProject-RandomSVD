# -*- coding: utf-8 -*-
"""
test_data_creation.py - tests for data creation methods
=======================================================

This module contains the tests for the data creation in all the examples.

"""
import unittest
import numpy as np
from scipy.linalg import svdvals
from data_loader import get_data
from Infrastructure.utils import RowVector, Matrix
from Infrastructure.enums import ExperimentType
from main import choose_singular_values


class TestDataCreation(unittest.TestCase):
    """
    A class which contains tests for the validity of the created data in all the examples
    """
    def test_example_no_1_data(self):
        """
        Test data creation for Example no. 1

        This test validates the data created is ``data_size x data_size``, has rank 20
        and posses the expected singular values.

        """
        experiment_type: str = ExperimentType.ExampleNo1
        data_size: int = 70
        singular_values: RowVector = choose_singular_values(experiment_type)
        rank: int = len(singular_values)
        data: Matrix = get_data(experiment_type)(data_size, singular_values).as_numpy_arr()
        calculated_singular_values: RowVector = svdvals(data, check_finite=False)[:rank]
        self.assertTrue(np.allclose(data.shape, (data_size, data_size)))  # Validate data shape.
        self.assertEqual(np.linalg.matrix_rank(data, tol=1.8e-16), rank)  # Validate data rank.
        self.assertTrue(np.allclose(singular_values, calculated_singular_values))  # Validate singular values.

    def test_example_no_2_data(self):
        """
        Test data creation for Example no. 2

        This test validates the data created is ``data_size x data_size``, has rank 20
        and posses the expected singular values.

        """
        experiment_type: str = ExperimentType.ExampleNo2
        data_size: int = 70
        singular_values: RowVector = choose_singular_values(experiment_type)
        rank: int = len(singular_values)
        data: Matrix = get_data(experiment_type)(data_size, singular_values).as_numpy_arr()
        calculated_singular_values: RowVector = svdvals(data, check_finite=False)[:rank]
        self.assertTrue(np.allclose(data.shape, (data_size, data_size)))  # Validate data shape.
        self.assertEqual(np.linalg.matrix_rank(data, tol=singular_values[rank - 1] / 2), rank)  # Validate data rank.
        self.assertTrue(np.allclose(singular_values, calculated_singular_values))  # Validate singular values.

    def test_example_no_3_data(self):
        """
        Test data creation for Example no. 3

        This test validates the data created is ``data_size x data_size``, has rank 60
        and posses the expected singular values.

        """
        experiment_type: str = ExperimentType.ExampleNo3
        data_size: int = 70
        singular_values: RowVector = choose_singular_values(experiment_type)
        rank: int = len(singular_values)
        data: Matrix = get_data(experiment_type)(data_size, singular_values).as_numpy_arr()
        calculated_singular_values: RowVector = svdvals(data, check_finite=False)[:rank]
        self.assertTrue(np.allclose(data.shape, (data_size, data_size)))  # Validate data shape.
        self.assertEqual(np.linalg.matrix_rank(data, tol=singular_values[rank - 1] / 2), rank)  # Validate data rank.
        self.assertTrue(np.allclose(singular_values, calculated_singular_values))  # Validate singular values.

    def test_example_no_4_data(self):
        """
        Test data creation for Example no. 4

        This test validates the data created is ``data_size x data_size`` and posses the expected singular values.

        """
        experiment_type: str = ExperimentType.ExampleNo4
        data_size: int = 80
        singular_values: RowVector = choose_singular_values(experiment_type)
        known_singular_values_num: int = len(singular_values)
        data: Matrix = get_data(experiment_type)(data_size, singular_values).as_numpy_arr()
        calculated_singular_values: RowVector = svdvals(data, check_finite=False)[:known_singular_values_num]
        self.assertTrue(np.allclose(data.shape, (data_size, data_size)))  # Validate data shape.
        self.assertTrue(np.allclose(singular_values, calculated_singular_values))  # Validate singular values.

    def test_example_no_5_data(self):
        """
        Test data creation for Example no. 5

        This test validates the data created is ``data_size x data_size`` and posses the expected singular value
        :math:`10^{-17}` as the second largest singular value with multiplicity of at least ``data_size - 2``.

        """
        experiment_type: str = ExperimentType.ExampleNo5
        data_size: int = 70
        singular_values: RowVector = choose_singular_values(experiment_type)
        known_singular_values_num: int = data_size - 2
        data: Matrix = get_data(experiment_type)(data_size, singular_values).as_numpy_arr()
        calculated_singular_values: RowVector = svdvals(data, check_finite=False)[1:known_singular_values_num + 1]
        known_singular_values: RowVector = singular_values[0] * np.ones(known_singular_values_num)
        self.assertTrue(np.allclose(data.shape, (data_size, data_size)))  # Validate data shape.
        self.assertTrue(np.allclose(known_singular_values, calculated_singular_values))  # Validate singular values.


if __name__ == '__main__':
    unittest.main()
