# -*- coding: utf-8 -*-
"""
data_loader.py - The data management module
===========================================

This module handles the fetching of the data from the local resources path, given in the configuration and arranging it
for our purposes of estimations. See the example for fetching the data for Example no. 1.

Example:
    get_data(ExperimentType.ExampleNo1) - Creating the data for Example no. 1 of the paper.

"""

from scipy.linalg import qr
import numpy as np
from Infrastructure.enums import ExperimentType
from Infrastructure.utils import Dict, RowVector, Matrix, create_factory, Callable
from randomized_decompositions import MatInSVDForm, ExperimentNo5Form


def _random_orthonormal_cols(data_size: int, columns: int) -> Matrix:
    return np.ascontiguousarray(qr(np.random.randn(data_size, columns), mode="economic", overwrite_a=True,
                                   check_finite=False)[0])


def _get_first_3_examples_data(data_size: int, singular_values: RowVector) -> Matrix:
    """
    A method which creates a random matrix of size data_size x data_size with given singular values.

    Args:
        data_size(int): The input data size n.
        singular_values(RowVector): The singular values to be set for the matrix to create.

    Returns:
        A random size data_size x data_size Matrix awith the given singular values.

    """
    rank: int = len(singular_values)
    U: Matrix = _random_orthonormal_cols(data_size, rank)
    V: Matrix = _random_orthonormal_cols(data_size, rank)
    return MatInSVDForm(U, singular_values, V)


def _get_example_4_data(data_size: int, singular_values: RowVector) -> Matrix:
    """
    A method which creates a data_size x data_size matrix whose singular values are the input values.

    Args:
        data_size(int): The input data size n.
        singular_values(RowVector): The singular values to be set for the matrix to create.

    Returns:
        A data_size x data_size Matrix with the given singular values.

    """
    U: Matrix = np.ascontiguousarray(
        np.stack([
            np.ones(data_size),
            np.tile([1, -1], data_size // 2),
            np.tile([1, 1, -1, -1], data_size // 4),
            np.tile([1, 1, 1, 1, -1, -1, -1, -1], data_size // 8)]).T) / np.sqrt(data_size)
    V: Matrix = np.ascontiguousarray(
        np.stack([
            np.concatenate([np.ones(data_size - 1), [0]]) / np.sqrt(data_size - 1),
            np.concatenate([np.zeros(data_size - 1), [1]]),
            np.concatenate([np.tile([1, -1], (data_size - 2) // 2) / np.sqrt(data_size - 2), [0, 0]]),
            np.concatenate([[1, 0, -1], np.zeros(data_size - 3)]) / np.sqrt(2)]).T)
    return MatInSVDForm(U, np.array(singular_values), V)


def _get_example_5_data(data_size: int, singular_values: RowVector) -> Matrix:
    """
    A method which creates a data_size x data_size matrix with singular values 1 and the other input singular values.

    Args:
        data_size(int): The input data size n.
        singular_value(RowVector): A 1x2 vector of singular values for the created matrix.

    Returns:
        A random size data_size x data_size Matrix with singular values 1 and the other input singular value.

    """
    return ExperimentNo5Form((data_size, data_size), singular_values[0])


# A private dictionary used to create the method "get_data"
_data_type_to_function: Dict[str, Callable] = {
    ExperimentType.ExampleNo1: _get_first_3_examples_data,
    ExperimentType.ExampleNo2: _get_first_3_examples_data,
    ExperimentType.ExampleNo3: _get_first_3_examples_data,
    ExperimentType.ExampleNo4: _get_example_4_data,
    ExperimentType.ExampleNo5: _get_example_5_data
}

# The public method which fetches the data loading methods.
get_data: Callable = create_factory(_data_type_to_function, are_methods=True)