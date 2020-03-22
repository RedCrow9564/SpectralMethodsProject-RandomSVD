# -*- coding: utf-8 -*-
""" The data management module

This module handles the fetching of the data from the local resources path, given in the configuration and arranging it
for our purposes of estimations. See the example for fetching the data for Example no. 1.

Example:
    get_data(ExperimentType.ExampleNo1) - Creating the data for Example no. 1 of the paper.

"""

from scipy.stats import ortho_group
import numpy as np
from Infrastructure.enums import ExperimentType
from Infrastructure.utils import Dict, RowVector, Matrix, create_factory, Callable


def _random_orthonormal_cols(data_size: int, columns: int) -> Matrix:
    return ortho_group.rvs(data_size, size=1)[:, :columns]


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
    VT: Matrix = _random_orthonormal_cols(data_size, rank).T
    return U.dot(np.diag(singular_values).dot(VT))


def _get_example_4_data(data_size: int, singular_values: RowVector) -> Matrix:
    """
    A method which creates a data_size x data_size matrix whose singular values are the input values.

    Args:
        data_size(int): The input data size n.
        singular_values(RowVector): The singular values to be set for the matrix to create.

    Returns:
        A data_size x data_size Matrix with the given singular values.

    """
    U: Matrix = np.stack([np.ones(data_size),
                          np.tile([1, -1], data_size // 2),
                          np.tile([1, 1, -1, -1], data_size // 4),
                          np.tile([1, 1, 1, 1, -1, -1, -1, -1], data_size // 8)]).T / np.sqrt(data_size)
    VT: Matrix = np.stack([np.concatenate([np.ones(data_size - 1), [0]]) / np.sqrt(data_size - 1),
                           np.concatenate([np.zeros(data_size - 1), [1]]),
                           np.concatenate([np.tile([1, -1], (data_size - 2) // 2) / np.sqrt(data_size - 2), [0, 0]]),
                           np.concatenate([[1, 0, -1], np.zeros(data_size - 3)]) / np.sqrt(2)])
    return U.dot(np.diag(singular_values).dot(VT))


def _get_example_5_data(data_size: int, singular_values: RowVector) -> Matrix:
    """
    A method which creates a data_size x data_size matrix with singular values 1 and the other input singular values.

    Args:
        data_size(int): The input data size n.
        singular_value(RowVector): A 1x2 vector of singular values for the created matrix.

    Returns:
        A random size data_size x data_size Matrix with singular values 1 and the other input singular value.

    """
    data: Matrix = singular_values[1] * np.eye(data_size)
    data[0, :] += singular_values[0] * np.ones(data_size) / np.sqrt(data_size)
    return data


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
