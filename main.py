#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
main.py - The main module of the project
========================================

This module contains the config for the experiment in the "config" function.
Running this module invokes the :func:`main` function, which then performs the experiment and saves its results
to the configured results folder. Example for running an experiment: ``python main.py``

"""
import numpy as np
from numpy.linalg import multi_dot
from Infrastructure.utils import ex, DataLog, List, Callable, Scalar, RowVector, Matrix, measure_time
from Infrastructure.enums import LogFields, ExperimentType
from data_loader import get_data
from randomized_decompositions import random_svd, random_id


def choose_singular_values(experiment_type: ExperimentType) -> RowVector:
    """

    This function sets the needed singular values, according to the given experiment_type

    Args:
        experiment_type(ExperimentType): The performed experiment. For example, ``ExperimentType.ExampleNo1``.

    Returns:
        A RowVector of the required singular values.

    """
    if experiment_type == ExperimentType.ExampleNo1:
        return np.concatenate([np.flip(np.geomspace(0.2e-15, 1, num=10)), 0.2e-15 * np.ones(10)])
    elif experiment_type == ExperimentType.ExampleNo2:
        return np.concatenate([np.flip(np.geomspace(1e-8, 1, num=10)), 1e-8 * np.ones(10)])
    elif experiment_type == ExperimentType.ExampleNo3:
        return np.concatenate([np.flip(np.geomspace(1e-9, 1, num=30)), 1e-9 * np.ones(30)])
    elif experiment_type == ExperimentType.ExampleNo4:
        return [1, 1, 1e-8, 1e-8]
    elif experiment_type == ExperimentType.ExampleNo5:
        return [1, 1e-17]
    return [-1]


def choose_increments(experiment_type: ExperimentType) -> List:
    """

    This function sets the needed increments, according to the given experiment_type

    Args:
        experiment_type(ExperimentType): The performed experiment. For example, ``ExperimentType.ExampleNo1``.

    Returns:
        A list of the required increments.

    """
    if experiment_type in [ExperimentType.ExampleNo1, ExperimentType.ExampleNo2,
                           ExperimentType.ExampleNo4, ExperimentType.ExampleNo5]:
        return [0]
    elif experiment_type == ExperimentType.ExampleNo3:
        return [0] + np.round(np.geomspace(2, 16, 4)).astype(int).tolist()
    return [-1]


def choose_approximation_ranks(experiment_type: ExperimentType) -> List:
    """

    This function sets the needed approximation ranks, according to the given experiment_type

    Args:
        experiment_type(ExperimentType): The performed experiment. For example, ``ExperimentType.ExampleNo1``.

    Returns:
        A list of the required approximation ranks.

    """
    if experiment_type in [ExperimentType.ExampleNo1, ExperimentType.ExampleNo2, ExperimentType.ExampleNo5]:
        return [10]
    elif experiment_type == ExperimentType.ExampleNo3:
        return [30]
    elif experiment_type == ExperimentType.ExampleNo4:
        return [2]
    return [-1]


def choose_data_sizes(experiment_type: ExperimentType) -> List:
    """

    This function sets the needed data sizes, according to the given experiment_type

    Args:
        experiment_type(ExperimentType): The performed experiment. For example, ``ExperimentType.ExampleNo1``.

    Returns:
        A list of the required data sizes.

    """
    if experiment_type in [ExperimentType.ExampleNo1, ExperimentType.ExampleNo2, ExperimentType.ExampleNo5]:
        return np.geomspace(1e+2, 1e+3, 2, dtype=int).tolist()
    elif experiment_type == ExperimentType.ExampleNo3:
        return [1e+5]
    elif experiment_type == ExperimentType.ExampleNo4:
        return (4 * np.geomspace(1e+2, 1e+3, 2, dtype=int)).tolist()
    return [-1]


@ex.config
def config():
    """ Config section

    This function contains all possible configuration for all experiments. Full details on each configuration values
    can be found in :mod:`enums.py`.
    """

    experiment_type: str = ExperimentType.ExampleNo1
    singular_values: RowVector = choose_singular_values(experiment_type)
    used_data_factory: Callable = get_data(experiment_type)
    data_sizes: List = choose_data_sizes(experiment_type)
    approximation_ranks: List = choose_approximation_ranks(experiment_type)
    increments: List = choose_increments(experiment_type)
    results_path: str = r'Results/'


@ex.automain
def main(data_sizes: List, approximation_ranks: List, increments: List, singular_values: RowVector,
         used_data_factory: Callable, results_path: str, experiment_type: str) -> None:
    """ The main function of this project

    This functions performs the desired experiment according to the given configuration.
    The function runs the random_svd and random_id for every combination of data_size, approximation rank and increment
    given in the config and saves all the results to a csv file in the results folder (given in the configuration).
    """
    results_log = DataLog(LogFields)  # Initializing an empty results log.
    random_svd_with_run_time: Callable = measure_time(random_svd)
    random_id_with_run_time: Callable = measure_time(random_id)

    for data_size in data_sizes:
        data_matrix: Matrix = used_data_factory(data_size, singular_values)

        for approximation_rank in approximation_ranks:
            next_singular_value: Scalar = singular_values[approximation_rank + 1] if \
                approximation_rank < len(singular_values) else singular_values[-1]

            for increment in increments:
                # Executing all the tested methods.
                print(f'n={data_size}, k={approximation_rank}, l={approximation_rank + increment}')
                U, sigma, VT, svd_duration = random_svd_with_run_time(data_matrix, approximation_rank, increment)
                random_svd_accuracy: Scalar = np.linalg.norm(data_matrix - multi_dot([U, np.diag(sigma), VT]))
                print(f'runtime={svd_duration}, accuracy={random_svd_accuracy}')
                B, P, id_duration = random_id_with_run_time(data_matrix, approximation_rank, increment)
                random_id_accuracy: Scalar = np.linalg.norm(data_matrix - np.dot(B, P))
                print(f'runtime={id_duration}, accuracy={random_id_accuracy}')

                # Appending all the experiment results to the log.
                results_log.append(LogFields.DataSize, data_size)
                results_log.append(LogFields.ApproximationRank, approximation_rank)
                results_log.append(LogFields.Increment, increment + approximation_rank)
                results_log.append(LogFields.NextSingularValue, next_singular_value)
                results_log.append(LogFields.RandomSVDAccuracy, random_svd_accuracy)
                results_log.append(LogFields.RandomIDAccuracy, random_id_accuracy)
                results_log.append(LogFields.RandomSVDDuration, svd_duration)
                results_log.append(LogFields.RandomIDDuration, id_duration)

    results_log.save_log(experiment_type + " results", results_folder_path=results_path)
