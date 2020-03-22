# -*- coding: utf-8 -*-
"""
enums.py - All enums section
============================

This module contains all possible enums of this project. Most of them are used by the configuration section in
:mod:`main`. An example for using enum: ``ExperimentType.ExampleNo1``

"""

from Infrastructure.utils import BaseEnum


class LogFields(BaseEnum):
    """
    The enum class of fields within experiments logs. Possible values:

    * ``LogFields.DataSize``

    * ``LogFields.ApproximationRank``

    * ``LogFields.Increment``

    * ``LogFields.NextSingularValue``

    * ``LogFields.RandomSVDDuration``

    * ``LogFields.RandomIDDuration``

    * ``LogFields.RandomSVDAccuracy``

    * ``LogFields.RandomIDAccuracy``
    """
    DataSize: str = "Data size"
    ApproximationRank: str = "k"
    Increment: str = "increment"
    NextSingularValue: str = "K+1 singular value"
    RandomSVDDuration: str = "Random SVD Duration in seconds"
    RandomIDDuration: str = "Random ID Duration in seconds"
    RandomSVDAccuracy: str = "Random SVD Accuracy"
    RandomIDAccuracy: str = "Random ID Accuracy"


class ExperimentType(BaseEnum):
    """
    The enum class of experiment types. Possible values:

    * ``ExperimentType.ExampleNo1``

    * ``ExperimentType.ExampleNo2``

    * ``ExperimentType.ExampleNo3``

    * ``ExperimentType.ExampleNo4``

    * ``ExperimentType.ExampleNo5``

    """
    ExampleNo1: str = "Example No. 1"
    ExampleNo2: str = "Example No. 2"
    ExampleNo3: str = "Example No. 3"
    ExampleNo4: str = "Example No. 4"
    ExampleNo5: str = "Example No. 5"
