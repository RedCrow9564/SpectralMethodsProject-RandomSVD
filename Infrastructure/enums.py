# -*- coding: utf-8 -*-
""" All enums section

This module contains all possible enums of this project. Most of them are used by the configuration section in main.py.
See the following example on using an enum.

Example
-------
    a = ExperimentType.ExampleNo1

"""

from typing import Iterator, List
import inspect


class _MetaEnum(type):
    """
    A private meta-class which given any BaseEnum object to be an iterable.
    This can be used for iterating all possible values of this enum. Should not be used explicitly.
    """
    def __iter__(self) -> Iterator:
        """
        This method gives any BaseEnum the ability of iterating over all the enum's values.

        Returns:
        --------
            An iterator for the collection of all the enum's values.

        """
        # noinspection PyUnresolvedReferences
        return self.enum_iter()

    def __contains__(self, item) -> bool:
        """
        This method give any BaseEnum the ability to test if a given item is a possible value for this enum class.

        Returns:
        --------
            A flag which indicates if 'item' is a possible value for this enum class.

        """
        # noinspection PyUnresolvedReferences
        return self.enum_contains(item)


class BaseEnum(metaclass=_MetaEnum):
    """
    A basic interface for all enum classes. Should be sub-classed in eny enum.

    Example:
    -------
        class AlgorithmsType(BaseEnum)

    """

    @classmethod
    def enum_iter(cls) -> Iterator:
        """
        This method gives any BaseEnum the ability of iterating over all the enum's values.

        Returns:
        --------
            An iterator for the collection of all the enum's values.

        """
        return iter(cls.get_all_values())

    @classmethod
    def enum_contains(cls, item) -> bool:
        """
        This method give any BaseEnum the ability to test if a given item is a possible value for this enum class.

        Returns:
        --------
                A flag which indicates if 'item' is a possible value for this enum class.

        """
        return item in cls.get_all_values()

    @classmethod
    def get_all_values(cls) -> List:
        """
        A method which fetches all possible values of an enum. Used for iterating over an enum.

        Returns:
        --------
            A list of all possible enum's values.

        """
        all_attributes: List = inspect.getmembers(cls, lambda a: not inspect.ismethod(a))
        all_attributes = [value for name, value in all_attributes if not (name.startswith('__') or name.endswith('__'))]
        return all_attributes


class LogFields(BaseEnum):
    """
    The enum class of fields within experiments logs.
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
    The enum class of experiment types.
    """
    ExampleNo1: str = "Example No. 1"
    ExampleNo2: str = "Example No. 2"
    ExampleNo3: str = "Example No. 3"
    ExampleNo4: str = "Example No. 4"
    ExampleNo5: str = "Example No. 5"
