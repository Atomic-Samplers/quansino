from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast, overload
from warnings import warn

import numpy as np

from quansino.protocols import Operation
from quansino.registry import get_typed_class

if TYPE_CHECKING:
    from quansino.mc.contexts import Context


T = TypeVar("T", bound="Operation")


class CompositeOperation[OperationType: Operation]:
    """
    Class to combine multiple operations into a single operation.

    This class allows for the combination of multiple operations of the same type,
    which are executed sequentially and their results combined. Operations can be
    accessed by index, iterated over, and the class supports addition and
    multiplication operations.

    Parameters
    ----------
    operations : list[OperationType]
        The operations to combine into a single operation. All operations must be
        of the same type specified by the OperationType generic parameter.

    Attributes
    ----------
    operations : list[OperationType]
        The list of operations to be executed.

    Returns
    -------
    Any
        The combined result of all operations, typically the sum of their individual results.
    """

    __slots__ = ("operations",)

    def __init__(self, operations: list[OperationType]) -> None:
        """Initialize the CompositeOperation object."""
        if len(operations) == 0:
            warn(
                f"No operations provided. The {self.__class__.__name__} will not perform any calculations.",
                UserWarning,
                2,
            )

        self.operations = operations

    def calculate(self, context: Context) -> Any:
        """
        Calculate the combined operation to perform on the atoms.

        Parameters
        ----------
        context : ContextType
            The context to use when calculating the operation.

        Returns
        -------
        Displacement
            The combined operation to perform on the atoms.
        """
        return np.sum([op.calculate(context) for op in self.operations], axis=0)

    @overload
    def __add__(self: T, other: CompositeOperation[T]) -> CompositeOperation[T]: ...

    @overload
    def __add__(self, other: CompositeOperation) -> CompositeOperation[Operation]: ...

    @overload
    def __add__(self: T, other: T) -> CompositeOperation[T]: ...

    @overload
    def __add__(self, other) -> CompositeOperation[Operation]: ...

    def __add__(self, other) -> CompositeOperation:
        """
        Combine two operations into a single operation.

        Parameters
        ----------
        other : Operation
            The operation to combine with the current operation.

        Returns
        -------
        Self
            The combined operation of the same type as the caller.

        Notes
        -----
        Works with both single operations and composite operations. If the other operation is a composite operation, the operations are combined into a single composite operation.
        """
        if isinstance(other, CompositeOperation):
            return type(self)(self.operations + other.operations)
        else:
            other = cast("OperationType", other)
            return self.__class__([*self.operations, other])

    def __mul__(self, n: int) -> Self:
        """
        Multiply the displacement move by an integer to create a composite move.

        Parameters
        ----------
        n : int
            The number of times to repeat the move.

        Returns
        -------
        CompositeDisplacementMove
            The composite move.
        """
        if n < 1 or not isinstance(n, int):
            raise ValueError(
                "The number of times the move is repeated must be a positive, non-zero integer."
            )
        return type(self)(self.operations * n)

    def __getitem__(self, index: int) -> OperationType:
        """
        Get the move at the specified index.

        Parameters
        ----------
        index : int
            The index of the move.

        Returns
        -------
        DisplacementMove
            The move at the specified index.
        """
        return self.operations[index]

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self):
        return iter(self.operations)

    __rmul__ = __mul__

    __imul__ = __mul__

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the operation to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            The dictionary representation of the operation.
        """
        return {
            "name": self.__class__.__name__,
            "kwargs": {
                "operations": [operation.to_dict() for operation in self.operations]
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a composite operation from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the operation.

        Returns
        -------
        Self
            The composite operation object created from the dictionary.
        """
        operations = []

        kwargs = deepcopy(data.get("kwargs", {}))

        if "operations" in kwargs:
            for operation_data in kwargs["operations"]:
                operation_class: type[Operation] = get_typed_class(
                    operation_data["name"], Operation
                )
                operation = operation_class.from_dict(operation_data)
                operations.append(operation)

        kwargs["operations"] = operations

        instance = cls(**kwargs)

        for key, value in data.get("attributes", {}).items():
            setattr(instance, key, value)

        return instance
