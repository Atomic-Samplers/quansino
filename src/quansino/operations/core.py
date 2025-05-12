from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quansino.mc.contexts import Context


class Operation(ABC):
    """
    Abstract base class for operations in Monte Carlo simulations.

    This class defines the interface for all operations that can be performed
    during Monte Carlo moves. Implementations must provide a `calculate` method
    that computes the operation based on the given context.

    Operations can be combined using the `+` operator or multiplied using the `*` operator
    to create composite operations.
    """

    @abstractmethod
    def calculate(self, context: Context, *args, **kwargs) -> Any:
        """
        Calculate the operation to perform based on the given context.

        Parameters
        ----------
        context : Context
            The context to use when calculating the operation.
        *args, **kwargs
            Additional arguments for the operation.

        Returns
        -------
        Any
            The result of the calculation, typically a displacement or strain tensor.
        """
        ...

    def __add__(self, other: Operation) -> CompositeOperation:
        """
        Combine two operations into a single operation.

        Parameters
        ----------
        other : Operation
            The operation to combine with the current operation.

        Returns
        -------
        CompositeOperation
            The combined operation.

        Notes
        -----
        Works with both single operations and composite operations. If the other operation is a composite operation, the operations are combined into a single composite operation.
        """
        if isinstance(other, CompositeOperation):
            return CompositeOperation([self, *other.operations])
        else:
            return CompositeOperation([self, other])

    def __mul__(self, n: int) -> CompositeOperation:
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
        return CompositeOperation([self] * n)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the operation to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            The dictionary representation of the operation.
        """
        return {"name": self.__class__.__name__}

    @staticmethod
    def from_dict(
        data: dict[str, Any], operations_registry: None | dict[str, Any] = None
    ) -> Operation:
        """
        Create an operation from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the operation.

        Returns
        -------
        Operation
            The operation object created from the dictionary.
        """
        if operations_registry is None:
            from quansino.operations import operations_registry

        return operations_registry[data["name"]](**data.get("kwargs", {}))


class CompositeOperation(Operation):
    """
    Class to combine multiple operations into a single operation.

    This class allows for the combination of multiple operations, which are executed
    sequentially and their results combined. Operations can be accessed by index,
    iterated over, and the class supports addition and multiplication operations.

    Parameters
    ----------
    operations : list[Operation]
        The operations to combine into a single operation.

    Attributes
    ----------
    operations : list[Operation]
        The list of operations to be executed.

    Returns
    -------
    Any
        The combined result of all operations, typically the sum of their individual results.
    """

    def __init__(self, operations: list[Operation]) -> None:
        """Initialize the CompositeOperation object."""
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

    def __add__(self, other: Operation) -> CompositeOperation:
        """
        Combine two operations into a single operation.

        Parameters
        ----------
        other : Operation
            The operation to combine with the current operation.

        Returns
        -------
        CompositeOperation
            The combined operation.

        Notes
        -----
        Works with both single operations and composite operations. If the other operation is a composite operation, the operations are combined into a single composite operation.
        """
        if isinstance(other, CompositeOperation):
            return CompositeOperation(self.operations + other.operations)
        else:
            return CompositeOperation([*self.operations, other])

    def __mul__(self, n: int) -> CompositeOperation:
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

    def __getitem__(self, index: int) -> Operation:
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
            "operations": [operation.to_dict() for operation in self.operations],
        }
