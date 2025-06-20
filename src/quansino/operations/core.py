from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, TypeVar, overload

from quansino.operations.composite import CompositeOperation

if TYPE_CHECKING:
    from quansino.mc.contexts import Context
    from quansino.operations.core import BaseOperation
    from quansino.protocols import Operation


T = TypeVar("T", bound="Operation")


class BaseOperation:
    """
    Abstract base class for operations in Monte Carlo simulations.

    This class defines the interface for all operations that can be performed
    during Monte Carlo moves. Implementations must provide a `calculate` method
    that computes the operation based on the given context.

    Operations can be combined using the `+` operator or multiplied using the `*` operator
    to create composite operations.
    """

    __slots__ = ()

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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the operation to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            The dictionary representation of the operation.
        """
        return {"name": self.__class__.__name__}

    @overload
    def __add__(self: T, other: CompositeOperation[T]) -> CompositeOperation[T]: ...

    @overload
    def __add__(self: T, other: T) -> CompositeOperation[T]: ...

    @overload
    def __add__(self, other) -> CompositeOperation[BaseOperation]: ...

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
        Multiply the operation by an integer to create a composite operation.

        Parameters
        ----------
        n : int
            The number of times to repeat the operation.

        Returns
        -------
        CompositeOperation
            The composite operation.
        """
        if n < 1 or not isinstance(n, int):
            raise ValueError(
                "The number of times the move is repeated must be a positive, non-zero integer."
            )
        return CompositeOperation([self] * n)

    __rmul__ = __mul__

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
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
        kwargs = data.get("kwargs", {})
        instance = cls(**kwargs)

        for key, value in data.get("attributes", {}).items():
            setattr(instance, key, value)

        return instance
