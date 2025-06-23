"""Module for Base Move class"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Self, TypeVar, overload

from quansino.mc.contexts import Context
from quansino.moves.composite import CompositeMove
from quansino.protocols import Operation
from quansino.registry import get_typed_class

if TYPE_CHECKING:
    from collections.abc import Callable

    from ase.cell import Cell

    from quansino.protocols import Move
    from quansino.type_hints import IntegerArray

T = TypeVar("T", bound="BaseMove")


class BaseMove[OperationType: Operation, ContextType: Context]:
    """
    Base class to build Monte Carlo moves. This is a generic base class for all Monte Carlo moves, parameterized by the operation type and context type it works with.

    Parameters
    ----------
    operation : OperationType | None
        The operation to perform, by default None. If None, the default operation for the move will be used.
    apply_constraints : bool, optional
        Whether to apply constraints to the move, by default True.

    Attributes
    ----------
    max_attempts : int
        The maximum number of attempts to make for a successful move, default is 10000.
    operation : OperationType
        The operation to perform in the move.
    apply_constraints : bool
        Whether to apply constraints to the move.
    context : ContextType
        The simulation context attached to this move.
    check_move : Callable[..., bool]
        A callable that returns True if the move should proceed, False otherwise.
    composite_move_type : type[CompositeMove]
        The type of composite move that this move can be combined into.

    Notes
    -----
    This class is a base class for all Monte Carlo moves, and should not be used directly. The __call__ method should be implemented in the subclass, performing the actual move and returning a boolean indicating whether the move was accepted. Classes inheriting from `BaseMove` should implement the `default_operation` property to provide a default operation when None is specified during initialization.
    """

    __slots__ = (
        "apply_constraints",
        "check_move",
        "composite_move_type",
        "context",
        "max_attempts",
        "operation",
    )

    def __init__(
        self, operation: OperationType | None, apply_constraints: bool = True
    ) -> None:
        """Initialize the `BaseMove` object."""
        self.operation: OperationType = operation or self.default_operation
        self.apply_constraints: bool = apply_constraints

        self.composite_move_type: type[CompositeMove] = CompositeMove[
            BaseMove[OperationType, ContextType]
        ]

        self.max_attempts = 10000

        self.check_move: Callable[..., bool] = lambda *_args, **_kwargs: True

    def __call__(self, context: ContextType) -> bool:
        """
        Call the move. This method should be implemented in the subclass, and should return a boolean indicating whether the move was accepted.

        Returns
        -------
        bool
            Whether the move was accepted.
        """
        self.operation.calculate(context)
        return True

    def on_atoms_changed(
        self, added_indices: IntegerArray, removed_indices: IntegerArray
    ) -> None:
        """
        Update the move when atoms are added or removed.

        Parameters
        ----------
        added_indices : IntegerArray
            The indices of the atoms that were added.
        removed_indices : IntegerArray
            The indices of the atoms that were removed.
        """

    def on_cell_changed(self, new_cell: Cell) -> None:
        """
        Update the move when the cell changes.

        Parameters
        ----------
        new_cell : Cell
            The new cell.
        """

    @property
    def default_operation(self) -> OperationType:
        """
        Get the default operation for the move.

        Returns
        -------
        OperationType
            The default operation for the move.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have a default operation defined."
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `BaseMove` object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `BaseMove` object.
        """
        return {
            "name": self.__class__.__name__,
            "kwargs": {
                "operation": self.operation.to_dict(),
                "apply_constraints": self.apply_constraints,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a `BaseMove` object from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the `BaseMove` object.

        Returns
        -------
        Self
            The `BaseMove` object created from the dictionary.
        """
        kwargs = deepcopy(data.get("kwargs", {}))

        if "operation" in kwargs:
            operation_data = kwargs["operation"]

            operation_class: type[Operation] = get_typed_class(
                operation_data["name"], Operation
            )

            kwargs["operation"] = operation_class.from_dict(operation_data)

        instance = cls(**kwargs)

        for key, value in data.get("attributes", {}).items():
            setattr(instance, key, value)

        return instance

    @overload
    def __add__(self: T, other: T) -> CompositeMove[T]: ...

    @overload
    def __add__(self: T, other: CompositeMove[T]) -> CompositeMove[T]: ...

    @overload
    def __add__(self, other: CompositeMove[Move]) -> CompositeMove[Move]: ...

    @overload
    def __add__(self, other: Move) -> CompositeMove[Move]: ...

    def __add__(self, other) -> CompositeMove:
        """
        Add two moves together to create a `CompositeMove`.

        Parameters
        ----------
        other : Move | CompositeMove
            The other displacement move to add.

        Returns
        -------
        CompositeMove
            The composite move.
        """
        if isinstance(other, CompositeMove):
            if type(self.composite_move_type) is type(other):
                return self.composite_move_type([self, *other.moves])
            else:
                return CompositeMove([self, *other.moves])
        elif isinstance(other, BaseMove):
            if self.composite_move_type is other.composite_move_type:
                return other.composite_move_type([self, other])
            else:
                return CompositeMove([self, other])

        raise TypeError(
            f"Cannot add {self.__class__.__name__} to {other.__class__.__name__}"
        )

    def __mul__(self, n: int) -> CompositeMove[Self]:
        """
        Multiply the move by an integer to create a `CompositeMove` with repeated moves.

        Parameters
        ----------
        n : int
            The number of times to repeat the move.

        Returns
        -------
        CompositeMove
            The composite move.
        """
        if n < 1 or not isinstance(n, int):
            raise ValueError(
                "The number of times the move is repeated must be a positive, non-zero integer."
            )

        return self.composite_move_type([self] * n)

    __rmul__ = __mul__

    def __copy__(self) -> Self:
        """
        Create a shallow copy of the move.

        Returns
        -------
        Self
            The shallow copy of the move.
        """
        return self.from_dict(self.to_dict())
