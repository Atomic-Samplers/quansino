"""Module for Base Move class"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Self, overload

from quansino.protocols import Move
from quansino.registry import get_typed_class

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ase.cell import Cell

    from quansino.mc.contexts import Context
    from quansino.type_hints import IntegerArray


class CompositeMove[MoveType: Move]:
    """
    Class to perform a composite move operation. This class is returned when adding or multiplying `Move` objects together.

    Parameters
    ----------
    moves : list[MoveType]
        The moves to perform in the composite move.

    Attributes
    ----------
    moves : list[MoveType]
        The moves to perform in the composite move.
    """

    __slots__ = ("moves",)

    def __init__(self, moves: list[MoveType]) -> None:
        """Initialize the `CompositeMove` object."""
        self.moves = moves

    def __call__(self, context: Context, *_args: Any, **_kwargs: Any) -> bool:
        """
        Perform the composite move by executing all constituent moves.

        Parameters
        ----------
        context : Context
            The context in which the move is performed, containing information about the system state.
        *_args : Any
            Additional positional arguments, not used in this method.
        **_kwargs : Any
            Additional keyword arguments, not used in this method.

        Returns
        -------
        bool
            Whether any of the constituent moves was successful.
        """
        return any([move(context) for move in self.moves])  # noqa: C419

    @overload
    def __add__(
        self: CompositeMove[MoveType], other: CompositeMove[MoveType]
    ) -> CompositeMove[MoveType]: ...

    @overload
    def __add__(
        self: CompositeMove[MoveType], other: MoveType
    ) -> CompositeMove[MoveType]: ...

    @overload
    def __add__(self, other: CompositeMove) -> CompositeMove[Move]: ...

    @overload
    def __add__(self, other: Move) -> CompositeMove[Move]: ...

    def __add__(self, other) -> CompositeMove:
        """
        Add two moves together to create a `CompositeMove` object.

        Parameters
        ----------
        other : CompositeMove | Move
            The other move to add.

        Returns
        -------
        CompositeMove
            The composite move.
        """
        from quansino.moves.core import BaseMove

        if isinstance(other, CompositeMove):
            if type(self) is type(other):
                return type(self)(self.moves + other.moves)

            return CompositeMove(self.moves + other.moves)
        elif isinstance(other, BaseMove):
            if type(self) is other.composite_move_type:
                return other.composite_move_type([*self.moves, other])
            else:
                return CompositeMove([*self.moves, other])

        raise TypeError(
            f"Cannot add {self.__class__.__name__} to {other.__class__.__name__}"
        )

    def on_atoms_changed(
        self, added_indices: IntegerArray, removed_indices: IntegerArray
    ) -> None:
        """
        Update the move by resetting the labels and updating the operation.

        Parameters
        ----------
        added_indices : IntegerArray
            The indices of the atoms to add.
        removed_indices : IntegerArray
            The indices of the atoms to remove.
        """
        for move in self.moves:
            move.on_atoms_changed(added_indices, removed_indices)

    def on_cell_changed(self, new_cell: Cell) -> None:
        """
        Update the move when the cell changes.

        Parameters
        ----------
        new_cell : Cell
            The new cell.
        """
        for move in self.moves:
            move.on_cell_changed(new_cell)

    def __mul__(self, n: int) -> Self:
        """
        Multiply the move by an integer to create a `CompositeMove` with repeated moves.

        Parameters
        ----------
        n : int
            The number of times to repeat the move.

        Returns
        -------
        CompositeMove
            The composite move with repeated moves.
        """
        if not isinstance(n, int):
            raise TypeError(
                f"The number of times the move is repeated must be a positive, non-zero integer. Got {type(n)}."
            )
        if n < 1:
            raise ValueError(
                "The number of times the move is repeated must be a positive, non-zero integer."
            )

        return type(self)(self.moves * n)

    def __getitem__(self, index: int) -> MoveType:
        """
        Get the move at the specified index.

        Parameters
        ----------
        index : int
            The index of the move.

        Returns
        -------
        MoveType
            The move at the specified index.
        """
        return self.moves[index]

    def __len__(self) -> int:
        """
        Get the number of moves in the composite move.

        Returns
        -------
        int
            The number of moves in the composite move.
        """
        return len(self.moves)

    def __iter__(self) -> Iterator[MoveType]:
        """
        Iterate over the moves in the composite move.

        Returns
        -------
        Iterator[MoveType]
            An iterator over the moves in the composite move.
        """
        return iter(self.moves)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `CompositeMove` object to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `CompositeMove` object.
        """
        return {
            "name": self.__class__.__name__,
            "kwargs": {"moves": [move.to_dict() for move in self.moves]},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a `CompositeMove` from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the composite move.

        Returns
        -------
        Self
            The `CompositeMove` object created from the dictionary.
        """
        moves = []
        kwargs = deepcopy(data.get("kwargs", {}))

        if "moves" in kwargs:
            for move_data in kwargs["moves"]:
                move_class: type[Move] = get_typed_class(move_data["name"], Move)
                move = move_class.from_dict(move_data)
                moves.append(move)

        kwargs["moves"] = moves

        instance = cls(**kwargs)

        for key, value in data.get("attributes", {}).items():
            setattr(instance, key, value)

        return instance
