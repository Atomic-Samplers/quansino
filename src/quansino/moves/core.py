"""Module for Base Move class"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Self

from quansino.mc.contexts import Context
from quansino.operations.core import Operation
from quansino.registry import get_typed_class


class BaseMove[ContextType: Context]:
    """
    Helper Class to build Monte Carlo moves.

    Parameters
    ----------
    operation: Operation
        The operation to perform in the move. The object must have a `calculate` method that takes a context as input.
    apply_constraints: bool, optional
        Whether to apply constraints to the move, by default True.

    Attributes
    ----------
    AcceptableContext : Context
        The required context class for this move.
    max_attempts : int
        The maximum number of attempts to make for a successful move, default is 10000.
    operation: Operation
        The operation to perform in the move.
    apply_constraints: bool
        Whether to apply constraints to the move.
    context: ContextType
        The simulation context attached to this move.

    Notes
    -----
    This class is a base class for all Monte Carlo moves, and should not be used directly.
    The __call__ method should be implemented in the subclass, performing the actual move
    and returning a boolean indicating whether the move was accepted.
    """

    AcceptableContext = Context
    max_attempts: int = 10000

    def __init__(self, operation: Operation, apply_constraints: bool = True) -> None:
        """Initialize the BaseMove object."""
        self.operation = operation
        self.apply_constraints: bool = apply_constraints

    def attach_simulation(self, context: ContextType) -> None:
        """
        Attach the simulation context to the move. This method must be called before the move is used, and should be used to set the context attribute. This must be done by the Monte Carlo classes.

        Parameters
        ----------
        context: Context
            The simulation context to attach to the move.

        Notes
        -----
        The context object aim to provide the necessary information for the move to perform its operation, without having to pass whole objects around. Classes inheriting from BaseMove should define a `AcceptableContext` attribute that specifies the context class that the move requires.
        """
        self.context: ContextType = context

    def check_move(self) -> bool:
        """Check if the move is accepted. This method should be implemented in the subclass, and should return a boolean indicating whether the move was accepted."""
        return True

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the move to a dictionary.

        Returns
        -------
        dict[str, str | bool]
            A dictionary representation of the move.
        """
        return {
            "name": self.__class__.__name__,
            "kwargs": {
                "operation": self.operation.to_dict(),
                "apply_constraints": self.apply_constraints,
            },
        }

    def __call__(self) -> bool:
        """
        Call the move. This method should be implemented in the subclass, and should return a boolean indicating whether the move was accepted.

        Returns
        -------
        bool
            Whether the move was accepted.
        """
        if self.check_move():
            self.operation.calculate(self.context)
            return True
        else:
            return False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a move from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the move.

        Returns
        -------
        Self
            The move created from the dictionary.
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


class CompositeMove[MoveType: BaseMove]:
    """
    Class to perform a composite displacement operation on atoms. This class is returned when adding or multiplying [`DisplacementMove`][quansino.moves.displacement.DisplacementMove] objects together.

    Parameters
    ----------
    moves : list[DisplacementMove]
        The moves to perform in the composite move.

    Attributes
    ----------
    is_updatable : Literal[True]
        Whether the move can be updated when atoms are added or removed.
    moves : list[DisplacementMove]
        The moves to perform in the composite move.
    displaced_labels : list[int | None]
        The labels of the atoms that were displaced in the last move.
    number_of_moved_particles : int
        The number of particles that were moved in the last move.
    with_replacement : bool
        Whether to allow the same label to be displaced multiple times in a single move.
    """

    def __init__(self, moves: list[MoveType]) -> None:
        self.moves = moves

    def __call__(self) -> bool:
        """
        Perform the displacement move. The following steps are performed:

        1. If no candidates are available, return False and does not register a move.
        2. Check if there are enough candidates to displace. If yes, select `displacements_per_move` number of candidates from the available candidates, if not, select the maximum number of candidates available.
        3. If `to_displace_labels` is None, select `displacements_per_move` candidates from the available candidates.
        4. Attempt to move each candidate using `attempt_move`. If any of the moves is successful, register a success and return True. Otherwise, register a failure and return False.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        return any(move() for move in self.moves)

    def __add__(self, other: CompositeMove[MoveType] | MoveType) -> Self:
        """
        Add two displacement moves together to create a composite move.

        Parameters
        ----------
        other : DisplacementMove
            The other displacement move to add.

        Returns
        -------
        CompositeDisplacementMove
            The composite move.
        """
        if isinstance(other, CompositeMove):
            return type(self)(self.moves + other.moves)
        else:
            return type(self)([*self.moves, other])

    def attach_simulation(self, context: Context) -> None:
        for move in self.moves:
            move.attach_simulation(context)

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
        DisplacementMove
            The move at the specified index.
        """
        return self.moves[index]

    def __len__(self) -> int:
        return len(self.moves)

    def __iter__(self):
        return iter(self.moves)

    __rmul__ = __mul__

    __imul__ = __mul__

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "kwargs": {"moves": [move.to_dict() for move in self.moves]},
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
        CompositeOperation
            The composite operation object created from the dictionary.
        """
        moves = []
        kwargs = deepcopy(data.get("kwargs", {}))

        if "moves" in kwargs:
            for move_data in kwargs["moves"]:
                move_class: type[BaseMove] = get_typed_class(
                    move_data["name"], BaseMove
                )
                move = move_class.from_dict(move_data)
                moves.append(move)

        kwargs["moves"] = moves

        instance = cls(**kwargs)

        for key, value in data.get("attributes", {}).items():
            setattr(instance, key, value)

        return instance
