"""Module for displacement moves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from quansino.mc.contexts import DisplacementContext
from quansino.moves.core import BaseMove
from quansino.operations.displacement import Box

if TYPE_CHECKING:
    from quansino.moves.exchange import ExchangeMove
    from quansino.operations.core import Operation
    from quansino.type_hints import IntegerArray


class DisplacementMove[ContextType: DisplacementContext](BaseMove[ContextType]):
    """
    Class for displacement moves that displaces one atom or a group of atoms. The class will use an [`Operation`][quansino.operations.core.Operation]. The class uses the `labels` attribute to determine which atoms can be displaced, if none, the move fails. If multiple atoms share the same label, they are considered to be part of the same group (molecule) and will be displaced together in a consistent manner.

    Move that displaces multiple labels at once can be created by adding multiple [`DisplacementMove`][quansino.moves.displacement.DisplacementMove] objects together. Similarly, a move can be multiplied by an integer to move multiple labels at once in the same manner. In this case a [`CompositeDisplacementMove`][quansino.moves.displacement.CompositeDisplacementMove] object is returned, which can be used as a normal DisplacementMove object.

    The move only modifies the `moving_indices` attribute of the context object, which might be needed for some operations (Rotation, for example).

    Parameters
    ----------
    moving_labels : IntegerArray
        The labels of the atoms to displace. Atoms with negative labels are not displaced.
    operation : Operation, optional
        The operation to perform in the move, by default None.
    apply_constraints : bool, optional
        Whether to apply constraints to the move, by default True.

    Attributes
    ----------
    is_updatable : Literal[True]
        Whether the move can be updated when atoms are added or removed.
    to_displace_labels : int | None
        The label of the atoms to displace. If None, the move will select the atoms to displace itself. Reset to None after move.
    displaced_labels : int | None
        The label of the atoms that were displaced in the last move. Reset to None after move.
    default_label : int | None
        The default label when adding new atoms. If None, labels for new atoms will be selected automatically.
    labels : IntegerArray
        The labels of the atoms in the simulation.
    unique_labels : IntegerArray
        The unique labels in the simulation (excluding negative labels).

    Important
    ---------
    1. At object creation, `moving_labels` must have the same length as the number of atoms in the simulation.
    2. Conditions for a successful move can be tightened either by subclassing the move and overriding the `check_move` method, or by using the [`MethodType`][types.MethodType] function to replace the method in the instance.
    """

    is_updatable: Literal[True] = True

    def __init__(
        self,
        moving_labels: IntegerArray,
        operation: Operation | None = None,
        apply_constraints: bool = True,
    ) -> None:
        """Initialize the DisplacementMove object.

        Parameters
        ----------
        moving_labels : IntegerArray
            The labels of the atoms to displace. Atoms with negative labels are not displaced.
        operation : Operation, optional
            The operation to perform in the move, by default None.
        apply_constraints : bool, optional
            Whether to apply constraints to the move, by default True.
        """

        self.to_displace_labels: int | None = None
        self.displaced_labels: int | None = None

        self.default_label: int | None = None
        self.set_moving_labels(moving_labels)

        if operation is None:
            operation = Box(0.1)

        super().__init__(operation, apply_constraints)

    def attempt_move(self: DisplacementMove | ExchangeMove) -> bool:
        """
        Attempt to move the atoms using the provided operation and check. The move is attempted `max_attempts` number of times. If the move is successful, return True, otherwise, return False.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        atoms = self.context.atoms
        old_positions = atoms.get_positions()

        for _ in range(self.max_attempts):
            translation = np.full((len(atoms), 3), 0.0)
            translation[self.context.moving_indices] = self.operation.calculate(
                self.context
            )

            atoms.set_positions(
                atoms.positions + translation, apply_constraint=self.apply_constraints
            )

            if self.check_move():
                return True

            atoms.positions = old_positions

        return False

    def __call__(self) -> bool:
        """
        Perform the displacement move. The following steps are performed:

        1. Reset the context, this is done to clear any previous move attributes such as `moving_indices`, which is needed by some specific operations.
        2. Check if the atoms to displace are manually set. If not, select a random label from the available labels, if no labels are available, the move fails.
        3. Find the indices of the atoms to displace and attempt to move them using `attempt_move`. If the move is successful, register a success and return True. Otherwise, register a failure and return False.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        self.context.reset()

        if self.to_displace_labels is None:
            if len(self.unique_moving_labels) == 0:
                return self.register_failure()

            self.to_displace_labels = self.context.rng.choice(self.unique_moving_labels)

        (self.context.moving_indices,) = np.where(
            self.moving_labels == self.to_displace_labels
        )

        if self.attempt_move():
            return self.register_success()
        else:
            return self.register_failure()

    def set_moving_labels(self, new_labels: IntegerArray) -> None:
        """
        Set the labels of the atoms to displace and update the unique labels. This function should always be used to set the labels.

        Parameters
        ----------
        new_labels : IntegerArray
            The new labels of the atoms to displace.
        """
        self.moving_labels: IntegerArray = np.asarray(new_labels)
        self.unique_moving_labels: IntegerArray = np.unique(
            self.moving_labels[self.moving_labels >= 0]
        )

    def register_success(self) -> Literal[True]:
        """
        Register a successful move, saving the current state.

        Returns
        -------
        Literal[True]
            Always returns True.
        """
        self.displaced_labels = self.to_displace_labels
        self.to_displace_labels = None

        return True

    def register_failure(self) -> Literal[False]:
        """
        Register a failed move, reverting any changes made.

        Returns
        -------
        Literal[False]
            Always returns False.
        """
        self.to_displace_labels = None
        self.displaced_labels = None

        return False

    def update(
        self, indices_to_add: IntegerArray, indices_to_remove: IntegerArray
    ) -> None:
        """
        Update the move by resetting the labels and updating the operation.

        Parameters
        ----------
        indices_to_add : IntegerArray
            The indices of the atoms to add.
        indices_to_remove : IntegerArray
            The indices of the atoms to remove.

        Raises
        ------
        ValueError
            If the length of the labels is not equal to the number of atoms.
        """
        if len(indices_to_add):
            label: int = self.default_label or (
                max(self.unique_moving_labels) + 1
                if len(self.unique_moving_labels)
                else 0
            )
            self.set_moving_labels(
                np.hstack((self.moving_labels, np.full(len(indices_to_add), label)))
            )

        if len(indices_to_remove):
            self.set_moving_labels(np.delete(self.moving_labels, indices_to_remove))

        if len(self.moving_labels) != len(self.context.atoms):
            raise ValueError(
                "Updating the labels went wrong, the length of the labels is not equal to the number of atoms."
            )

    def __add__(
        self: DisplacementMove, other: DisplacementMove | CompositeDisplacementMove
    ) -> CompositeDisplacementMove:
        """
        Add two displacement moves together to create a composite move.

        Parameters
        ----------
        other : DisplacementMove | CompositeMove
            The other displacement move to add.

        Returns
        -------
        CompositeMove
            The composite move.
        """
        if isinstance(other, CompositeDisplacementMove):
            return CompositeDisplacementMove([self, *other.moves])
        else:
            return CompositeDisplacementMove([self, other])

    def __mul__(self: DisplacementMove, n: int) -> CompositeDisplacementMove:
        """
        Multiply the displacement move by an integer to create a composite move.

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
        return CompositeDisplacementMove([self] * n)

    __rmul__ = __mul__

    def __copy__(self) -> DisplacementMove:
        """
        Create a shallow copy of the move.

        Returns
        -------
        DisplacementMove
            The shallow copy of the move.
        """
        new_move = DisplacementMove(
            self.moving_labels, self.operation, self.apply_constraints
        )
        new_move.__dict__.update(self.__dict__)

        return new_move

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "moving_labels": self.moving_labels}


class CompositeDisplacementMove[ContextType: DisplacementContext](
    BaseMove[ContextType]
):
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

    is_updatable: Literal[True] = True

    def __init__(self, moves: list[DisplacementMove[ContextType]]) -> None:
        self.moves = moves

        self.displaced_labels: list[int | None] = []
        self.number_of_moved_particles: int = 0

        self.with_replacement = False

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
        self.reset()

        for move in self.moves:
            if move.to_displace_labels is None and self.with_replacement is False:
                filtered_displaced_labels = [
                    atom for atom in self.displaced_labels if atom is not None
                ]
                available_candidates = np.setdiff1d(
                    move.unique_moving_labels,
                    filtered_displaced_labels,
                    assume_unique=True,
                )
                if len(available_candidates) == 0:
                    self.register_failure()
                    continue

                move.to_displace_labels = move.context.rng.choice(available_candidates)

            if move():
                self.register_success(move)
            else:
                self.register_failure()

        return self.number_of_moved_particles > 0

    def register_success(self, move: DisplacementMove) -> None:
        """Register a successful move, saving the current state."""
        self.displaced_labels.append(move.displaced_labels)
        self.number_of_moved_particles += 1

    def register_failure(self) -> None:
        """Register a failed move, reverting any changes made."""
        self.displaced_labels.append(None)

    def reset(self) -> None:
        self.displaced_labels = []
        self.number_of_moved_particles = 0

    def __add__(
        self, other: DisplacementMove | CompositeDisplacementMove
    ) -> CompositeDisplacementMove:
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
        if isinstance(other, CompositeDisplacementMove):
            return type(self)(self.moves + other.moves)
        else:
            return type(self)([*self.moves, other])

    def attach_simulation(self, context: ContextType) -> None:
        for move in self.moves:
            move.attach_simulation(context)

    def update(
        self, indices_to_add: IntegerArray, indices_to_remove: IntegerArray
    ) -> None:
        """
        Update the move by resetting the labels and updating the operation.

        Parameters
        ----------
        indices_to_add : IntegerArray
            The indices of the atoms to add.
        indices_to_remove : IntegerArray
            The indices of the atoms to remove.

        Raises
        ------
        ValueError
            If the length of the labels is not equal to the number of atoms.
        """
        for move in self.moves:
            move.update(indices_to_add, indices_to_remove)

    def __mul__(self, n: int) -> CompositeDisplacementMove:
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

    def __getitem__(self, index: int) -> DisplacementMove:
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

    def to_dict(self):
        return {**super().to_dict(), "moves": [move.to_dict() for move in self.moves]}
