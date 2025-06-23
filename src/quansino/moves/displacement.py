"""Module for displacement moves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from quansino.mc.contexts import DisplacementContext
from quansino.moves.composite import CompositeMove
from quansino.moves.core import BaseMove
from quansino.operations.displacement import Ball
from quansino.protocols import Operation

if TYPE_CHECKING:
    from quansino.type_hints import IntegerArray


class DisplacementMove[OperationType: Operation, ContextType: DisplacementContext](
    BaseMove[OperationType, ContextType]
):
    """
    Class for displacement moves that displaces one atom or a group of atoms. The class will use an [`Operation`][quansino.operations.core.Operation]. The class uses the `labels` attribute to determine which atoms can be displaced, if none, the move fails. If multiple atoms share the same label, they are considered to be part of the same group (molecule) and will be displaced together in a consistent manner.

    Move that displaces multiple labels at once can be created by adding multiple [`DisplacementMove`][quansino.moves.displacement.DisplacementMove] objects together. Similarly, a move can be multiplied by an integer to move multiple labels at once in the same manner. In this case a [`CompositeDisplacementMove`][quansino.moves.displacement.CompositeDisplacementMove] object is returned, which can be used as a normal DisplacementMove object.

    The move only modifies the `moving_indices` attribute of the context object, which might be needed for some operations (Rotation, for example).

    Parameters
    ----------
    labels : IntegerArray
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
    At object creation, `labels` must have the same length as the number of atoms in the simulation.
    """

    __slots__ = (
        "default_label",
        "displaced_labels",
        "labels",
        "to_displace_labels",
        "unique_labels",
    )

    def __init__(
        self,
        labels: IntegerArray,
        operation: OperationType | None = None,
        apply_constraints: bool = True,
    ) -> None:
        """Initialize the DisplacementMove object.

        Parameters
        ----------
        labels : IntegerArray
            The labels of the atoms to displace. Atoms with negative labels are not displaced.
        operation : Operation, optional
            The operation to perform in the move, by default None.
        apply_constraints : bool, optional
            Whether to apply constraints to the move, by default True.
        """
        self.to_displace_labels: int | None = None
        self.displaced_labels: int | None = None

        self.default_label: int | None = None
        self.set_labels(labels)

        super().__init__(operation, apply_constraints)

        self.composite_move_type = CompositeDisplacementMove

    def attempt_displacement(self, context: ContextType) -> bool:
        """
        Attempt to move the atoms using the provided operation and check. The move is attempted `max_attempts` number of times. If the move is successful, return True, otherwise, return False.

        Parameters
        ----------
        context : ContextType
            The context for the move.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        atoms = context.atoms
        old_positions = atoms.get_positions()

        for _ in range(self.max_attempts):
            translation = np.full((len(atoms), 3), 0.0)
            translation[context._moving_indices] = self.operation.calculate(context)

            atoms.set_positions(
                atoms.positions + translation, apply_constraint=self.apply_constraints
            )

            if self.check_move(context):
                return True

            atoms.positions = old_positions

        return False

    def __call__(self, context: ContextType) -> bool:
        """
        Perform the displacement move. The following steps are performed:

        1. Check if the atoms to displace are manually set. If not, select a random label from the available labels, if no labels are available, the move fails.
        2. Find the indices of the atoms to displace and attempt to move them using `attempt_displacement`. If the move is successful, register a success and return True. Otherwise, register a failure and return False.

        Parameters
        ----------
        context : ContextType
            The context for the move.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        if self.to_displace_labels is None:
            if len(self.unique_labels) == 0:
                return self.register_failure()

            self.to_displace_labels = context.rng.choice(self.unique_labels)

        (context._moving_indices,) = np.where(self.labels == self.to_displace_labels)

        if self.attempt_displacement(context):
            return self.register_success()
        else:
            return self.register_failure()

    def set_labels(self, new_labels: IntegerArray) -> None:
        """
        Set the labels of the atoms to displace and update the unique labels. This function should always be used to set the labels.

        Parameters
        ----------
        new_labels : IntegerArray
            The new labels of the atoms to displace.
        """
        self.labels: IntegerArray = np.asarray(new_labels)
        self.unique_labels: IntegerArray = np.unique(self.labels[self.labels >= 0])

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

    @property
    def default_operation(self) -> Operation:
        """
        Get the default operation for the move.

        Returns
        -------
        OperationType
            The default operation for the move.
        """
        return Ball(0.1)

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

        Raises
        ------
        ValueError
            If the length of the labels is not equal to the number of atoms.
        """
        if len(added_indices):
            label: int = self.default_label or (
                np.max(self.unique_labels) + 1 if len(self.unique_labels) else 0
            )
            self.set_labels(
                np.hstack((self.labels, np.full(len(added_indices), label)))
            )

        if len(removed_indices):
            self.set_labels(np.delete(self.labels, removed_indices))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `DisplacementMove` object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `DisplacementMove` object.
        """
        dictionary = super().to_dict()
        dictionary.setdefault("kwargs", {})["labels"] = self.labels
        dictionary.setdefault("attributes", {})["default_label"] = self.default_label

        return dictionary


class CompositeDisplacementMove(CompositeMove[DisplacementMove]):
    """
    Class to perform a composite displacement operation on atoms. This class is returned when adding or multiplying [`DisplacementMove`][quansino.moves.displacement.DisplacementMove] objects together.

    Parameters
    ----------
    moves : list[DisplacementMove]
        The moves to perform in the composite move.

    Attributes
    ----------
    moves : list[DisplacementMove]
        The moves to perform in the composite move.
    displaced_labels : list[int | None]
        The labels of the atoms that were displaced in the last move.
    number_of_moved_particles : int
        The number of particles that were moved in the last move.
    """

    __slots__ = ("displaced_labels",)

    def __init__(self, moves: list[DisplacementMove]) -> None:
        super().__init__(moves)

        self.displaced_labels: list[int | None] = []

    def __call__(self, context: DisplacementContext) -> bool:
        """
        Perform the composite displacement move. The following steps are performed:

        1. Reset the displaced_labels list to prepare for the new move.
        2. For each move in the composite, find available candidates that haven't been displaced yet in this composite move.
        3. If no candidates are available for a move, register a failure for that move and continue.
        4. Select a random candidate from the available labels and attempt the displacement.
        5. Register success or failure for each individual move.
        6. Return True if at least one particle was moved, False otherwise.

        Parameters
        ----------
        context : DisplacementContext
            The context for the move.

        Returns
        -------
        bool
            Whether at least one move in the composite was successful.
        """
        self.reset()

        for move in self.moves:
            filtered_displaced_labels = [
                atom for atom in self.displaced_labels if atom is not None
            ]
            available_candidates = np.setdiff1d(
                move.unique_labels, filtered_displaced_labels, assume_unique=True
            )
            if len(available_candidates) == 0:
                self.register_failure()
                continue

            move.to_displace_labels = context.rng.choice(available_candidates)

            if move(context):
                self.register_success(move)
            else:
                self.register_failure()

        return self.number_of_moved_particles > 0

    def register_success(self, move: DisplacementMove) -> None:
        """Register a successful move, saving the current state."""
        self.displaced_labels.append(move.displaced_labels)

    def register_failure(self) -> None:
        """Register a failed move, reverting any changes made."""
        self.displaced_labels.append(None)

    def reset(self) -> None:
        self.displaced_labels = []

    @property
    def number_of_moved_particles(self) -> int:
        """
        The number of particles that were moved in the last move.

        Returns
        -------
        int
            The number of particles that were moved.
        """
        return sum(True for label in self.displaced_labels if label is not None)
