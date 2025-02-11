"""Module for displacement moves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from quansino.mc.contexts import DisplacementContext
from quansino.moves.composite import CompositeDisplacementMove
from quansino.moves.core import BaseMove
from quansino.moves.operations import Box, CompositeOperation, Operation

if TYPE_CHECKING:
    from quansino.type_hints import IntegerArray


class DisplacementMove[
    OperationType: Operation | CompositeOperation, ContextType: DisplacementContext
](BaseMove[OperationType, ContextType]):
    """
    Class for displacement moves that displaces one atom or a group of atoms. The class will use an [`Operation`][quansino.moves.operations.Operation]. The class uses the `labels` attribute to determine which atoms can be displaced, if none, the move fails. If multiple atoms share the same label, they are considered to be part of the same group (molecule) and will be displaced together in a consistent manner.

    Move that displaces multiple labels at once can be created by adding multiple [`DisplacementMove`][quansino.moves.displacements.DisplacementMove] objects together. Similarly, a move can be multiplied by an integer to move multiple labels at once in the same manner. In this case a [`CompositeDisplacementMove`][quansino.moves.composite.CompositeDisplacementMove] object is returned, which can be used as a normal DisplacementMove object.

    The move only modify the `moving_indices` attribute of the context object, which might be needed for some operations (Rotation, for example).

    Parameters
    ----------
    moving_labels : IntegerArray
        The labels of the atoms to displace. Atoms with negative labels are not displaced.
    operation : OperationType, optional
        The operation to perform in the move, by default None.
    apply_constraints : bool, optional
        Whether to apply constraints to the move, by default True.

    Attributes
    ----------
    CompositeMove : CompositeDisplacementMove
        The composite move class that is returned when adding or multiplying moves.
    AcceptableContext : DisplacementContext
        The required context for the move. The context object aim to provide the necessary information for the move to perform its operation, without having to pass whole objects around. Classes inheriting from [`BaseMove`][quansino.moves.core.BaseMove] should define a `AcceptableContext` attribute that specifies the context class that the move requires.
    to_displace_labels : int | None
        The label of the atoms to displace. If None, the move will select the atoms to displace itself. Reset to None after move.
    displaced_labels : int | None
        The label of the atoms that were displaced in the last move. Reset to None after move.
    default_label : int | None
        The default label when adding new atoms. If None, labels for new atoms will be selected automatically.

    Important
    ---------
    1. At object creation, `moving_labels` must have the same length as the number of atoms in the simulation.
    2. Conditions for a successful move can be tightened either by subclassing the move and overriding the `check_move` method, or by using the [`MethodType`][types.MethodType] function to replace the method in the instance.
    """

    CompositeMove = CompositeDisplacementMove
    AcceptableContext = DisplacementContext

    def __init__(
        self,
        moving_labels: IntegerArray,
        operation: OperationType | None = None,
        apply_constraints: bool = True,
    ) -> None:
        """Initialize the DisplacementMove object.

        Parameters
        ----------
        moving_labels : IntegerArray
            The labels of the atoms to displace. Atoms with negative labels are not displaced.
        operation : OperationType, optional
            The operation to perform in the move, by default None.
        apply_constraints : bool, optional
            Whether to apply constraints to the move, by default True.
        """

        self.to_displace_labels: int | None = None
        self.displaced_labels: int | None = None

        self.default_label: int | None = None
        self.set_labels(moving_labels)

        if operation is None:
            operation = cast(OperationType, Box(0.1))

        super().__init__(operation, apply_constraints)

    def attempt_move(self) -> bool:
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
            if len(self.unique_labels) == 0:
                return self.register_failure()

            self.to_displace_labels = self.context.rng.choice(self.unique_labels)

        (self.context.moving_indices,) = np.where(
            self.labels == self.to_displace_labels
        )

        if self.attempt_move():
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

    def __add__(
        self: DisplacementMove, other: DisplacementMove | CompositeMove
    ) -> CompositeMove:
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
        if isinstance(other, self.CompositeMove):
            return self.CompositeMove([self, *other.moves])
        else:
            return self.CompositeMove([self, other])

    def __mul__(self: DisplacementMove, n: int) -> CompositeMove:
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
        return self.CompositeMove([self] * n)

    __rmul__ = __mul__

    def __copy__(self) -> DisplacementMove:
        """
        Create a shallow copy of the move.

        Returns
        -------
        DisplacementMove
            The shallow copy of the move.
        """
        new_move = DisplacementMove(self.labels, self.operation, self.apply_constraints)
        new_move.__dict__.update(self.__dict__)

        return new_move
