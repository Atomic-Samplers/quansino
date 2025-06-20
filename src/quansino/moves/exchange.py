"""Module for the ExchangeMove class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from quansino.mc.contexts import ExchangeContext
from quansino.moves.composite import CompositeMove
from quansino.moves.displacement import DisplacementMove
from quansino.operations.displacement import Translation
from quansino.protocols import Operation

if TYPE_CHECKING:
    from ase.atoms import Atoms

    from quansino.type_hints import IntegerArray


class ExchangeMove[OperationType: Operation, ContextType: ExchangeContext](
    DisplacementMove[OperationType, ContextType]
):
    """
    Class for atomic/molecular exchange moves that exchanges atom(s). The class will either add `exchange_atoms` in the unit cell or delete a (group) of atom(s) present in `labels`.

    For addition, the move uses the `attempt_move` method in the parent [`DisplacementMove`][quansino.moves.displacement.DisplacementMove] class with the provided [`operation`][quansino.operations.core.Operation] (Translation by default for single atoms, TranslationRotation for multiple atoms).

    For deletion, the move will attempt to remove atoms with non-negative labels from the parent class [`DisplacementMove`][quansino.moves.displacement.DisplacementMove]. The context's `save_state()` method must be called after successful moves to update the `labels` of other [`DisplacementMoves`][quansino.moves.displacement.DisplacementMove] linked to the simulation.

    Parameters
    ----------
    labels : IntegerArray
        The labels of the atoms that can be exchanged (already present).
    operation : Operation | None, optional
        The operation to perform in the move, default is None, which will use the default operation (Translation).
    bias_towards_insert : float, optional
        The probability of inserting atoms instead of deleting, default is 0.5.
    apply_constraints : bool, optional
        Whether to apply constraints during the move, default is True.

    Attributes
    ----------
    bias_towards_insert : float
        The probability of inserting atoms instead of deleting, can be used to bias the move towards insertion or deletion.
    to_add_atoms : Atoms | None
        The atoms to add during the next move, reset after each move.
    to_delete_label : int | None
        The indices of the atoms to delete during the next move, reset after each move.

    Important
    ---------
    1. At object creation, `labels` must have the same length as the number of atoms in the simulation.
    2. Any labels that are not negative integers are considered exchangeable (deletable).
    3. Atoms that share the same label are considered to be part of the same group (molecule) and will be deleted together.
    4. Monte Carlo simulations like [`GrandCanonical`][quansino.mc.gcmc.GrandCanonical] will automatically update the labels of all linked moves to keep them in sync.
    """

    __slots__ = (
        "bias_towards_insert",
        "exchange_atoms",
        "to_add_atoms",
        "to_delete_label",
    )

    def __init__(
        self,
        labels: IntegerArray,
        operation: OperationType | None = None,
        bias_towards_insert: float = 0.5,
        apply_constraints: bool = True,
    ) -> None:
        """Initialize the ExchangeMove object."""
        self.bias_towards_insert = bias_towards_insert

        self.to_add_atoms: Atoms | None = None
        self.to_delete_label: int | None = None

        super().__init__(labels, operation, apply_constraints)

        self.composite_move_type = CompositeExchangeMove

    def attempt_addition(self, context: ContextType) -> IntegerArray:
        """
        Attempt to add atoms to the simulation. If `to_add_atoms` is not set, it will use the `exchange_atoms` from the context.

        Returns
        -------
        IntegerArray
            The indices of the added atoms.
        """
        self.to_add_atoms = self.to_add_atoms or context.exchange_atoms

        context.atoms.extend(self.to_add_atoms)
        context._moving_indices = np.arange(len(context.atoms))[
            -len(self.to_add_atoms) :
        ]

        if not super().attempt_displacement(context):
            del context.atoms[context._moving_indices]
            return []

        return context._moving_indices

    def attempt_deletion(self, context: ContextType) -> IntegerArray:
        """
        Attempt to delete atoms from the simulation. If `to_delete_label` is not set, it will randomly select a label from the unique labels of the context.

        Returns
        -------
        IntegerArray
            The indices of the deleted atoms.
        """
        if self.to_delete_label is None:
            if not len(self.unique_labels):
                return []

            self.to_delete_label = int(context.rng.choice(self.unique_labels))

        (indices,) = np.where(self.labels == self.to_delete_label)

        if not len(indices):
            return []

        return indices

    def __call__(self, context: ContextType) -> bool:
        """
        Perform the exchange move. The following steps are performed:

        1. Decide whether to insert or delete atoms, this can be pre-selected by setting the `to_add_atoms` or `to_delete_label` attributes before calling the move. If not, the decision is made randomly based on the `bias_towards_insert` attribute.
        2. If adding atoms, add the atoms to the atoms object and attempt to place them using the parent class [`DisplacementMove.attempt_displacement`][quansino.moves.displacement.DisplacementMove.attempt_displacement]. If the move is not successful, remove the atoms from the atoms object and register the exchange failure. If deleting atoms, remove the atoms from the atoms object, failure is only possible if all labels are negative integers (no atoms to delete).
        3. During these steps, attributes in the context object are updated to keep track of the move and can be used later for multiple purposes such as calculating the acceptance probability.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        if self.to_add_atoms is None and self.to_delete_label is None:
            is_addition = context.rng.random() < self.bias_towards_insert
        else:
            is_addition = bool(self.to_add_atoms)

        if is_addition:
            indices = self.attempt_addition(context)

            if len(indices):
                context._added_indices = np.hstack(
                    (context._added_indices, indices), dtype=np.int_, casting="unsafe"
                )
                context._added_atoms += context.atoms[indices]
                context.particle_delta += 1

                return self.register_success()
        else:
            indices = self.attempt_deletion(context)

            if len(indices):
                context._deleted_indices = np.hstack(
                    (context._deleted_indices, indices), dtype=np.int_, casting="unsafe"
                )
                context._deleted_atoms += context.atoms[indices]
                context.particle_delta -= 1

                del context.atoms[indices]
                return self.register_success()

        return self.register_failure()

    def register_success(self) -> Literal[True]:
        """
        Register a successful exchange move, in which case all information is retained except the prior move attributes.

        Returns
        -------
        Literal[True]
            Always returns True.
        """
        self.to_add_atoms = None
        self.to_delete_label = None

        return True

    @property
    def default_operation(self) -> Operation:
        """
        Get the default operation for the exchange move.

        Returns
        -------
        Operation
            The default operation, which is Translation for single atoms or TranslationRotation for molecules.
        """
        return Translation()

    def register_failure(self) -> Literal[False]:
        """
        Register a failed exchange move, in which case all information is retained except the prior move attributes.

        Returns
        -------
        Literal[False]
            Always returns False.
        """
        self.to_add_atoms = None
        self.to_delete_label = None

        return False

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ExchangeMove object to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `ExchangeMove` object
        """
        dictionary = super().to_dict()

        kwargs = dictionary.setdefault("kwargs", {})
        kwargs["bias_towards_insert"] = self.bias_towards_insert

        return dictionary


class CompositeExchangeMove(CompositeMove[ExchangeMove]):

    def __init__(self, moves: list[ExchangeMove]) -> None:
        """
        Initialize the CompositeExchangeMove object.

        Parameters
        ----------
        moves : list[ExchangeMove]
            The moves to perform in the composite move.
        """
        super().__init__(moves)

        self.bias_towards_insert: float = 0.5

    def __call__(self, context: ExchangeContext) -> bool:
        """
        Perform the composite exchange move. The following steps are performed:

        1. Decide whether to perform addition or deletion based on the `bias_towards_insert` attribute. This will be the same for all moves in the composite, if you want to have different biases for each move, use the `CompositeMove` class with individual `ExchangeMove` objects.
        2. For addition moves, attempt to add atoms using attempt_addition(). If successful, register the success.
        3. For deletion moves, filter out already deleted labels to avoid conflicts, then select an available candidate from unique_labels. If no candidates are available, register deletion failure and continue to next move.
        4. For valid deletion candidates, set the move's to_delete_label and attempt deletion. If successful, register the deletion success.
        5. Return True if any of the individual moves were successful, False otherwise.

        Returns
        -------
        bool
            Whether any of the exchange moves in the composite were valid.
        """
        is_addition = context.rng.random() < self.bias_towards_insert

        if is_addition:
            success = False

            for move in self.moves:
                indices = move.attempt_addition(context)

                if len(indices):
                    context._added_indices = np.hstack(
                        (context._added_indices, indices),
                        dtype=np.int_,
                        casting="unsafe",
                    )
                    context._added_atoms += context.atoms[indices]
                    context.particle_delta += 1
                    success = True

                move.to_add_atoms = None
            return success
        else:
            deleted_labels = []
            deleted_indices = np.array([], dtype=np.int_)

            for move in self.moves:
                available_candidates = np.setdiff1d(
                    move.unique_labels, deleted_labels, assume_unique=True
                )
                if len(available_candidates) == 0:
                    continue

                to_delete_label = context.rng.choice(available_candidates)

                (indices,) = np.where(move.labels == to_delete_label)

                deleted_indices = np.hstack(
                    (deleted_indices, indices), dtype=np.int_, casting="unsafe"
                )
                deleted_labels.append(to_delete_label)

            if len(deleted_indices) == 0:
                return False

            context._deleted_indices = deleted_indices
            context._deleted_atoms += context.atoms[deleted_indices]
            context.particle_delta -= len(np.unique(deleted_labels))

            del context.atoms[deleted_indices]

            return True
