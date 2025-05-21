"""Module for the ExchangeMove class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from ase.atoms import Atoms
from ase.build import molecule

from quansino.mc.contexts import ExchangeContext
from quansino.moves.displacement import DisplacementMove
from quansino.operations.displacement import Translation, TranslationRotation

if TYPE_CHECKING:
    from quansino.operations.core import Operation
    from quansino.type_hints import IntegerArray


class ExchangeMove[ContextType: ExchangeContext](DisplacementMove[ContextType]):
    """
    Class for atomic/molecular exchange moves that exchanges atom(s). The class will either add `exchange_atoms` in the unit cell or delete a (group) of atom(s) present in `labels`.

    For addition, the move uses the `attempt_move` method in the parent [`DisplacementMove`][quansino.moves.displacement.DisplacementMove] class with the provided [`operation`][quansino.operations.core.Operation] (Translation by default for single atoms, TranslationRotation for multiple atoms).

    For deletion, the move will attempt to remove atoms with non-negative labels from the parent class [`DisplacementMove`][quansino.moves.displacement.DisplacementMove]. The context's `save_state()` method must be called after successful moves to update the `labels` of other [`DisplacementMoves`][quansino.moves.displacement.DisplacementMove] linked to the simulation.

    Parameters
    ----------
    exchange_atoms : Atoms | str
        The atoms to exchange. If a string is provided, it will be converted to an Atoms object using ase.build.molecule.
    labels : IntegerArray
        The labels of the atoms that can be exchanged (already present).
    operation : Operation, optional
        The operation to perform in the move, by default None (will use Translation for single atoms or TranslationRotation for multiple atoms).
    bias_towards_insert : float, optional
        The probability of inserting atoms instead of deleting, by default 0.5.
    apply_constraints : bool, optional
        Whether to apply constraints during the move, by default True.

    Attributes
    ----------
    exchange_atoms : Atoms
        The atoms to exchange.
    bias_towards_insert : float
        The probability of inserting atoms instead of deleting, can be used to bias the move towards insertion or deletion.
    to_add_atoms : Atoms | None
        The atoms to add during the next move, reset after each move.
    to_delete_indices : int | None
        The indices of the atoms to delete during the next move, reset after each move.

    Important
    ---------
    1. At object creation, `labels` must have the same length as the number of atoms in the simulation.
    2. Any labels that are not negative integers are considered exchangeable (deletable).
    3. Atoms that share the same label are considered to be part of the same group (molecule) and will be deleted together.
    4. Monte Carlo simulations like [`GrandCanonical`][quansino.mc.gcmc.GrandCanonical] will automatically update the labels of all linked moves to keep them in sync.
    """

    AcceptableContext = ExchangeContext

    def __init__(
        self,
        exchange_atoms: Atoms | str,
        labels: IntegerArray,
        operation: Operation | None = None,
        bias_towards_insert: float = 0.5,
        apply_constraints: bool = True,
    ) -> None:
        """Initialize the ExchangeMove object."""
        if isinstance(exchange_atoms, str):
            exchange_atoms = molecule(exchange_atoms)

        self.exchange_atoms = cast(Atoms, exchange_atoms)

        self.bias_towards_insert = bias_towards_insert

        self.to_add_atoms: Atoms | None = None
        self.to_delete_indices: int | None = None

        if operation is None:
            if len(exchange_atoms) > 1:
                default_operation = TranslationRotation()
            else:
                default_operation = Translation()

            operation = default_operation

        super().__init__(labels, operation, apply_constraints)

    def __call__(self) -> bool:
        """
        Perform the exchange move. The following steps are performed:

        1. Reset the context, this is done to clear any previous move attributes such as `moving_indices`, `added_indices`, `deleted_indices`, `particle_delta`, `added_atoms`, and `deleted_atoms`, which are needed to keep track of the move and calculate the acceptance probability.
        2. Decide whether to insert or delete atoms, this can be pre-selected by setting the `to_add_atoms` or `to_delete_indices` attributes before calling the move. If not, the decision is made randomly based on the `bias_towards_insert` attribute.
        3. If adding atoms, add the atoms to the atoms object and attempt to place them at the new positions using the parent class [`DisplacementMove.attempt_move`][quansino.moves.displacement.DisplacementMove.attempt_move]. If the move is not successful, remove the atoms from the atoms object and register the exchange failure. If deleting atoms, remove the atoms from the atoms object, failure is only possible if all labels are negative integers (no atoms to delete).
        3. In case of an addition, attempt to place the atoms at the new positions using the parent class [`DisplacementMove.attempt_move`][quansino.moves.displacement.DisplacementMove.attempt_move]. If the move is not successful, register the exchange failure and return False.
        4. During these steps, all attribute in the context object are updated to keep track of the move and can be used later for multiple purposes such as calculating the acceptance probability.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        self.context.reset()

        if self.to_add_atoms is None and self.to_delete_indices is None:
            is_addition = self.context.rng.random() < self.bias_towards_insert
        else:
            is_addition = bool(self.to_add_atoms)

        if is_addition:
            self.to_add_atoms = self.to_add_atoms or self.exchange_atoms

            self.addition()
            self.context.moving_indices = np.arange(len(self.context.atoms))[
                -len(self.to_add_atoms) :
            ]

            if not super().attempt_move():
                del self.context.atoms[self.context.moving_indices]
                return self.register_failure()

            self.context.added_indices = self.context.moving_indices
            self.context.added_atoms = self.to_add_atoms
        else:
            if self.to_delete_indices is None:
                if not len(self.unique_labels):
                    return self.register_failure()

                self.to_delete_indices = int(
                    self.context.rng.choice(self.unique_labels)
                )

            (self.context.deleted_indices,) = np.where(
                self.labels == self.to_delete_indices
            )

            self.context.deleted_atoms = self.context.atoms[
                self.context.deleted_indices
            ]  # type: ignore
            self.deletion()

        if bool(self.context.added_atoms) or bool(self.context.deleted_atoms):
            return self.register_success()
        else:
            return self.register_failure()

    def addition(self) -> None:
        """
        Add atoms to the atoms object.
        """
        self.context.atoms.extend(self.to_add_atoms)

    def deletion(self) -> None:
        """
        Delete atoms from the atoms object.
        """
        del self.context.atoms[self.context.deleted_indices]

    def register_success(self) -> Literal[True]:
        """
        Register a successful exchange move, in which case all information is retained except the prior move attributes.

        Returns
        -------
        Literal[True]
            Always returns True.
        """
        self.to_add_atoms = None
        self.to_delete_indices = None

        return True

    def register_failure(self) -> Literal[False]:
        """
        Register a failed exchange move, in which case all information is retained except the prior move attributes.

        Returns
        -------
        Literal[False]
            Always returns False.
        """
        self.to_add_atoms = None
        self.to_delete_indices = None

        return False

    def to_dict(self) -> dict[str, Any]:
        dictionary = super().to_dict()
        kwargs = dictionary.setdefault("kwargs", {})
        kwargs["exchange_atoms"] = self.exchange_atoms
        kwargs["bias_towards_insert"] = self.bias_towards_insert

        return dictionary
