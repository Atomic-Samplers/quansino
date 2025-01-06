"""Module for the ExchangeMove class."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from ase.build import molecule

from quansino.mc.contexts import Context, ExchangeContext
from quansino.moves.displacements import DisplacementMove
from quansino.moves.operations import (
    Exchange,
    ExchangeTranslation,
    ExchangeTranslationRotation,
    Operation,
)

if TYPE_CHECKING:
    from ase.atoms import Atoms

    from quansino.typing import IntegerArray


class ExchangeMove(DisplacementMove):
    """
    Class for an atomic exchange move that exchanges atom(s). The class will use an [Exchange][quansino.moves.operations.Exchange] operation to place the Atoms object `exchange_atoms` in the unit cell and update the indices of the `candidate_indices` of [`DisplacementMoves`][quansino.moves.displacements.DisplacementMove] objects listed in the `moves` dictionary of [`ExchangeContext`][quansino.mc.contexts.ExchangeContext]

    Parameters
    ----------
    exchange_atoms : Atoms | str
        The atoms to exchange.
    candidate_indices : IntegerArray, optional
        The indices of the already present atoms that can be exchanged, by default None.
    bias_towards_insert : float, optional
        The probability of inserting atoms instead of deleting, by default 0.5.
    move_type : Operation, optional
        The operation to perform in the move, by default None.
    move_updates_to_skip : str | list[str], optional
        The move updates to skip, by default None.

    Attributes
    ----------
    exchange_atoms : Atoms
        The atoms to exchange.
    move_operator : Exchange
        The exchange operation, must be an instance of `quansino.moves.operations.Exchange`.
    bias_towards_insert : float
        The probability of inserting atoms instead of deleting, can be used to bias the move towards insertion or deletion.
    move_updates_to_skip : list[str]
        By default, `ExchangeMove` will update the indices of all displacement moves belongig to the current context. This list can be used to skip updating the indices of specific moves.
    context : ExchangeContext
        The context for the move.
    required_context : ExchangeContext
        The required context for the move.

    Notes
    -----
    The `ExchangeMove` class is a subclass of `DisplacementMove` and is used to perform atomic exchanges in a Monte Carlo simulation. The move can be biased towards insertion or deletion, and can be used to exchange multiple atoms at once. The move can be used with any operation that is a subclass of `quansino.moves.operations.Exchange`.
    """

    required_context = ExchangeContext

    def __init__(
        self,
        exchange_atoms: Atoms | str,
        candidate_indices: IntegerArray | None = None,
        bias_towards_insert: float = 0.5,
        move_type: Operation | None = None,
        move_updates_to_skip: str | list[str] | None = None,
    ) -> None:
        """Initialize the ExchangeMove object."""
        if isinstance(exchange_atoms, str):
            exchange_atoms = molecule(exchange_atoms)

        self.exchange_atoms = exchange_atoms

        default_move = (
            ExchangeTranslationRotation()
            if len(exchange_atoms) > 1
            else ExchangeTranslation()
        )

        move_type = move_type or default_move

        super().__init__(
            displacement_type=move_type, candidate_indices=candidate_indices
        )

        self.move_operator = cast(Exchange, self.move_operator)

        self.bias_towards_insert = bias_towards_insert

        self.set_move_updates_to_skip(move_updates_to_skip or [])

    def __call__(self) -> bool:
        """
        Perform the exchange move. The following steps are performed:

        1. Reset the context, and decide whether to insert or delete atoms.
        2. Perform the move using the move operator attached to the move.
        3. In case of an addition, attempt to place the atoms at the new positions using the parent class `DisplacementMove.attempt_move`. If the move is not successful, register the exchange failure and return False.
        4. In case of a deletion, remove the atoms from the atoms object.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        self.context.reset()

        if self.context.rng.random() < self.bias_towards_insert:
            if self.context.addition_candidates is None:
                self.context.addition_candidates = self.exchange_atoms

            self.move_operator.addition(self.context)
            moving_indices = np.arange(len(self.context.atoms))[
                -len(self.context.addition_candidates) :
            ]

            if not super().attempt_move(moving_indices):
                self.context.register_exchange_failure()
                return False

            self.context.last_added_indices = moving_indices
        else:
            if self.context.deletion_candidates is None:
                if not self._number_of_available_candidates:
                    return False

                self.context.deletion_candidates = np.asarray(
                    self.context.rng.choice(self._unique_candidates)
                )

            mask = np.isin(self.candidate_indices, self.context.deletion_candidates)
            self.context.last_deleted_indices = np.where(mask)[0]
            self.context.last_deleted_atoms = self.context.atoms[mask]  # type: ignore

            del self.context.atoms[mask]

        return True

    def set_move_updates_to_skip(self, move_updates_to_skip: str | list[str]) -> None:
        """
        Set the move updates to skip.

        Parameters
        ----------
        move_updates_to_skip : str | list[str]
            The move updates to skip.
        """
        if isinstance(move_updates_to_skip, str):
            self.move_updates_to_skip = [move_updates_to_skip]
        else:
            self.move_updates_to_skip = move_updates_to_skip

    def attach_simulation(
        self, context: Context, update_candidates: bool = True
    ) -> None:
        """
        Attach the simulation to the move.

        Parameters
        ----------
        context : Context
            The context to attach to the move. Must be an instance of [`ExchangeContext`][quansino.mc.contexts.ExchangeContext].
        update_candidates : bool, optional
            Whether to update the candidates, by default True.

        Raises
        ------
        ValueError
            If the context is not an instance of [`ExchangeContext`][quansino.mc.contexts.ExchangeContext].
        """
        super().attach_simulation(context, update_candidates)
        self.context = cast(ExchangeContext, self.context)

    def update_moves(self) -> None:
        """Update the indices of the displacement moves in the context."""
        for name, move in self.context.moves.items():
            if name not in self.move_updates_to_skip:
                move.update_indices(
                    self.context.last_added_indices, self.context.last_deleted_indices
                )
