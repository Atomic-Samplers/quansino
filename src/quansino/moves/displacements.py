"""Module for displacement moves."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from quansino.mc.contexts import DisplacementContext
from quansino.moves.core import BaseMove
from quansino.moves.operations import Ball, Operation

if TYPE_CHECKING:
    from quansino.mc.contexts import Context
    from quansino.typing import IntegerArray


class DisplacementMove(BaseMove):
    """
    Class for displacement moves that displaces one atom or a group of atoms. The class will use an [`Operation`][quansino.moves.operations.Operation] object to calculate `displacements_per_move` number of displacements aiming to move atom(s) listed in `candidate_indices`.

    Parameters
    ----------
    displacement_type : Operation, optional
        The operation to perform in the move, by default None.
    candidate_indices : IntegerArray, optional
        The indices of the atoms that can be displaced, by default None.
    displacements_per_move : int, optional
        The number of atom/molecules displacements to perform in each move, by default 1.
    apply_constraints : bool, optional
        Whether to apply constraints to the move, by default True.

    Attributes
    ----------
    displacements_per_move : int
        The number of atom/molecules displacements to perform in each move.
    max_attempts : int
        The maximum number of attempts to perform the move. If the move is not accepted after `max_attempts`, the move will be rejected. By default, `max_attempts` is set to 10000.
    required_context : DisplacementContext
        The required context for the move. The context object aim to provide the necessary information for the move to perform its operation, without having to pass whole objects around. Classes inheriting from [`BaseMove`][quansino.moves.core.BaseMove] should define a `required_context` attribute that specifies the context class that the move requires.
    """

    max_attempts: int = 10000
    required_context = DisplacementContext

    def __init__(
        self,
        displacement_type: Operation | None = None,
        candidate_indices: IntegerArray | None = None,
        displacements_per_move: int = 1,
        apply_constraints: bool = True,
    ) -> None:
        """Initialize the DisplacementMove object."""
        if candidate_indices is None:
            candidate_indices = []

        self.set_candidate_indices(candidate_indices)

        displacement_type = displacement_type or Ball()

        self.displacements_per_move = displacements_per_move

        super().__init__(
            move_operator=displacement_type, apply_constraints=apply_constraints
        )

    def attempt_move(self, moving_indices: IntegerArray) -> bool:
        """
        Attempt to move the atoms using the provided operation and check.

        Parameters
        ----------
        moving_indices : IntegerArray
            The indices of the atoms to displace.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        atoms = self.context.atoms
        old_positions = atoms.get_positions()

        self.context.moving_indices = moving_indices

        for _ in range(self.max_attempts):
            translation = np.full((len(atoms), 3), 0.0)
            translation[moving_indices] = self.move_operator.calculate(self.context)

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

        1. If no candidates are available, return False and does not register a move.
        2. Check if there are enough candidates to displace. If yes, select `displacements_per_move` number of candidates from the available candidates, if not, select the maximum number of candidates available.
        3. If `selected_candidates` is None, select `displacements_per_move` candidates from the available candidates.
        4. Attempt to move each candidate using `attempt_move`. If any of the moves is successful, register a success and return True. Otherwise, register a failure and return False.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        if not self._number_of_available_candidates:
            return False

        candidate_count = (
            len(self._unique_candidates)
            if self._number_of_available_candidates < self.displacements_per_move
            else self.displacements_per_move
        )

        if self.context.selected_candidates is None:
            self.context.selected_candidates = self.context.rng.choice(
                self._unique_candidates, size=candidate_count, replace=False
            )

        # ruff: noqa: C419
        attempted_moves = any(
            [
                self.attempt_move(np.where(self.candidate_indices == candidate)[0])
                for candidate in self.context.selected_candidates
            ]
        )

        if attempted_moves:
            self.context.register_success()
        else:
            self.context.register_failure()

        return attempted_moves

    def set_candidate_indices(self, new_indices: IntegerArray) -> None:
        """
        Set the candidate indices for the move. The candidate indices are the indices of the atoms that can be displaced.

        Parameters
        ----------
        new_indices : IntegerArray
            The new candidate indices.
        """
        self.candidate_indices = np.asarray(new_indices)
        self._unique_candidates = np.unique(
            self.candidate_indices[self.candidate_indices >= 0]
        )
        self._number_of_available_candidates = len(self._unique_candidates)

    def update_indices(
        self, to_add: IntegerArray | None = None, to_remove: IntegerArray | None = None
    ) -> None:
        """
        Update the candidate indices by adding or removing indices. Only to be used when new atoms are added or removed from the system.

        Parameters
        ----------
        to_add : IntegerArray, optional
            The indices of the atoms to add, by default None.
        to_remove : IntegerArray, optional
            The indices of the atoms to remove, by default None.

        Raises
        ------
        ValueError
            If neither or both `to_add` or `to_remove` are provided.
        """
        is_addition = to_add is not None
        is_removal = to_remove is not None

        if not (is_addition ^ is_removal):
            raise ValueError("Either `to_add` or `to_remove` should be provided")

        if is_addition:
            index: int = (
                np.max(self.candidate_indices) + 1
                if self._number_of_available_candidates
                else 0
            )
            self.set_candidate_indices(
                np.hstack((self.candidate_indices, np.full(len(to_add), index)))
            )
        elif is_removal:
            self.set_candidate_indices(np.delete(self.candidate_indices, to_remove))  # type: ignore

    def attach_simulation(
        self, context: Context, update_candidates: bool = True
    ) -> None:
        """
        Attach the simulation to the move.

        Parameters
        ----------
        context : Context
            The context to attach to the move. Must be an instance of `quansino.mc.contexts.DisplacementContext`.
        update_candidates : bool, optional
            Whether to update the candidates, by default True.
        """
        if not isinstance(context, self.required_context):
            raise TypeError(
                f"Expected a {self.required_context.__name__}, got {type(context).__name__}"
            )

        if update_candidates:
            self.set_candidate_indices(np.arange(len(context.atoms)))

        super().attach_simulation(context)
        self.context = cast(DisplacementContext, self.context)
