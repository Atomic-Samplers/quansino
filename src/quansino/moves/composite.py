from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quansino.mc.contexts import DisplacementContext
from quansino.moves.core import BaseMove

if TYPE_CHECKING:
    from quansino.moves.displacements import DisplacementMove
    from quansino.moves.operations import DisplacementOperation, Operation


class CompositeDisplacementMove[
    OperationType: Operation, ContextType: DisplacementContext
](BaseMove[OperationType, ContextType]):
    """
    Class to perform a composite displacement operation on atoms.

    Parameters
    ----------
    moves : list[DisplacementMove]
        The moves to perform in the composite move.

    Attributes
    ----------
    moves : list[DisplacementMove]
        The moves to perform in the composite move

    Methods
    -------
    calculate(context: DisplacementContext) -> Displacement
        Calculate the composite displacement operation to perform on the atoms.
    """

    def __init__(
        self, moves: list[DisplacementMove[DisplacementOperation, ContextType]]
    ) -> None:
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
                    move.unique_labels, filtered_displaced_labels, assume_unique=True
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
