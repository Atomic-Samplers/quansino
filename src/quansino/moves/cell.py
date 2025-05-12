"""Module for cell moves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from quansino.mc.contexts import StrainContext
from quansino.moves.core import BaseMove

if TYPE_CHECKING:
    from quansino.operations.core import Operation


class CellMove[ContextType: StrainContext](BaseMove[ContextType]):
    """
    Class for cell moves that change the size and shape of the unit cell.

    Parameters
    ----------
    operation : Operation
        The operation to perform in the move.
    scale_atoms : bool, optional
        Whether to scale the atom positions when the cell changes, by default True.
    apply_constraints : bool, optional
        Whether to apply constraints to the move, by default True.

    Attributes
    ----------
    is_updatable : Literal[False]
        Whether the move can be updated when atoms are added or removed.
    scale_atoms : bool
        Whether to scale the atom positions when the cell changes.
    """

    is_updatable: Literal[False] = False

    def __init__(
        self,
        operation: Operation,
        scale_atoms: bool = True,
        apply_constraints: bool = True,
    ) -> None:
        super().__init__(operation, apply_constraints)

        self.scale_atoms = scale_atoms

    def attempt_move(self) -> bool:
        """
        Attempt to move the atoms using the provided operation and check. The move is attempted `max_attempts` number of times. If the move is successful, return True, otherwise, return False.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        atoms = self.context.atoms
        old_cell = atoms.cell.copy()
        old_positions = atoms.positions.copy()

        for _ in range(self.max_attempts):
            deformation_gradient = self.operation.calculate(self.context)

            atoms.set_cell(
                deformation_gradient @ old_cell,
                scale_atoms=self.scale_atoms,
                apply_constraint=self.apply_constraints,
            )

            if self.check_move():
                return True

            atoms.cell = old_cell

            if self.scale_atoms:
                atoms.positions = old_positions

        return False

    def __call__(self) -> bool:
        """
        Perform the cell move.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        return self.attempt_move()
