"""Module for cell moves."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quansino.mc.contexts import DeformationContext
from quansino.moves.composite import CompositeMove
from quansino.moves.core import BaseMove
from quansino.operations.cell import IsotropicDeformation

if TYPE_CHECKING:
    from quansino.operations.core import Operation


class CellMove[OperationType: Operation, ContextType: DeformationContext](
    BaseMove[OperationType, ContextType]
):
    """
    Class for cell moves that change the size and shape of the unit cell.

    Parameters
    ----------
    operation : OperationType | None, optional
        The operation to perform in the move, by default None. If None, it defaults to [`IsotropicDeformation`][quansino.operations.cell.IsotropicDeformation].
    scale_atoms : bool, optional
        Whether to scale the atom positions when the cell changes, by default True.
    apply_constraints : bool, optional
        Whether to apply constraints to the move, by default True.

    Attributes
    ----------
    scale_atoms : bool
        Whether to scale the atom positions when the cell changes.
    composite_move_type : CompositeMove[CellMove[OperationType, ContextType]]
        The type of composite move returned by this move when adding it with another `CellMove`.
    """

    __slots__ = ("scale_atoms",)

    def __init__(
        self,
        operation: OperationType | None = None,
        scale_atoms: bool = True,
        apply_constraints: bool = True,
    ) -> None:
        """Initialize the `CellMove` object."""
        super().__init__(operation, apply_constraints)

        self.scale_atoms = scale_atoms

        self.composite_move_type = CompositeMove[CellMove[OperationType, ContextType]]

    def attempt_deformation(self, context: ContextType) -> bool:
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
        old_cell = atoms.cell.copy()
        old_positions = atoms.positions.copy()

        for _ in range(self.max_attempts):
            deformation_gradient = self.operation.calculate(context)

            atoms.set_cell(
                deformation_gradient @ old_cell,
                scale_atoms=self.scale_atoms,
                apply_constraint=self.apply_constraints,
            )

            if self.check_move(context):
                return True

            atoms.cell = old_cell

            if self.scale_atoms:
                atoms.positions = old_positions

        return False

    def __call__(self, context: ContextType) -> bool:
        """
        Perform the cell move.

        Parameters
        ----------
        context : ContextType
            The context for the move.

        Returns
        -------
        bool
            Whether the move was valid.
        """
        return self.attempt_deformation(context)

    @property
    def default_operation(self) -> Operation:
        """
        Get the default operation for the move.

        Returns
        -------
        Operation
            The default operation for the move.
        """
        return IsotropicDeformation(0.05)
