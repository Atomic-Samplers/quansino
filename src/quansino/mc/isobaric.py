"""Module to perform isobaric (NPT) Monte Carlo simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from warnings import warn

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import StrainContext
from quansino.mc.criteria import CanonicalCriteria, IsobaricCriteria
from quansino.moves.cell import CellMove
from quansino.moves.displacement import DisplacementMove
from quansino.moves.protocol import CellProtocol, DisplacementProtocol

if TYPE_CHECKING:
    from collections.abc import Generator

    from ase.atoms import Atoms

    from quansino.mc.core import MoveStorage


class Isobaric[
    MoveProtocol: DisplacementProtocol | CellProtocol, ContextType: StrainContext
](Canonical[MoveProtocol, ContextType]):
    """
    Isobaric (NPT) Monte Carlo simulation object.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on, will be acted upon in place.
    temperature : float
        The temperature of the simulation in Kelvin.
    pressure : float, optional
        The pressure of the simulation in eV/Å^3, by default 0.0.
    max_cycles : int, optional
        The number of Monte Carlo cycles to perform, by default equal to the number of atoms.
    default_displacement_move : MoveStorage[MoveProtocol] | MoveProtocol, optional
        The default displacement move to perform in each cycle. If a `MoveStorage` object is provided,
        it will be used to initialize the move with its criteria and other parameters.
        If a `MoveProtocol` object is provided, it will be added using the default criteria and parameters, by default None.
    default_cell_move : MoveStorage[MoveProtocol] | MoveProtocol, optional
        The default cell move to perform in each cycle. If a `MoveStorage` object is provided,
        it will be used to initialize the move with its criteria and other parameters.
        If a `MoveProtocol` object is provided, it will be added using the default criteria and parameters, by default None.
    **mc_kwargs
        Additional keyword arguments to pass to the [`MonteCarlo`][quansino.mc.core.MonteCarlo] parent class.

    Attributes
    ----------
    pressure : float
        The pressure of the simulation in eV/Å^3.
    last_cell : Cell
        The last cell of the atoms object in the simulation.
    """

    default_criteria: ClassVar = {
        DisplacementMove: CanonicalCriteria,
        CellMove: IsobaricCriteria,
    }
    default_context: ClassVar = StrainContext

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        pressure: float = 0.0,
        max_cycles: int | None = None,
        default_displacement_move: (
            MoveStorage[MoveProtocol] | MoveProtocol | None
        ) = None,
        default_cell_move: MoveStorage[MoveProtocol] | MoveProtocol | None = None,
        **mc_kwargs,
    ) -> None:
        """Initialize the Isobaric Monte Carlo object."""
        super().__init__(
            atoms, temperature, max_cycles, default_displacement_move, **mc_kwargs
        )

        self.pressure = pressure

        if default_cell_move:
            self.add_move(default_cell_move, name="default_cell_move")

        self.set_default_probability()

        if self.default_logger:
            self.default_logger.add_field("AcptRate", lambda: self.acceptance_rate)

    @property
    def pressure(self) -> float:
        """
        The pressure of the simulation.

        Returns
        -------
        float
            The pressure in eV/Å^3.
        """
        return self.context.pressure

    @pressure.setter
    def pressure(self, pressure: float) -> None:
        """
        Set the pressure of the simulation.

        Parameters
        ----------
        pressure : float
            The pressure in eV/Å^3.
        """
        self.context.pressure = pressure

    def set_default_probability(self) -> None:
        """
        Set the default probability for the cell and displacement moves.

        The probability for cell moves is set to 1/(N+1) and the probability for displacement moves
        is set to 1/(1+1/N), where N is the number of atoms.
        """
        if cell_move := self.moves.get("default_cell_move"):
            cell_move.probability = 1 / (len(self.atoms) + 1)
        if displacement_move := self.moves.get("default_displacement_move"):
            displacement_move.probability = 1 / (1 + 1 / len(self.atoms))

    def step(self) -> Generator[str, None, None]:
        """
        Perform operations before the Monte Carlo step.

        This method saves the current cell state in addition to the operations
        performed by the parent class.
        """
        self.context.last_cell = self.atoms.get_cell()
        yield from super().step()

    def save_state(self) -> None:
        """
        Save the current state of the context and update the last positions, cell, and results.

        This method saves the cell state in addition to the state saved by the parent class.
        """
        super().save_state()
        self.context.last_cell = self.atoms.get_cell()

    def revert_state(self) -> None:
        """
        Revert to the previously saved state and undo the last move.

        This method restores the cell state in addition to the state restored by the parent class.

        Raises
        ------
        AttributeError
            If the atoms object does not have a calculator attached.
        """
        super().revert_state()
        self.atoms.set_cell(self.context.last_cell, scale_atoms=False)

        try:
            self.atoms.calc.atoms.cell = self.atoms.cell.copy()  # type: ignore
        except AttributeError:
            warn("Atoms object does not have calculator attached.", stacklevel=2)
