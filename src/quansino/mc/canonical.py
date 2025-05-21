"""Module to perform canonical (NVT) Monte Carlo simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from warnings import warn

import numpy as np

from quansino.mc.contexts import DisplacementContext
from quansino.mc.core import MonteCarlo, MoveStorage
from quansino.mc.criteria import CanonicalCriteria
from quansino.moves.displacement import DisplacementMove
from quansino.moves.protocol import DisplacementProtocol

if TYPE_CHECKING:
    from collections.abc import Generator

    from ase.atoms import Atoms


class Canonical[MoveProtocol: DisplacementProtocol, ContextType: DisplacementContext](
    MonteCarlo[MoveProtocol, ContextType]
):
    """
    Canonical Monte Carlo simulation object for performing NVT simulations. This class is a subclass of the [`MonteCarlo`][quansino.mc.core.MonteCarlo] class and provides additional functionality specific to canonical simulations. By default, it uses the [`DisplacementContext`][quansino.mc.contexts.DisplacementContext] context and the [`CanonicalCriteria`][quansino.mc.criteria.CanonicalCriteria] criteria to perform the simulation.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on, will be acted upon in place.
    temperature : float
        The temperature of the simulation in Kelvin.
    max_cycles : int, optional
        The number of Monte Carlo cycles to perform, by default equal to the number of atoms.
    default_displacement_move : MoveStorage[MoveProtocol] | MoveProtocol, optional
        The default displacement move to perform in each cycle. If a `MoveStorage` object is provided, it will be used to initialize the move with its criteria and other parameters. If a `MoveProtocol` object is provided, it will be added using the default criteria and parameters, by default None.
    **mc_kwargs
        Additional keyword arguments to pass to the [`MonteCarlo`][quansino.mc.core.MonteCarlo] parent class.

    Attributes
    ----------
    accepted_moves : list[str]
        The names of the moves that were accepted in the last step.
    acceptance_rate : float
        The acceptance rate of the moves in the last step.
    temperature : float
        The temperature of the simulation in Kelvin.
    last_results : dict
        The last results from the calculator attached to the atoms object.
    last_positions : Positions
        The last positions of the atoms in the simulation.
    """

    default_criteria: ClassVar = {DisplacementMove: CanonicalCriteria}
    default_context: ClassVar = DisplacementContext

    def __init__(
        self,
        atoms: Atoms,
        temperature: float = 298.15,
        max_cycles: int | None = None,
        default_displacement_move: (
            MoveStorage[MoveProtocol] | MoveProtocol | None
        ) = None,
        **mc_kwargs: Any,
    ) -> None:
        """Initialize the Canonical Monte Carlo object."""
        if max_cycles is None:
            max_cycles = len(atoms)

        super().__init__(atoms, max_cycles=max_cycles, **mc_kwargs)

        self.temperature = temperature

        self.acceptance_rate: float = 0.0
        self.performed_moves: list[tuple[str, bool]] = []

        if default_displacement_move:
            self.add_move(default_displacement_move, name="default_displacement_move")

        if self.default_logger:
            self.default_logger.add_field("AcptRate", lambda: self.acceptance_rate)

    @property
    def temperature(self) -> float:
        """
        The temperature of the simulation in Kelvin, retrieved from the context.

        Returns
        -------
        float
            The temperature in Kelvin.
        """
        return self.context.temperature

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        """
        Set the temperature of the simulation in Kelvin, updating the context.

        Parameters
        ----------
        temperature : float
            The temperature in Kelvin.
        """
        self.context.temperature = temperature

    def step(self) -> Generator[str, None, None]:
        """Perform a single Monte Carlo step, iterating over all selected moves in [`yield_moves`][quansino.mc.canonical.Canonical.yield_moves]."""
        self.context.last_positions = self.atoms.get_positions()

        if np.isnan(self.context.last_energy):
            self.context.last_energy = self.atoms.get_potential_energy()

        yield from super().step()

    def save_state(self) -> None:
        """Save the current state of the context and update the last positions and results."""
        self.context.last_positions = self.atoms.get_positions()
        self.context.last_energy = self.atoms.get_potential_energy()

        super().save_state()

    def revert_state(self) -> None:
        """Revert to the previously saved state and undo the last move by restoring the last positions."""
        super().revert_state()

        self.atoms.positions = self.context.last_positions.copy()

        try:
            self.atoms.calc.atoms.positions = self.atoms.positions.copy()  # type: ignore
        except AttributeError:
            warn(
                "The calculator does not support restoring positions. Please check that your calculator is fully compatible with quansino.",
                UserWarning,
                2,
            )
