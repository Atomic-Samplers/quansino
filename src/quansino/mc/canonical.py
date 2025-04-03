"""Module to perform canonical (NVT) Monte Carlo simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from warnings import warn

import numpy as np

from quansino.mc.contexts import DisplacementContext
from quansino.mc.core import MonteCarlo, MoveStorage
from quansino.mc.criteria import CanonicalCriteria
from quansino.moves.displacements import DisplacementMove
from quansino.moves.protocol import DisplacementProtocol

if TYPE_CHECKING:
    from collections.abc import Generator

    from ase.atoms import Atoms


class Canonical[MoveProtocol: DisplacementProtocol, ContextType: DisplacementContext](
    MonteCarlo[MoveProtocol, ContextType]
):
    """
    Canonical Monte Carlo simulation object.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on, will be acted upon in place.
    temperature : float
        The temperature of the simulation in Kelvin.
    num_cycles : int, optional
        The number of Monte Carlo cycles to perform, by default equal to the number of atoms.
    default_move : MoveStorage[MoveType, ContextType] | MoveType, optional
        The default move to perform in each cycle. If a `MoveStorage` object is provided, it will be used to initialize the move with its criteria and other parameters. If a `MoveType` object is provided, it will be added using the default criteria and parameters, by default None.
    **mc_kwargs
        Additional keyword arguments to pass to the [`MonteCarlo`][quansino.mc.core.MonteCarlo] parent class.

    Attributes
    ----------
    acceptable_moves : ClassVar[dict[DisplacementMove, MetropolisCriteria]]
        The acceptable moves for the simulation.
    accepted_moves : list[str]
        The names of the moves that were accepted in the last step.
    acceptance_rate : float
        The acceptance rate of the moves in the last step.
    """

    default_criteria: ClassVar = {DisplacementMove: CanonicalCriteria}
    default_context: ClassVar = DisplacementContext

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        num_cycles: int | None = None,
        default_move: MoveStorage[MoveProtocol] | MoveProtocol | None = None,
        **mc_kwargs,
    ) -> None:
        """Initialize the Canonical Monte Carlo object."""
        if num_cycles is None:
            num_cycles = len(atoms)

        super().__init__(atoms, num_cycles=num_cycles, **mc_kwargs)

        self.context.temperature = temperature

        if isinstance(default_move, DisplacementMove):
            self.add_move(default_move, name="default_move")
        elif isinstance(default_move, MoveStorage):
            self.add_move(
                default_move.move,
                default_move.criteria,
                "default_move",
                default_move.interval,
                default_move.probability,
                default_move.minimum_count,
            )

        self.last_results: dict[str, Any] = {}

        if self.default_logger:
            self.default_logger.add_field("AcptRate", lambda: self.acceptance_rate)

    @property
    def temperature(self) -> float:
        """The temperature of the simulation in Kelvin."""
        return self.context.temperature

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        """Set the temperature of the simulation in Kelvin."""
        self.context.temperature = temperature

    def calculate_energy_difference(self) -> float:
        """
        Calculate the energy difference between the current and last state.

        Returns
        -------
        float
            The energy difference.
        """
        return self.atoms.get_potential_energy() - self.last_results["energy"]

    def yield_moves(self) -> Generator[str, None, None]:
        """
        Yield the names of accepted moves after evaluating their acceptance criteria.

        Yields
        ------
        Generator[str, None, None]
            The names of the accepted moves.
        """
        for move_name in super().yield_moves():
            move_storage = self.moves[move_name]
            move = move_storage.move

            if move():
                is_accepted = move_storage.criteria.evaluate(self.context)

                if is_accepted:
                    self.save_state()
                    yield move_name
                else:
                    self.revert_state()

    def pre_step(self) -> None:
        """Perform operations before the Monte Carlo step."""
        if np.isnan(self.context.last_energy):
            self.context.last_energy = self.atoms.get_potential_energy()
            self.last_results = self.atoms.calc.results  # type: ignore

    def step(self) -> Any:
        """Perform a single Monte Carlo step, iterating over all selected moves."""
        self.pre_step()
        self.accepted_moves = list(self.yield_moves())
        self.acceptance_rate = len(self.accepted_moves) / self.num_cycles

    def save_state(self) -> None:
        """Save the current state of the context and update the last positions and results."""
        self.context.last_positions = self.atoms.get_positions()
        try:
            self.last_results = self.atoms.calc.results  # type: ignore
        except AttributeError:
            warn(
                "Atoms object does not have calculator results attached.", stacklevel=2
            )
            self.last_results = {}

    def revert_state(self) -> None:
        """Revert to the previously saved state and undo the last move."""
        self.atoms.positions = self.context.last_positions
        try:
            self.atoms.calc.atoms = self.atoms  # type: ignore
            self.atoms.calc.results = self.last_results  # type: ignore
        except AttributeError:
            warn("Atoms object does not have calculator attached.", stacklevel=2)
