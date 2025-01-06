"""Module to perform canonical (NVT) Monte Carlo simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.units import kB

from quansino.mc.core import MonteCarlo
from quansino.moves.core import BaseMove
from quansino.moves.displacements import DisplacementMove

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms

    from quansino.typing import Positions


class Canonical(MonteCarlo):
    """Canonical Monte Carlo simulation object.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on, will be act upon in place.
    temperature : float
        The temperature of the simulation in Kelvin.
    num_cycles : int, optional
        The number of Monte Carlo cycles to perform, by default equal to the number of atoms.
    moves : list[BaseMove] | BaseMove, optional
        The moves to perform in each cycle, by default a single DisplacementMove acting on all atoms. The move provided should be a subclass of [`BaseMove`][quansino.moves.core.BaseMove] and will have an equal probability of being selected and run at each cycle. To modify the probability of a move being selected, either use the [`add_move`][quansino.mc.core.MonteCarlo.add_move] method or modify the `move_probabilities` attribute.
    **mc_kwargs
        Additional keyword arguments to pass to the MonteCarlo superclass. See [`MonteCarlo`][quansino.mc.core.MonteCarlo] for more information.

    Attributes
    ----------
    temperature : float
        The temperature of the simulation in Kelvin.
    acceptance_rate : float
        The acceptance rate of the moves performed in the last cycle.
    last_positions : Positions
        The positions of the atoms since the last accepted move.
    last_results : dict[str, Any]
        The results of the atoms since the last accepted move.

    Notes
    -----
    This Class assumes that the atoms object has a calculator attached to it, and possess the `atoms` and `results` attributes.
    """

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        num_cycles: int | None = None,
        moves: dict[str, BaseMove] | list[BaseMove] | BaseMove | None = None,
        **mc_kwargs,
    ) -> None:
        """Initialize the Canonical Monte Carlo object."""
        self.temperature: float = temperature

        if num_cycles is None:
            num_cycles = len(atoms)

        if moves is None:
            moves = {
                "default": DisplacementMove(candidate_indices=np.arange(len(atoms)))
            }

        super().__init__(atoms, num_cycles=num_cycles, **mc_kwargs)

        if isinstance(moves, BaseMove):
            moves = [moves]
        if isinstance(moves, list):
            moves = {
                f"{move.__class__.__name__}_{index}": move
                for index, move in enumerate(moves)
            }

        for name, move in moves.items():
            self.add_move(move, name=name)

        self.last_positions: Positions = self.atoms.get_positions()
        self.last_results: dict[str, Any] = {}

        self.acceptance_rate: float = 0

        if self.default_logger:
            self.default_logger.add_field("AcptRate", lambda: self.acceptance_rate)

    def todict(self) -> dict:
        """Return a dictionary representation of the object."""
        return {"temperature": self.temperature, **super().todict()}

    def get_metropolis_criteria(self, energy_difference: float) -> bool:
        """Return whether the move should be accepted based on the Metropolis criteria.

        Parameters
        ----------
        energy_difference : float
            The difference in energy between the current and the previous state.
        """
        return energy_difference < 0 or self._rng.random() < np.exp(
            -energy_difference / (self.temperature * kB)
        )

    def step(self):  # type: ignore
        """Perform a single Monte Carlo step, iterating over all selected moves. Due to the caching performed by ASE in calculators, the atoms object of the calculator is updated as well."""
        self.acceptance_rate = 0

        if not self.last_results.get("energy", None):
            self.atoms.get_potential_energy()
            self.last_results = self.atoms.calc.results  # type: ignore

        self.last_positions = self.atoms.get_positions()

        for move in self.yield_moves():
            if move:
                self.current_energy = self.atoms.get_potential_energy()
                energy_difference = self.current_energy - self.last_results["energy"]

                if self.get_metropolis_criteria(energy_difference):
                    self.last_positions = self.atoms.get_positions()
                    self.last_results = self.atoms.calc.results  # type: ignore # same as above
                    self.acceptance_rate += 1
                else:
                    self.atoms.positions = self.last_positions
                    self.atoms.calc.atoms.positions = self.last_positions  # type: ignore # same as above # copy needed ?
                    self.atoms.calc.results = self.last_results  # type: ignore # same as above

        self.acceptance_rate /= self.num_cycles
