"""Module to perform canonical Monte Carlo simulation"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.units import kB

from quansino.mc.core import MonteCarlo
from quansino.moves.atomic import AtomicMove

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms

    from quansino.moves.base import BaseMove
    from quansino.typing import Positions


class Canonical(MonteCarlo):
    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        num_cycles: int | None = None,
        moves: list[BaseMove] | BaseMove | None = None,
        **mc_kwargs,
    ) -> None:
        """Initialize the Canonical Monte Carlo object."""
        self.temperature: float = temperature * kB

        if num_cycles is None:
            num_cycles = len(atoms)

        if moves is None:
            moves = [AtomicMove(0.1, np.arange(len(atoms)))]

        super().__init__(atoms, num_cycles=num_cycles, **mc_kwargs)

        if not isinstance(moves, list):
            moves = [moves]

        for move in moves:
            self.add_move(move)

        self.last_positions: Positions = self.atoms.get_positions()
        self.last_results: dict[str, Any] = {}

        self.acceptance_rate: float = 0

        if self.default_logger:
            self.default_logger.add_field("AcptRate", lambda: self.acceptance_rate)

    def todict(self) -> dict:
        return {"temperature": self.temperature / kB, **super().todict()}

    def get_metropolis_criteria(self, energy_difference: float) -> bool:
        return energy_difference < 0 or self._rng.random() < np.exp(
            -energy_difference / self.temperature
        )

    def step(self):  # type: ignore
        self.acceptance_rate = 0

        if not self.last_results.get("energy", None):
            self.atoms.get_potential_energy()
            self.last_results = self.atoms.calc.results  # type: ignore # if self.atoms.calc is None, get_potential_energy will raise an error anyway, so we can safely ignore this

        self.last_positions = self.atoms.get_positions()

        for move in self.yield_moves():
            if move:
                current_energy = self.atoms.get_potential_energy()
                energy_difference = current_energy - self.last_results["energy"]

                if self.get_metropolis_criteria(energy_difference):
                    self.last_positions = self.atoms.get_positions()
                    self.last_results = self.atoms.calc.results  # type: ignore # same as above
                    self.acceptance_rate += 1
                else:
                    self.atoms.positions = self.last_positions
                    self.atoms.calc.atoms.positions = self.last_positions.copy()  # type: ignore # same as above
                    self.atoms.calc.results = self.last_results  # type: ignore # same as above

        self.acceptance_rate /= self.num_cycles
