"""Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.optimize.optimize import Dynamics
from numpy.random import PCG64, Generator

from quansino.io import Logger

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms

    from quansino.moves.base import BaseMove


@dataclass
class MoveStore:
    move: BaseMove
    interval: int
    probability: float
    minimum_count: int


@dataclass
class MonteCarloContext:
    atoms: Atoms
    rng: Generator


class MonteCarlo(Dynamics):
    """Base-class for all Monte Carlo classes.

    Parameters
    ----------
    atoms: Atoms
        The Atoms object to operate on.
    seed: int
        Seed for the random number generator.
    trajectory: str | Path | None
        Trajectory file name to auto-attach. Default: None.
    logfile: str | Path | None
        Log file name to auto-attach. Default: None. Use '-' for stdout.
    append_trajectory: bool
        Defaults to False, which causes the trajectory file to be
        overwriten each time the dynamics is restarted from scratch.
        If True, the new structures are appended to the trajectory
        file instead.
    loginterval: int
        Number of steps between log entries. Default is 1.

    Notes
    -----
    The Monte Carlo class provides a common interface for all Monte Carlo classes, and
    is not intended to be used directly. It is a subclass of the ASE Dynamics class,
    and inherits all of its methods and attributes.

    The Monte Carlo class is responsible for setting up the random number generator
    which is set as a private attribute. The random number generator can be accessed
    via the _MonteCarlo__rng attribute, but should not be modified directly.
    """

    context = MonteCarloContext

    def __init__(
        self,
        atoms: Atoms,
        num_cycles=1,
        seed: int | None = None,
        trajectory: str | Path | None = None,
        logfile: str | Path | None = None,
        append_trajectory: bool = False,
        loginterval: int = 1,
    ) -> None:
        """Initialize the Monte Carlo object."""
        self.moves = {}

        self._seed = seed or PCG64().random_raw()
        self._rng = Generator(PCG64(seed))

        if isinstance(trajectory, Path):
            trajectory = str(trajectory)

        self.num_cycles = num_cycles

        Dynamics.__init__(
            self, atoms, trajectory=trajectory, append_trajectory=append_trajectory
        )

        if logfile:
            self.default_logger = Logger(logfile)
            self.default_logger.add_mc_fields(self)
            self.attach(self.closelater(self.default_logger), loginterval)
        else:
            self.default_logger = None

    def add_move(
        self,
        move: BaseMove,
        name: str = "default",
        interval: int = 1,
        probability: float = 1.0,
        minimum_count: int = 0,
    ) -> None:
        """Add a move to the Monte Carlo object.

        Parameters
        ----------
        move
            The move to add to the Monte Carlo object.
        """
        context_attributes = {field.name for field in fields(self.context)}

        missing = move.REQUIRED_ATTRIBUTES - context_attributes
        if missing:
            raise ValueError(
                f"Move {move.__class__.__name__} requires context attributes {missing} "
                f"which are not available in {self.context.__name__} of {self.__class__.__name__}"
            )

        forced_moves_total_number = sum(
            move.minimum_count for move in self.moves.values() if move.minimum_count > 0
        )
        assert forced_moves_total_number + minimum_count <= self.num_cycles

        move.context = MonteCarloContext(self.atoms, self._rng)
        self.moves[name] = MoveStore(move, interval, probability, minimum_count)

    def irun(self, *args, **kwargs) -> Generator[bool]:  # type: ignore
        """Run the Monte Carlo simulation for a given number of steps."""
        if self.default_logger:
            self.default_logger.write_header()
        return super().irun(*args, **kwargs)  # type: ignore

    def run(self, *args, **kwargs) -> bool:  # type: ignore
        """Run the Monte Carlo simulation for a given number of steps."""
        if self.default_logger:
            self.default_logger.write_header()
        return super().run(*args, **kwargs)  # type: ignore

    def todict(self) -> dict[str, Any]:
        """Return a dictionary representation of the Monte Carlo object.

        Returns
        -------
        dict
            A dictionary representation of the Monte Carlo object.
        """
        return {
            "type": "monte-carlo",
            "mc-type": self.__class__.__name__,
            "seed": self._seed,
            "rng_state": self._rng.bit_generator.state,
            "nsteps": self.nsteps,
        }

    def yield_moves(self):
        available_moves = [
            move for move in self.moves.values() if self.nsteps % move.interval == 0
        ]

        if not available_moves:
            yield False

        forced_moves = [move for move in available_moves if move.minimum_count > 0]
        optional_moves = [move for move in available_moves if move.minimum_count == 0]

        remaining_cycles = self.num_cycles - len(forced_moves)

        if remaining_cycles > 0 and optional_moves:
            move_probabilities = np.array([move.probability for move in optional_moves])
            move_probabilities /= np.sum(move_probabilities)

            selected_move = self._rng.choice(
                optional_moves, p=move_probabilities, size=remaining_cycles
            )

            all_moves = np.concatenate((forced_moves, selected_move))
        else:
            all_moves = forced_moves

        self._rng.shuffle(all_moves)

        for move in all_moves:
            yield move.move()

    def converged(self) -> bool:  # type: ignore
        """MC is 'converged' when number of maximum steps is reached.

        Returns
        -------
        bool
            True if the maximum number of steps is reached.
        """
        return self.nsteps >= self.max_steps
