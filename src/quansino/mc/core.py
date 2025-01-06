"""Module to run and create Monte Carlo simulations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.optimize.optimize import Dynamics
from numpy.random import PCG64, Generator

from quansino.io import Logger
from quansino.mc.contexts import DisplacementContext

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms

    from quansino.moves.core import BaseMove


class MonteCarlo(Dynamics):
    """
    Base-class for all Monte Carlo classes.

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

    Attributes
    ----------
    moves: dict[str, BaseMove]
        Dictionary of moves to perform.
    move_intervals: dict[str, int]
        Dictionary of intervals at which moves are attempted.
    move_probabilities: dict[str, float]
        Dictionary of probabilities of moves being attempted.
    move_minimum_count: dict[str, int]
        Dictionary of minimum number of times moves must be performed.
    _seed: int
        Seed for the random number generator.
    _rng: Generator
        Random number generator.

    Notes
    -----
    The Monte Carlo class provides a common interface for all Monte Carlo classes, and
    is not intended to be used directly. It is a subclass of the ASE Dynamics class,
    and inherits all of its methods and attributes.

    The Monte Carlo class is responsible for setting up the random number generator
    which is set as a private attribute. The random number generator can be accessed
    via the _MonteCarlo__rng attribute, but should not be modified directly.
    """

    context = DisplacementContext

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
        self.moves: dict[str, BaseMove] = {}
        self.move_intervals: dict[str, int] = {}
        self.move_probabilities: dict[str, float] = {}
        self.move_minimum_count: dict[str, int] = {}

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

        assert (
            self.atoms.calc is not None
        ), "Atoms object must have a calculator attached"

    def add_move(
        self,
        move: BaseMove,
        name: str = "default",
        interval: int = 1,
        probability: float = 1.0,
        minimum_count: int = 0,
    ) -> None:
        """
        Add a move to the Monte Carlo object.

        Parameters
        ----------
        move : BaseMove
            The move to add to the Monte Carlo object.
        name : str
            Name of the move. Default: 'default'.
        interval : int
            The interval at which the move is attempted. Default: 1.
        probability : float
            The probability of the move being attempted. Default: 1.0.
        minimum_count : int
            The minimum number of times the move must be performed. Default: 0.
        """
        forced_moves_total_number = sum(list(self.move_minimum_count.values()))
        assert forced_moves_total_number + minimum_count <= self.num_cycles

        move.attach_simulation(self.context(self.atoms, self._rng))

        self.moves[name] = move
        self.move_intervals[name] = interval
        self.move_probabilities[name] = probability
        self.move_minimum_count[name] = minimum_count

    def irun(self, *args, **kwargs):
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
        """
        Return a dictionary representation of the Monte Carlo object.

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
        """
        Yield moves to be performed given the move parameters.

        Yields
        ------
        bool
            Whether the move was successful or not.
        """
        available_moves = [
            name for name in self.moves if self.nsteps % self.move_intervals[name] == 0
        ]

        if not available_moves:
            yield False

        optional_moves, forced_moves = [], []

        for name in available_moves:
            if self.move_minimum_count[name] > 0:
                forced_moves.append(name)
            else:
                optional_moves.append(name)

        remaining_cycles = self.num_cycles - len(forced_moves)

        if remaining_cycles > 0 and optional_moves:
            move_probabilities = np.array(
                [self.move_probabilities[name] for name in optional_moves]
            )
            move_probabilities /= np.sum(move_probabilities)

            selected_move = self._rng.choice(
                optional_moves,  # type: ignore
                p=move_probabilities,
                size=remaining_cycles,  # type: ignore
            )

            all_moves = np.concatenate((forced_moves, selected_move))  # type: ignore
        else:
            all_moves = forced_moves

        self._rng.shuffle(all_moves)  # type: ignore

        for move in all_moves:
            yield self.moves[move]()

    def converged(self) -> bool:  # type: ignore
        """
        The Monte Carlo simulation is 'converged' when number of maximum steps is reached.

        Returns
        -------
        bool
            True if the maximum number of steps is reached.
        """
        return self.nsteps >= self.max_steps
