"""Module to run and create Monte Carlo simulations."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from ase.optimize.optimize import Dynamics
from numpy.random import PCG64
from numpy.random import Generator as RNG

from quansino.io import Logger
from quansino.mc.contexts import Context
from quansino.moves.protocol import BaseProtocol

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from ase.atoms import Atoms

    from quansino.mc.criteria import Criteria


class MonteCarlo[MoveProtocol: BaseProtocol, ContextType: Context](Dynamics):
    """
    Base-class for all Monte Carlo classes. This is a common interface for all Monte Carlo classes, and
    is not intended to be used directly. It is a subclass of the ASE Dynamics class,
    and inherits all of its methods and attributes. The Monte Carlo class is responsible for selecting moves to perform via the [`yield_moves`][quansino.mc.core.MonteCarlo.yield_moves] method. This class is also responsible for managing the moves, their parameters (interval, probability, minimum count), and their acceptance criteria. Logging and trajectory writing are managed by the ASE parent class, Dynamics. When necessary, communication between the Monte Carlo simulation and the moves is facilitated by the context object. The Monte Carlo class and its subclasses should NOT directly modify the moves, but rather interact using the context object. The generics MoveProtocol and ContextType are used to specify the type of move and context object, respectively. Moves themselves use a ContextType generics, the non-interaction between the Monte Carlo and the moves is enforced via the missing ContextType generics in the MoveProtocol generics, i.e. The MonteCarlo class does not know what the context object of its moves is, only that it exists.

    Parameters
    ----------
    atoms: Atoms
        The Atoms object to operate on.
    num_cycles: int
        Number of Monte Carlo cycles per step.
    seed: int
        Seed for the random number generator.
    trajectory: str | Path | None
        Trajectory file name to auto-attach.
    logfile: str | Path | None
        Log file name to auto-attach. Use '-' for stdout.
    append_trajectory: bool
        If false, causes the trajectory file to be
        overwriten each time the dynamics is restarted from scratch.
        If True, the new structures are appended to the trajectory
        file instead.
    loginterval: int
        Number of steps between log entries.

    Attributes
    ----------
    moves: dict[str, MoveStorage[MoveProtocol, ContextType]]
        Dictionary of moves to perform.
    _seed: int
        Seed for the random number generator.
    _rng: Generator
        Random number generator.
    num_cycles: int
        Number of Monte Carlo cycles per step.
    default_logger: Logger | None
        Default logger object.
    context: ContextType
        Context object for the simulation used to store the state of the simulation and provide information to the moves/criteria.
    """

    default_criteria: ClassVar[dict[type[BaseProtocol], type[Criteria]]] = {}
    default_context: ClassVar[type[Context]] = Context

    def __init__(
        self,
        atoms: Atoms,
        num_cycles: int = 1,
        seed: int | None = None,
        trajectory: str | Path | None = None,
        logfile: str | Path | None = None,
        append_trajectory: bool = False,
        loginterval: int = 1,
    ) -> None:
        """Initialize the MonteCarlo object."""
        self.moves: dict[str, MoveStorage[MoveProtocol]] = {}

        self._seed = seed or PCG64().random_raw()
        self._rng = RNG(PCG64(seed))

        if isinstance(trajectory, Path):
            trajectory = str(trajectory)

        self.num_cycles = num_cycles

        super().__init__(
            atoms, trajectory=trajectory, append_trajectory=append_trajectory
        )

        if logfile:
            self.default_logger = Logger(logfile)
            self.default_logger.add_mc_fields(self)
            self.attach(self.closelater(self.default_logger), loginterval)
        else:
            self.default_logger = None

        self.context = self.create_context(atoms, self._rng)

    def add_move(
        self,
        move: MoveProtocol,
        criteria: Criteria | None = None,
        name: str = "default",
        interval: int = 1,
        probability: float = 1.0,
        minimum_count: int = 0,
    ) -> None:
        """
        Add a move to the Monte Carlo object.

        Parameters
        ----------
        move : MoveProtocol
            The move to add to the Monte Carlo object.
        criteria : Criteria, optional
            The acceptance criteria for the move. If none, the move must be an instance of a known move type.
        name : str
            Name of the move.
        interval : int
            The interval at which the move is attempted.
        probability : float
            The probability of the move being attempted.
        minimum_count : int
            The minimum number of times the move must be performed.
        """
        if criteria is None:
            for acceptable_move in self.default_criteria:
                if isinstance(move, acceptable_move):
                    criteria = self.default_criteria[acceptable_move]()
                    break

            if criteria is None:
                raise ValueError(
                    "No acceptance criteria found for the move. Please provide one."
                )

        forced_moves_count = sum(
            [self.moves[name].minimum_count for name in self.moves]
        )

        if forced_moves_count + minimum_count > self.num_cycles:
            raise ValueError("The number of forced moves exceeds the number of cycles.")

        move.attach_simulation(self.context)

        self.moves[name] = MoveStorage[MoveProtocol](
            move=move,
            interval=interval,
            probability=probability,
            minimum_count=minimum_count,
            criteria=criteria,
        )

    def irun(self, *args, **kwargs) -> Generator[bool, None, None]:  # type: ignore
        """
        Run the Monte Carlo simulation for a given number of steps.

        Returns
        -------
        Generator[bool, None, None]
            Generator that yields True if the simulation is converged.
        """
        if self.default_logger:
            self.default_logger.write_header()

        self.validate_simulation()

        return super().irun(*args, **kwargs)

    def run(self, *args, **kwargs) -> bool:  # type: ignore
        """
        Run the Monte Carlo simulation for a given number of steps.

        Returns
        -------
        bool
            True if the simulation is converged.
        """
        if self.default_logger:
            self.default_logger.write_header()

        self.validate_simulation()

        return super().run(*args, **kwargs)

    def create_context(self, atoms: Atoms, rng: RNG) -> ContextType:
        return cast(ContextType, self.default_context(atoms, rng))

    def validate_simulation(self) -> None:
        """Validate the simulation object by checking if the atoms object has a calculator attached to it."""
        if self.atoms.calc is None:
            raise AttributeError("Atoms object must have a calculator attached to it.")
        if not hasattr(self.atoms.calc, "results"):
            raise AttributeError("Calculator object must have a results attribute.")

    def to_dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the Monte Carlo object.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the Monte Carlo object.
        """
        return {
            "type": "monte-carlo",
            "mc-type": self.__class__.__name__,
            "seed": self._seed,
            "rng_state": self._rng.bit_generator.state,
            "nsteps": self.nsteps,
        }

    def todict(self):
        return self.to_dict()

    def yield_moves(self) -> Generator[str, None, None]:
        """
        Yield moves to be performed given the move parameters. The moves are selected based on the probability of the move and the interval at which the move is attempted. Forced moves are introduced based on their minimum count. moves are yielded separately, re-constructing the move_probabilities array each time, allowing for a dynamic change in the probability of moves.

        Yields
        ------
        Generator[str, None, None]
            The name of the move to be performed.
        """
        available_moves: list[str] = [
            name for name in self.moves if self.nsteps % self.moves[name].interval == 0
        ]

        if not available_moves:
            return

        counts = [self.moves[name].minimum_count for name in available_moves]
        forced_moves = np.repeat(available_moves, counts)
        forced_moves_index = self._rng.choice(
            np.arange(self.num_cycles), size=len(forced_moves), replace=False
        )
        forced_moves_mapping = dict(zip(forced_moves_index, forced_moves, strict=True))

        for index in range(self.num_cycles):
            if index in forced_moves_mapping:
                yield forced_moves_mapping[index]
            else:
                move_probabilities = np.array(
                    [self.moves[name].probability for name in available_moves]
                )
                move_probabilities /= np.sum(move_probabilities)

                selected_move = self._rng.choice(available_moves, p=move_probabilities)

                yield selected_move

    def converged(self) -> bool:  # type: ignore
        """
        The Monte Carlo simulation is 'converged' when number of maximum steps is reached.

        Returns
        -------
        bool
            True if the maximum number of steps is reached.
        """
        return self.nsteps >= self.max_steps


@dataclass
class MoveStorage[MoveProtocol]:
    """
    Dataclass to store the moves and their acceptance criteria.

    Attributes
    ----------
    move: MoveProtocol
        The move object.
    interval: int
        The interval at which the move is selected.
    probability: float
        The probability of the move being selected.
    minimum_count: int
        The minimum number of times the move must be performed in a cycle.
    criteria: AcceptanceCriteria
        The acceptance criteria for the move.
    """

    move: MoveProtocol
    interval: int
    probability: float
    minimum_count: int
    criteria: Criteria
