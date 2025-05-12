"""Module to run and create Monte Carlo simulations."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, cast
from warnings import warn

import numpy as np
from numpy.random import PCG64
from numpy.random import Generator as RNG

from quansino.io import Logger
from quansino.io.restart import RestartObserver
from quansino.io.trajectory import TrajectoryObserver
from quansino.mc.contexts import Context
from quansino.mc.criteria import Criteria
from quansino.moves import moves_registry
from quansino.moves.protocol import BaseProtocol

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from ase.atoms import Atoms

    from quansino.io.core import Observer


class Driver:
    """
    Base class for all Monte Carlo drivers. This class is not intended to be used directly, but rather as a base class for other Monte Carlo classes. It provides the basic functionality for running a Monte Carlo simulation, including the ability to attach a calculator to the atoms object, and to write the results to a file.
    """

    def __init__(
        self,
        atoms: Atoms,
        logfile: Logger | Path | str | None = None,
        trajectory: TrajectoryObserver | Path | str | None = None,
        restart_file: RestartObserver | Path | str | None = None,
        logging_interval: int = 1,
        logging_mode: str = "a",
    ) -> None:
        """
        Dynamics object.

        Parameters
        ----------
        atoms : Atoms object
            The Atoms object to operate on.

        logfile : file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory : Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        append_trajectory : bool
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.

        master : bool
            Defaults to None, which causes only rank 0 to save files. If set to
            true, this rank will save files.

        comm : Communicator object
            Communicator to handle parallel file reading and writing.

        loginterval : int, default: 1
            Only write a log line for every *loginterval* time steps.
        """
        self.atoms = atoms

        self.observers: dict[str, Observer] = {}

        self.logging_interval = logging_interval
        self.logging_mode = logging_mode

        self.default_logger = logfile
        self.default_trajectory = trajectory
        self.default_restart = restart_file

        self.nsteps: int = 0
        self.max_steps: int = 0

    def attach_observer(self, name: str, observer: Observer):
        """
        Attach callback function.

        If *interval > 0*, at every *interval* steps, call *function* with
        arguments *args* and keyword arguments *kwargs*.

        If *interval <= 0*, after step *interval*, call *function* with
        arguments *args* and keyword arguments *kwargs*.  This is
        currently zero indexed."""

        self.observers[name] = observer

    def detach_observer(self, name: str) -> None:
        """Detach callback function."""
        if self.observers.pop(name, None) is None:
            warn(f"Observer {name} not found.", FutureWarning, 2)

    def call_observers(self) -> None:
        for observer in self.observers.values():
            interval = observer.interval
            if (interval > 0 and self.nsteps % interval == 0) or (
                interval < 0 and self.nsteps == abs(interval)
            ):
                observer()

    def to_dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the Driver object.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the Driver object.
        """
        return {
            "name": self.__class__.__name__,
            "nsteps": self.nsteps,
            "max_steps": self.max_steps,
            "logging_interval": self.logging_interval,
            "logging_mode": self.logging_mode,
            "default_logger": (
                None if self._default_logger is None else str(self._default_logger)
            ),
            "default_trajectory": (
                None
                if self._default_trajectory is None
                else str(self._default_trajectory)
            ),
            "default_restart": (
                None if self._default_restart is None else str(self._default_restart)
            ),
        }

    @property
    def default_trajectory(self) -> TrajectoryObserver | None:
        """Return the trajectory object."""
        return self._default_trajectory

    @default_trajectory.setter
    def default_trajectory(
        self, default_trajectory: TrajectoryObserver | Path | str | None
    ) -> None:
        """Set the trajectory object."""
        if default_trajectory is None:
            self._default_trajectory = None
            return
        elif isinstance(default_trajectory, Path):
            default_trajectory = str(default_trajectory)

        if isinstance(default_trajectory, str):
            self._default_trajectory = TrajectoryObserver(
                self.atoms, default_trajectory, self.logging_interval, self.logging_mode
            )
        else:
            self._default_trajectory = default_trajectory

        self.attach_observer("default_trajectory", self._default_trajectory)

    @property
    def default_logger(self) -> Logger | None:
        return self._default_logger

    @default_logger.setter
    def default_logger(self, default_logger: Logger | Path | str | None) -> None:
        """Set the default logger object."""
        if default_logger is None:
            self._default_logger = None
            return
        elif isinstance(default_logger, Path):
            default_logger = str(default_logger)

        if isinstance(default_logger, str):
            self._default_logger = Logger(
                default_logger, self.logging_interval, self.logging_mode
            )
        else:
            self._default_logger = default_logger

        self.attach_observer("default_logger", self._default_logger)

    @property
    def default_restart(self) -> RestartObserver | None:
        """Return the restart file object."""
        return self._default_restart

    @default_restart.setter
    def default_restart(
        self, restart_observer: RestartObserver | Path | str | None
    ) -> None:
        """Set the restart file object."""
        if restart_observer is None:
            self._default_restart = None
            return
        elif isinstance(restart_observer, Path):
            restart_observer = str(restart_observer)

        if isinstance(restart_observer, str):
            self._default_restart = RestartObserver(
                self, restart_observer, self.logging_interval, self.logging_mode
            )
        else:
            self._default_restart = restart_observer

        self.attach_observer("default_restart", self._default_restart)


class MonteCarlo[MoveProtocol: BaseProtocol, ContextType: Context](Driver):
    """
    Base-class for all Monte Carlo classes. This is a common interface for all Monte Carlo classes, and
    is not intended to be used directly. It is a subclass of the ASE Dynamics class,
    and inherits all of its methods and attributes. The Monte Carlo class is responsible for selecting moves to perform via the [`yield_moves`][quansino.mc.core.MonteCarlo.yield_moves] method. This class is also responsible for managing the moves, their parameters (interval, probability, minimum count), and their acceptance criteria. Logging and trajectory writing are managed by the ASE parent class, Dynamics. When necessary, communication between the Monte Carlo simulation and the moves is facilitated by the context object. The Monte Carlo class and its subclasses should NOT directly modify the moves, but rather interact using the context object.

    Parameters
    ----------
    atoms: Atoms
        The Atoms object to operate on.
    max_cycles: int
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
    log_interval: int
        Number of steps between log entries.

    Attributes
    ----------
    moves: dict[str, MoveStorage[MoveProtocol]]
        Dictionary of moves to perform.
    _seed: int
        Seed for the random number generator.
    _rng: Generator
        Random number generator.
    max_cycles: int
        Number of Monte Carlo cycles per step.
    current_yield_index: int
        Current index of the yield_moves generator.
    default_logger: Logger | None
        Default logger object.
    context: ContextType
        Context object for the simulation used to store the state of the simulation and provide information to the moves/criteria.
    default_criteria: ClassVar[dict[type[BaseProtocol], type[Criteria]]]
        Dictionary mapping move types to their default criteria classes.
    default_context: ClassVar[type[Context]]
        The default context type for this Monte Carlo simulation.
    """

    default_criteria: ClassVar[dict[type[BaseProtocol], type[Criteria]]] = {}
    default_context: ClassVar[type[Context]] = Context

    def __init__(
        self,
        atoms: Atoms,
        max_cycles: int = 1,
        seed: int | None = None,
        trajectory: TrajectoryObserver | Path | str | None = None,
        logfile: Logger | Path | str | None = None,
        restart_file: RestartObserver | Path | str | None = None,
        logging_interval: int = 1,
        logging_mode: str = "a",
        **driver_kwargs,
    ) -> None:
        """Initialize the MonteCarlo object."""
        self.moves: dict[str, MoveStorage[MoveProtocol]] = {}

        self._seed = seed or PCG64().random_raw()
        self._rng = RNG(PCG64(seed))

        self.max_cycles = max_cycles

        self.last_results: dict[str, Any] = {}

        super().__init__(
            atoms,
            logfile=logfile,
            trajectory=trajectory,
            restart_file=restart_file,
            logging_interval=logging_interval,
            logging_mode=logging_mode,
            **driver_kwargs,
        )

        if self.default_logger:
            self.default_logger.add_mc_fields(self)

        self.context = self.create_context(atoms, self._rng)

    def add_move(
        self,
        move: MoveProtocol | MoveStorage[MoveProtocol],
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
        forced_moves_count = sum(
            [self.moves[name].minimum_count for name in self.moves]
        )

        if forced_moves_count + minimum_count > self.max_cycles:
            raise ValueError("The number of forced moves exceeds the number of cycles.")

        if not isinstance(move, MoveStorage):
            if criteria is None:
                for acceptable_move in self.default_criteria:
                    if isinstance(move, acceptable_move):
                        criteria = self.default_criteria[acceptable_move]()
                        break

                if criteria is None:
                    raise ValueError(
                        "No acceptance criteria found for the move, please provide one."
                    )

            move_storage = MoveStorage[MoveProtocol](
                move=move,
                interval=interval,
                probability=probability,
                minimum_count=minimum_count,
                criteria=criteria,
            )
        else:
            move_storage = move

        move_storage.move.attach_simulation(self.context)
        self.moves[name] = move_storage

    def irun(self, steps=100_000_000) -> Generator[bool, None, None]:  # type: ignore
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

        self.max_steps = self.nsteps + steps

        self.atoms.get_potential_energy()
        self.call_observers()

        is_converged = self.converged()
        yield is_converged

        while not is_converged and self.nsteps < self.max_steps:
            self.step()
            self.nsteps += 1

            self.atoms.get_potential_energy()
            self.call_observers()

            is_converged = self.converged()
            yield is_converged

    def step(self) -> Any: ...  # type: ignore

    def run(self, steps=100_000_000) -> bool:  # type: ignore
        """
        Run the Monte Carlo simulation for a given number of steps.

        Returns
        -------
        bool
            True if the simulation is converged.
        """
        return list(self.irun(steps=steps))[-1]

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
            **super().to_dict(),
            "name": self.__class__.__name__,
            "atoms": self.atoms.copy(),
            "context": self.context.to_dict(),
            "max_cycles": self.max_cycles,
            "seed": self._seed,
            "rng_state": self._rng.bit_generator.state,
            "last_results": self.last_results,
            "moves": {
                name: move_storage.to_dict()
                for name, move_storage in self.moves.items()
            },
        }

    todict = to_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Load the Monte Carlo object from a dictionary.
        This method is used to restore the state of the Monte Carlo object from a saved state.
        It is not intended to be used directly, but rather as a helper method for the `from_dict` method of the ASE Dynamics class.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary representation of the Monte Carlo object.

        Returns
        -------
        Self
            A Monte Carlo object.
        """
        mc = cls(
            atoms=data["atoms"],
            max_cycles=data["max_cycles"],
            seed=data["seed"],
            trajectory=data["default_trajectory"],
            logfile=data["default_logger"],
            restart_file=data["default_restart"],
            logging_interval=data["logging_interval"],
            logging_mode=data["logging_mode"],
        )

        mc.nsteps = data["nsteps"]
        mc._rng.bit_generator.state = data["rng_state"]
        mc.last_results = data["last_results"]

        for key, value in data["context"].items():
            setattr(mc.context, key, value)

        for name, move_data in data["moves"].items():
            move_storage = MoveStorage[MoveProtocol].from_dict(move_data)
            mc.add_move(move=move_storage, name=name)

        return mc

    fromdict = from_dict

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
            np.arange(self.max_cycles), size=len(forced_moves), replace=False
        )
        forced_moves_mapping = dict(zip(forced_moves_index, forced_moves, strict=True))

        for index in range(self.max_cycles):
            if index in forced_moves_mapping:
                yield forced_moves_mapping[index]
            else:
                move_probabilities = np.array(
                    [self.moves[name].probability for name in available_moves]
                )
                move_probabilities /= np.sum(move_probabilities)

                selected_move = self._rng.choice(available_moves, p=move_probabilities)

                yield selected_move

    def converged(self) -> bool:
        """
        The Monte Carlo simulation is 'converged' when number of maximum steps is reached.

        Returns
        -------
        bool
            True if the maximum number of steps is reached.
        """
        return self.nsteps >= self.max_steps

    def __repr__(self) -> str:
        """
        Return a string representation of the Monte Carlo object.

        Returns
        -------
        str
            A string representation of the Monte Carlo object.
        """
        return f"MonteCarlo(atoms={self.atoms}, max_cycles={self.max_cycles}, seed={self._seed}, moves={self.moves}, nsteps={self.nsteps}, default_logger={self.default_logger}, default_trajectory={self.default_trajectory}, default_restart={self.default_restart})"


@dataclass
class MoveStorage[MoveProtocol: BaseProtocol]:
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
    criteria: Criteria
        The acceptance criteria for the move.
    """

    move: MoveProtocol
    interval: int
    probability: float
    minimum_count: int
    criteria: Criteria

    def to_dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the MoveStorage object.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the MoveStorage object.
        """
        return {
            "move": self.move.to_dict(),
            "interval": self.interval,
            "probability": self.probability,
            "minimum_count": self.minimum_count,
            "criteria": self.criteria.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoveStorage[MoveProtocol]:
        """
        Load the MoveStorage object from a dictionary.
        This method is used to restore the state of the MoveStorage object from a saved state.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary representation of the MoveStorage object.
        """
        move_data = data["move"]
        move = cast(
            MoveProtocol, moves_registry[move_data["name"]].from_dict(move_data)
        )
        criteria = Criteria.from_dict(data["criteria"])

        return cls(
            move=move,
            interval=data["interval"],
            probability=data["probability"],
            minimum_count=data["minimum_count"],
            criteria=criteria,
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the MoveStorage object.

        Returns
        -------
        str
            A string representation of the MoveStorage object.
        """
        return f"MoveStorage(move={self.move}, interval={self.interval}, probability={self.probability}, minimum_count={self.minimum_count}, criteria={self.criteria})"
