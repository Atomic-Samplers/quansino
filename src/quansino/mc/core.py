"""Module to run and create Monte Carlo simulations."""

from __future__ import annotations

from collections.abc import Generator
from copy import deepcopy
from typing import IO, TYPE_CHECKING, ClassVar, Final, Self, cast
from warnings import warn

import numpy as np
from numpy.random import PCG64
from numpy.random import Generator as RNG

from quansino.mc.contexts import Context
from quansino.mc.criteria import BaseCriteria
from quansino.mc.driver import Driver
from quansino.moves.core import BaseMove
from quansino.protocols import Criteria, Move
from quansino.registry import get_typed_class
from quansino.utils.moves import MoveStorage

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from typing import Any

    from ase.atoms import Atoms

    from quansino.io.logger import Logger
    from quansino.io.restart import RestartObserver
    from quansino.io.trajectory import TrajectoryObserver


class MonteCarlo[MoveType: Move, CriteriaType: Criteria](Driver):
    """
    Base class providing an interface for all Monte Carlo classes. The `MonteCarlo` class is responsible for selecting moves to perform via the [`yield_moves`][quansino.mc.core.MonteCarlo.yield_moves] method. This class is also responsible for managing the moves, their parameters (interval, probability, minimum count), and their acceptance criteria. Logging and trajectory writing are handled by the parent [`Driver`][quansino.mc.driver.Driver] class. When necessary, communication between the Monte Carlo simulation and the moves is facilitated by the context object. The Monte Carlo class and its subclasses should not directly modify the moves, but rather interact using the context object.

    Parameters
    ----------
    atoms: Atoms
        The Atoms object to operate on.
    max_cycles: int, optional
        Number of Monte Carlo cycles per step, by default 1.
    seed: int | None, optional
        Seed for the random number generator, by default None. If None, a random seed is generated.
    logfile: Logger | IO | Path | str | None, optional
        Logger observer to auto-attach, by default None.
    trajectory: TrajectoryObserver | IO | Path | str | None, optional
        Trajectory observer to auto-attach, by default None.
    restart_file: RestartObserver | IO | Path | str | None, optional
        Restart observer to auto-attach, by default None.
    logging_interval: int, optional
        Interval at which to call the observers, by default 1.
    logging_mode: str, optional
        Mode in which to open the observers, by default "a".

    Attributes
    ----------
    __seed: Final[int]
        The seed used for the random number generator, initialized to a random value if not provided.
    _rng: Generator
        Random number generator.
    acceptance_rate: float
        The acceptance rate of the moves in the simulation, initialized to 0.0.
    context: Context
        Context object for the simulation used to store the state of the simulation and provide information to the moves/criteria.
    default_logger: Logger | None
        Default logger object.
    default_criteria: ClassVar[dict[type[MoveProtocol], type[Criteria]]]
        Dictionary mapping move types to their default criteria classes.
    default_context: ClassVar[type[Context]]
        The default context type for this Monte Carlo simulation.
    last_results: dict[str, Any]
        The last results of the simulation, typically the calculator results from the Atoms object.
    max_cycles: int
        Number of Monte Carlo cycles per step.
    move_history: list[tuple[str, bool | None]]
        History of moves performed in the current step, where each entry is a tuple of the move name and whether it was accepted (True), rejected (False), or not attempted (None).
    moves: dict[str, MoveStorage[MoveType, CriteriaType]]
        A dictionary of moves to perform, where the key is the name of the move and the value is a `MoveStorage` object containing the move, its criteria, interval, probability, and minimum count.
    step_count: int
        The number of steps performed in the simulation, initialized to 0.
    """

    default_criteria: ClassVar[dict[type[Move], type[Criteria]]] = {
        BaseMove: BaseCriteria
    }
    default_context: ClassVar[type[Context]] = Context

    def __init__(
        self,
        atoms: Atoms,
        max_cycles: int = 1,
        seed: int | None = None,
        logfile: Logger | IO | Path | str | None = None,
        trajectory: TrajectoryObserver | IO | Path | str | None = None,
        restart_file: RestartObserver | IO | Path | str | None = None,
        logging_interval: int = 1,
        logging_mode: str = "a",
    ) -> None:
        """Initialize the MonteCarlo object."""
        self.moves: dict[str, MoveStorage[MoveType, CriteriaType]] = {}

        self.__seed: Final = seed or PCG64().random_raw()
        self._rng = RNG(PCG64(self.__seed))

        self.max_cycles = max_cycles

        self.last_results: dict[str, Any] = {}

        self.acceptance_rate = 0.0
        self.move_history: list[tuple[str, bool | None]] = []

        self.context = self.default_context(atoms, self._rng)

        super().__init__(
            atoms,
            logfile=logfile,
            trajectory=trajectory,
            restart_file=restart_file,
            logging_interval=logging_interval,
            logging_mode=logging_mode,
        )

        if self.default_logger:
            self.default_logger.add_mc_fields(self)

    def add_move(
        self,
        move: MoveType,
        criteria: CriteriaType | None = None,
        name: str = "default",
        interval: int = 1,
        probability: float = 1.0,
        minimum_count: int = 0,
    ) -> None:
        """
        Add a move to the `MonteCarlo` object.

        Parameters
        ----------
        move : MoveType
            The move to add to the `MonteCarlo` object.
        criteria : CriteriaType | None, optional
            The acceptance criteria for the move, by default None. If None, the move must be an instance of a known move type.
        name : str, optional
            Name of the move, used to identify the move in the `moves` dictionary, by default "default".
        interval : int, optional
            The interval at which the move is attempted, by default 1.
        probability : float, optional
            The probability of the move being attempted, by default 1.0.
        minimum_count : int, optional
            The minimum number of times the move must be performed, by default 0.
        """
        forced_moves_count = sum(
            [self.moves[name].minimum_count for name in self.moves]
        )

        if forced_moves_count + minimum_count > self.max_cycles:
            raise ValueError("The number of forced moves exceeds the number of cycles.")

        if criteria is None:
            for move_type in self.default_criteria:
                if isinstance(move, move_type):
                    criteria = cast("CriteriaType", self.default_criteria[move_type]())
                    break

        if criteria is None:
            raise ValueError(
                f"No criteria provided, and no default criteria found for move type {type(move)}."
            )

        self.moves[name] = MoveStorage[MoveType, CriteriaType](
            move=move,
            criteria=criteria,
            interval=interval,
            probability=probability,
            minimum_count=minimum_count,
        )

    def srun(self, steps=100_000_000) -> Generator[Any, None, None]:
        """
        Run the simulation for a given number of steps, yielding the steps.

        Parameters
        ----------
        steps : int, optional
            The number of steps to run the simulation for, by default 100,000,000.

        Yields
        ------
        Generator[Any, None, None]
            A `Generator` yielding the steps of the simulation.
        """
        for step in self.irun(steps):
            for _ in step:
                pass

            yield step

    def validate_simulation(self) -> None:
        """
        Validate the simulation setup. Checks that moves have been added to the Monte Carlo simulation and that the Atoms object has a calculator with results.
        """
        if len(self.moves) == 0:
            warn(
                "No moves have been added to the Monte Carlo simulation. "
                "Please add moves using the `add_move` method.",
                UserWarning,
                2,
            )

        try:
            self.last_results = self.atoms.calc.results  # type: ignore[try-attr]
        except AttributeError:
            warn(
                "Atoms object does not have calculator attached, or does not support the `results` attribute.",
                UserWarning,
                2,
            )
            self.last_results = {}

        super().validate_simulation()

    def run(self, steps=100_000_000) -> None:
        """
        Run the simulation for a given number of steps.

        Parameters
        ----------
        steps : int, optional
            The number of steps to run the simulation for, by default 100,000,000.
        """
        for step in self.irun(steps):
            for _ in step:
                pass

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `MonteCarlo` object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `MonteCarlo` object.
        """
        dictionary = {
            **super().to_dict(),
            "name": self.__class__.__name__,
            "context": self.context.to_dict(),
            "rng_state": self._rng.bit_generator.state,
            "moves": {
                name: move_storage.to_dict()
                for name, move_storage in self.moves.items()
            },
        }

        dictionary.setdefault("kwargs", {}).update(
            {"max_cycles": self.max_cycles, "seed": self.__seed}
        )
        dictionary.setdefault("attributes", {}).update(
            {"last_results": self.last_results}
        )

        return dictionary

    todict = to_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs_override: Any) -> Self:
        """
        Create a `MonteCarlo` object from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the object.
        **kwargs_override : Any
            Additional keyword arguments to override the ones in the dictionary.

        Returns
        -------
        Self
            The `MonteCarlo` object created from the dictionary.
        """
        data = deepcopy(data)

        kwargs = data.get("kwargs", {})
        kwargs = kwargs | kwargs_override

        mc = cls(data["atoms"], **kwargs)
        mc._rng.bit_generator.state = data["rng_state"]

        for key, value in data.get("attributes", {}).items():
            setattr(mc, key, value)

        for key, value in data.get("context", {}).items():
            setattr(mc.context, key, value)

        for name, move_storage_data in data.get("moves", {}).items():
            move_storage_class: type[MoveStorage[MoveType, CriteriaType]] = (
                get_typed_class(move_storage_data["name"], MoveStorage)
            )
            move_storage = move_storage_class.from_dict(move_storage_data)

            mc.moves[name] = move_storage

        return mc

    def yield_moves(self) -> Generator[str, None, None]:
        """
        Yield moves to be performed given the move configured. The moves are selected based on their probability, and their interval. Forced moves are introduced based on their minimum count. Moves are yielded separately, re-constructing the move_probabilities array each time, allowing for a dynamic change in the probability of moves between moves.

        Yields
        ------
        Generator[str, None, None]
            The name of the move to be performed.
        """
        available_moves: list[str] = [
            name
            for name in self.moves
            if self.step_count % self.moves[name].interval == 0
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

    def step(self) -> Generator[str, None, None]:
        """
        Perform a single step of the simulation.

        Yields
        ------
        Generator[str, None, None]
            A generator yielding the names of the moves that will be performed in this step.
        """
        self.move_history = []

        for move_name in self.yield_moves():
            yield move_name

            move_storage = self.moves[move_name]
            move = move_storage.move

            if move(self.context):
                is_accepted = move_storage.criteria.evaluate(self.context)

                if is_accepted:
                    self.save_state()
                else:
                    self.revert_state()
            else:
                is_accepted = None

            self.move_history.append((move_name, is_accepted))

        self.acceptance_rate = np.mean(
            [1 if is_accepted else 0 for _, is_accepted in self.move_history]
        )

    def save_state(self) -> None:
        """
        Save the current state of the simulation. This method is called when a move is accepted.
        """
        self.context.save_state()

        try:
            self.last_results = self.atoms.calc.results  # type: ignore[try-attr]
        except AttributeError:
            warn(
                "Atoms object does not have calculator results attached.",
                UserWarning,
                2,
            )
            self.last_results = {}

    def revert_state(self) -> None:
        """
        Revert the last move made by the simulation. This method is called when a move is rejected.
        """
        self.context.revert_state()

        try:
            self.atoms.calc.results = self.last_results  # type: ignore[try-attr]
        except AttributeError:
            warn(
                "Atoms object does not have calculator attached, or does not support the `results` attribute",
                UserWarning,
                2,
            )

    def __repr__(self) -> str:
        """
        Return a string representation of the Monte Carlo object.

        Returns
        -------
        str
            A string representation of the Monte Carlo object.
        """
        return f"{self.__class__.__name__}(atoms={self.atoms}, max_cycles={self.max_cycles}, seed={self.__seed}, moves={self.moves}, step_count={self.step_count}, default_logger={self.default_logger}, default_trajectory={self.default_trajectory}, default_restart={self.default_restart})"
