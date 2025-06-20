from __future__ import annotations

from collections.abc import Generator
from typing import IO, TYPE_CHECKING, Final
from warnings import warn

from quansino.io.file import FileManager
from quansino.io.logger import Logger
from quansino.io.restart import RestartObserver
from quansino.io.trajectory import TrajectoryObserver

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from typing import Any

    from ase.atoms import Atoms

    from quansino.io.core import Observer


class Driver:
    """
    Base class for managing atomistic simulations. This class provides the basic functionality for running simulations, including observer management, file I/O handling, and simulation state tracking. It is not intended to be used directly, but rather as a base class for specific simulation implementations.

    Parameters
    ----------
    atoms : Atoms
        The ASE Atoms object to operate on.
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
    atoms : Atoms
        The ASE Atoms object being simulated.
    observers : dict[str, Observer]
        Dictionary of attached observers, keyed by their names.
    logging_interval : int
        The interval at which observers are called.
    logging_mode : str
        The file mode for opening log files.
    step_count : int
        The current step count of the simulation.
    max_steps : int
        The maximum number of steps to run in the simulation.
    file_manager : FileManager
        File manager for handling file I/O operations.
    default_logger : Logger | None
        The default logger for the simulation, if set.
    default_trajectory : TrajectoryObserver | None
        The default trajectory observer for the simulation, if set.
    default_restart : RestartObserver | None
        The default restart observer for the simulation, if set.
    """

    def __init__(
        self,
        atoms: Atoms,
        logfile: Logger | IO | Path | str | None = None,
        trajectory: TrajectoryObserver | IO | Path | str | None = None,
        restart_file: RestartObserver | IO | Path | str | None = None,
        logging_interval: int = 1,
        logging_mode: str = "a",
    ) -> None:
        """Initialize the `Driver` object."""
        self.atoms = atoms

        self.observers: dict[str, Observer] = {}

        self.logging_interval = logging_interval
        self.logging_mode = logging_mode

        self.step_count: int = 0
        self.max_steps: int = 0

        self.file_manager: Final = FileManager()

        self.default_logger = logfile
        self.default_trajectory = trajectory
        self.default_restart = restart_file

    def attach_observer(self, name: str, observer: Observer) -> None:
        """
        Attach an observer to the simulation.

        Parameters
        ----------
        name : str
            The name of the observer.
        observer : Observer
            The observer object to attach. Must be a subclass of `Observer`.
        """
        observer.attach_simulation(self.file_manager)
        self.observers[name] = observer

    def detach_observer(self, name: str) -> None:
        """
        Detach an observer by name.

        Parameters
        ----------
        name : str
            The name of the observer to detach.
        """
        if self.observers.pop(name, None) is None:
            warn(f"`Observer` '{name}' not found when deleting.", UserWarning, 2)

    def call_observers(self) -> None:
        """
        Call all attached observers based on their configured intervals. The observers will be called if their interval matches the current step count.
        """
        for observer in self.observers.values():
            interval = observer.interval
            if (interval > 0 and self.step_count % interval == 0) or (
                interval < 0 and self.step_count == abs(interval)
            ):
                observer()

    def validate_simulation(self) -> None:
        """
        Validate the simulation setup. Checks that the atoms object has a properly configured calculator and computes the initial potential energy.

        Raises
        ------
        AttributeError
            If atoms object has no calculator or calculator has no results attribute.
        """
        if self.atoms.calc is None:
            raise AttributeError("Atoms object must have a calculator attached to it.")
        if not hasattr(self.atoms.calc, "results"):
            raise AttributeError("Calculator object must have a results attribute.")

    def irun(self, steps=100_000_000) -> Generator[Any, None, None]:
        """
        Run the simulation as a `Generator`.

        Parameters
        ----------
        steps : int
            Maximum number of simulation steps to perform, by default 100,000,000.

        Yields
        ------
        Generator[Any, None, None]
            The result of each simulation step.
        """
        self.validate_simulation()
        self.max_steps = self.step_count + steps

        if self.step_count == 0:
            if self.default_logger:
                self.default_logger.write_header()

            self.call_observers()

        while not self.converged():
            yield self.step()
            self.step_count += 1

            self.atoms.get_potential_energy()

            self.call_observers()

    def run(self, steps=100_000_000) -> None:
        """
        Run the simulation for a given number of steps.

        Parameters
        ----------
        steps : int, optional
            Maximum number of simulation steps to perform, by default 100,000,000.
        """
        for _ in self.irun(steps):
            pass

    def step(self) -> Any:
        """
        Perform a single step.

        Returns
        -------
        Any
            Result of the Monte Carlo step. Implementation-dependent.
        """
        raise NotImplementedError(
            f"The `step` method must be implemented in subclasses of {self.__class__.__name__}."
        )

    def converged(self) -> bool:
        """
        Check if the simulation has converged.

        Returns
        -------
        bool
            True if the maximum number of steps has been reached.
        """
        return self.step_count >= self.max_steps

    def to_dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the `Driver` object.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `Driver` object.
        """
        return {
            "name": self.__class__.__name__,
            "atoms": self.atoms.copy(),
            "kwargs": {
                "logging_interval": self.logging_interval,
                "logging_mode": self.logging_mode,
            },
            "attributes": {"step_count": self.step_count},
        }

    @property
    def default_trajectory(self) -> TrajectoryObserver | None:
        """
        Get the default trajectory observer, if set.

        Returns
        -------
        TrajectoryObserver | None
            The default trajectory observer, or None if not set.
        """
        return self._default_trajectory

    @default_trajectory.setter
    def default_trajectory(
        self, default_trajectory: TrajectoryObserver | IO | Path | str | None
    ) -> None:
        """
        (Un)set the default trajectory observer.

        Parameters
        ----------
        default_trajectory : TrajectoryObserver | IO | Path | str | None
            Trajectory observer or file specification, or None to unset the default trajectory observer.
        """
        if default_trajectory is None:
            self._default_trajectory = None
            return

        if not isinstance(default_trajectory, TrajectoryObserver):
            self._default_trajectory = TrajectoryObserver(
                atoms=self.atoms,
                file=default_trajectory,
                interval=self.logging_interval,
                mode=self.logging_mode,
            )
        else:
            self._default_trajectory = default_trajectory

        self.attach_observer("default_trajectory", self._default_trajectory)

    @property
    def default_logger(self) -> Logger | None:
        """
        Get the default logger, if set.

        Returns
        -------
        Logger | None
            The default logger, or None if not set.
        """
        return self._default_logger

    @default_logger.setter
    def default_logger(self, default_logger: Logger | IO | Path | str | None) -> None:
        """
        (Un)set the default logger.

        Parameters
        ----------
        default_logger : Logger | IO | Path | str | None
            Logger object or file specification, or None to unset the default logger.
        """
        if default_logger is None:
            self._default_logger = None
            return

        if not isinstance(default_logger, Logger):
            self._default_logger = Logger(
                logfile=default_logger,
                interval=self.logging_interval,
                mode=self.logging_mode,
            )
        else:
            self._default_logger = default_logger

        self.attach_observer("default_logger", self._default_logger)

    @property
    def default_restart(self) -> RestartObserver | None:
        """
        Get the default restart observer, if set.

        Returns
        -------
        RestartObserver | None
            The default restart observer, or None if not set.
        """
        return self._default_restart

    @default_restart.setter
    def default_restart(
        self, restart_observer: RestartObserver | IO | Path | str | None
    ) -> None:
        """
        (Un)set the default restart observer.

        Parameters
        ----------
        restart_observer : RestartObserver | IO | Path | str | None
            Restart observer or file specification, or None to unset the default restart observer.
        """
        if restart_observer is None:
            self._default_restart = None
            return

        if not isinstance(restart_observer, RestartObserver):
            self._default_restart = RestartObserver(
                simulation=self,
                file=restart_observer,
                interval=self.logging_interval,
                mode=self.logging_mode,
            )
        else:
            self._default_restart = restart_observer

        self.attach_observer("default_restart", self._default_restart)

    def close(self) -> None:
        """Close the file manager and clean up resources."""
        self.file_manager.close()
