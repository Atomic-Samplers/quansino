"""Monte Carlo."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ase.optimize.optimize import Dynamics
from numpy.random import PCG64, Generator

from quansino.io import Logger

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms


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

    def __init__(
        self,
        atoms: Atoms,
        seed: int,
        trajectory: str | Path | None = None,
        logfile: str | Path | None = None,
        append_trajectory: bool = False,
        loginterval: int = 1,
    ) -> None:
        """Initialize the Monte Carlo object."""
        self.__seed = seed
        self.__rng = Generator(PCG64(seed))

        if isinstance(trajectory, Path):
            trajectory = str(trajectory)

        Dynamics.__init__(
            self, atoms, trajectory=trajectory, append_trajectory=append_trajectory
        )

        if logfile:
            self.default_logger = Logger(logfile)
            self.default_logger.add_mc_fields(self)
            self.attach(self.closelater(self.default_logger), loginterval)

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
            "seed": self.__seed,
            "rng_state": self.__rng.bit_generator.state,
            "nsteps": self.nsteps,
        }

    def converged(self) -> bool:
        """MC is 'converged' when number of maximum steps is reached.

        Returns
        -------
        bool
            True if the maximum number of steps is reached.
        """
        return self.nsteps >= self.max_steps
