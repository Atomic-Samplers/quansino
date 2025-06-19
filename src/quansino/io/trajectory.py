from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any
from weakref import proxy

from ase.io.extxyz import write_xyz

if TYPE_CHECKING:
    from pathlib import Path

    from ase.atoms import Atoms

from quansino.io.core import TextObserver


class TrajectoryObserver(TextObserver):

    __slots__ = ("atoms", "write_kwargs")

    def __init__(
        self,
        atoms: Atoms,
        file: IO | Path | str,
        interval: int = 1,
        mode: str = "a",
        write_kwargs: dict[str, Any] | None = None,
        **observer_kwargs: Any,
    ) -> None:
        """
        Initialize the `TrajectoryObserver` with mode, and other parameters. In `quansino`, trajectory files are written in the XYZ format using ASE's `write_xyz` function.

        Parameters
        ----------
        atoms : Atoms
            The ASE Atoms object to write to the trajectory file.
        file : IO | str | Path
            The file or path to the trajectory file.
        interval : int, optional
            The interval at which to write the trajectory, by default 1.
        mode : str, optional
            The mode in which to open the file (e.g., 'a' for append), by default "a".
        write_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments to pass to the writing function, by default None.
        **observer_kwargs : Any
            Additional keyword arguments for the observer class.
        """
        super().__init__(file=file, interval=interval, mode=mode, **observer_kwargs)

        self.atoms = proxy(atoms)
        self.write_kwargs = write_kwargs or {}

    def __call__(self) -> None:
        """Call the function to write the trajectory to the file."""
        write_xyz(self._file, images=self.atoms, **self.write_kwargs)
        self._file.flush()
