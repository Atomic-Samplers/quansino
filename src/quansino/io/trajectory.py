from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ase.io.extxyz import write_xyz

if TYPE_CHECKING:
    from pathlib import Path

    from ase.atoms import Atoms

from quansino.io.core import TextObserver


class TrajectoryObserver(TextObserver):
    def __init__(
        self,
        atoms: Atoms,
        trajectory_file: str | Path,
        interval: int = 1,
        mode: str = "a",
        write_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the Trajectory observer with a trajectory file, mode, and other parameters.

        Parameters
        ----------
        trajectory_file : IO | str | Path
            The trajectory file to write to.
        mode : str
            The mode in which to open the file (e.g., 'a' for append).
        communicator : Any
            The communicator to use for parallel processing, if applicable.
        function : Callable
            The function to call when writing to the file.
        function_kwargs : dict[str, Any] | None
            Additional keyword arguments to pass to the function.
        """

        super().__init__(filename=trajectory_file, interval=interval, mode=mode)

        self.atoms = atoms
        self.write_kwargs = write_kwargs or {}

    def __call__(self) -> None:
        """
        Call the function to write the trajectory to the file.

        Parameters
        ----------
        *args : Any
            Positional arguments to pass to the function.
        **kwargs : Any
            Keyword arguments to pass to the function.
        """
        write_xyz(self.file, images=self.atoms, **self.write_kwargs)
