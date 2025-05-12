from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ase.io.jsonio import write_json

from quansino.io.core import TextObserver

if TYPE_CHECKING:

    from quansino.mc.core import Driver


class RestartObserver(TextObserver):
    def __init__(
        self,
        simulation: Driver,
        restart_file: str | Path,
        interval: int = 1,
        unique: bool = False,
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
        super().__init__(filename=restart_file, interval=interval, mode=mode)

        self.simulation = simulation
        self.write_kwargs = write_kwargs or {}

        self.original_path = Path(restart_file)
        self.unique = unique

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
        if self.unique:
            self.file.seek(0)
            self.file.truncate()
        else:
            stem, ext = self.original_path.stem, self.original_path.suffix
            self.filename = f"{stem}_{self.simulation.nsteps}{ext}"

        write_json(self.file, obj=self.simulation, **self.write_kwargs)

    @staticmethod
    def separate_filename(string: str) -> tuple[str, str]:
        """
        Return the stem filename of the trajectory file.

        Returns
        -------
        str
            The string without the file extension.
        """
        i = string.rfind(".")
        if 0 < i < len(string) - 1:
            return (string[:i], string[i + 1 :])
        else:
            return (string, "")
