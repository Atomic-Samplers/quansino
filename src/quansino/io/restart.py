from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any

from ase.io.jsonio import write_json

from quansino.io.core import TextObserver

if TYPE_CHECKING:
    from pathlib import Path

    from quansino.mc.core import Driver


class RestartObserver(TextObserver):

    accept_stream: bool = False

    def __init__(
        self,
        simulation: Driver,
        file: IO | Path | str,
        interval: int = 1,
        mode: str = "a",
        unique: bool = True,
        write_kwargs: dict[str, Any] | None = None,
        **observer_kwargs: Any,
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
        super().__init__(file=file, interval=interval, mode=mode, **observer_kwargs)

        self.simulation = simulation
        self.write_kwargs = write_kwargs or {}

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
            self._file.seek(0)
            self._file.truncate()
        else:
            self.file = (
                self.file_path.parent
                / f"{self.file_path.stem}_{self.simulation.step_count}{self.file_path.suffix}"
            )
        write_json(self._file, obj=self.simulation, **self.write_kwargs)
        self._file.flush()

    @property
    def unique(self) -> bool:
        """Check if the file is unique."""
        return self._unique

    @unique.setter
    def unique(self, value: bool) -> None:
        """Set the file to be unique."""
        self._unique = value

        if not value:
            self.file_path = self.get_file_path()
