from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from quansino.io.file import FileManager

if TYPE_CHECKING:
    from pathlib import Path


class Observer(FileManager):

    def __init__(self, interval: int = 1) -> None:
        """
        Initialize the observer with an interval and communicator.

        Parameters
        ----------
        interval : int
            The interval at which to call the observer.
        communicator : Any
            The communicator to use for parallel processing, if applicable.
        """
        super().__init__()
        self.interval = interval

    def __call__(self, *args, **kwargs): ...


class TextObserver(Observer):

    def __init__(
        self, filename: str | Path, interval: int = 1, mode: str = "a"
    ) -> None:
        """
        Initialize the TextObserver with a filename, function, and other parameters.

        Parameters
        ----------
        filename : IO | str | Path
            The filename or file object to write to.
        function : Callable
            The function to call when writing to the file.
        interval : int
            The interval at which to call the function.
        mode : str
            The mode in which to open the file (e.g., 'a' for append).
        communicator : Any
            The communicator to use for parallel processing, if applicable.
        function_kwargs : dict[str, Any] | None
            Additional keyword arguments to pass to the function.
        """
        super().__init__(interval)

        self.mode = mode
        self.filename = filename

    @property
    def filename(self) -> str:
        """Return the filename."""
        return self._filename

    @filename.setter
    def filename(self, filename: str | Path) -> None:
        """Set the filename."""
        if hasattr(self, "file"):
            with suppress(Exception):
                self.file.close()

        self._filename = str(filename)
        self.file = self.open_file(filename, self.mode)

    def __str__(self):
        return self._filename

    def __repr__(self):
        return f"{self.__class__.__name__}({self._filename})"
