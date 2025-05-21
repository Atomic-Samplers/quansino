from __future__ import annotations

import atexit
import sys
from contextlib import suppress
from io import IOBase
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quansino.io.file import FileManager


class Observer:

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

        atexit.register(self.close)

    def __call__(self, *args, **kwargs): ...

    def close(self) -> None: ...

    __del__ = close

    def attach_simulation(self, file_manager: FileManager) -> None: ...

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the observer to a dictionary.

        Returns
        -------
        dict[str, str | int]
            A dictionary representation of the observer.
        """
        return {"name": self.__class__.__name__, "kwargs": {"interval": self.interval}}


class TextObserver(Observer):

    accept_stream: bool = True

    def __init__(
        self,
        file: IO | Path | str,
        interval: int = 1,
        mode: str = "a",
        encoding: str | None = None,
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

        self.mode: str = mode
        self.encoding: str | None = encoding or ("utf-8" if "b" not in mode else None)

        self.file = file

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__str__()}, {self.mode})"

    def __str__(self):
        if self._file in (sys.stdout, sys.stderr, sys.stdin):
            return f"Stream:{self._file.name}"

        if hasattr(self._file, "name"):
            name = self._file.name

            if (
                not isinstance(name, int)
                and isinstance(name, str)
                and name not in ("", ".")
            ):
                return f"Path:{name}"

        if hasattr(self._file, "__class__"):
            return f"Class:{self._file.__class__.__name__}"

        return "Class:<Unknown>"

    @property
    def file(self) -> IO:
        """Return the filename as a Path object."""
        return self._file

    @file.setter
    def file(self, value: IO | str | Path) -> None:
        """Set the filename and open the file."""
        if hasattr(self, "_file"):
            self.close()

        if isinstance(value, str):
            value = Path(value)

        if isinstance(value, Path):
            self._file = value.open(mode=self.mode, encoding=self.encoding)
        elif (
            hasattr(value, "read")
            or hasattr(value, "write")
            or isinstance(value, IOBase)
        ):
            if getattr(value, "closed", False):
                raise ValueError(
                    f"Impossible to link a closed file for '{self.__class__.__name__}'."
                )
            if not self.accept_stream:
                is_seekable = False

                if hasattr(value, "seekable"):
                    is_seekable = value.seekable()
                elif hasattr(value, "seek"):
                    is_seekable = True

                if not is_seekable:
                    raise ValueError(
                        f"{self.__class__.__name__} does not accept non-file streams (non-seekable). Please use a different file type for this `Observer`."
                    )

            self._file: IO = value
        else:
            raise TypeError(
                f"Invalid file type: {type(value)}. Expected str, Path, or file-like object."
            )

    def get_file_path(self) -> Path:
        """
        Find the path of this file, if it exists.

        Parameters
        ----------
        string : str
            The string to search for.

        Returns
        -------
        Path
            The path of the file.
        """
        string = str(self)

        if "Path:" in string:
            path = Path(string.split("Path:")[-1].strip())

            if path.exists() and path.is_file():
                return path.expanduser().resolve()

        raise ValueError(
            f"Invalid file path: {self.file}. Must be a regular file when using the `{self.__class__.__name__}` observer."
        )

    def attach_simulation(self, file_manager: FileManager) -> None:
        """
        Attach the simulation to the observer.

        Parameters
        ----------
        driver : Driver
            The simulation driver to attach to the observer.
        """
        file_manager.register(self.close)

    def close(self) -> None:
        """
        Close the file if it is not a stream.
        """
        if not hasattr(self, "_file"):
            return

        if self._file not in (sys.stdout, sys.stderr, sys.stdin):
            with suppress(OSError, AttributeError, ValueError):
                self._file.close()

                atexit.unregister(self.close)
