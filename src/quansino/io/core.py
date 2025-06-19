from __future__ import annotations

import atexit
import sys
from abc import abstractmethod
from contextlib import suppress
from io import IOBase
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quansino.io.file import FileManager


class Observer:
    """
    Base class for observers which are used to monitor and record the state of the simulation at specified intervals.

    Parameters
    ----------
    interval : int, optional
        The interval at which the observer will be called, by default 1.

    Attributes
    ----------
    interval : int
        The interval at which the observer will be called.
    """

    __slots__ = ("interval",)

    def __init__(self, interval: int = 1) -> None:
        """Initialize the `Observer` object."""
        self.interval = interval

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any):
        """
        Call the observer with the given arguments. This method should be overridden by subclasses to implement specific behavior.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to the observer.
        **kwargs : Any
            Keyword arguments passed to the observer.
        """
        ...

    @abstractmethod
    def attach_simulation(self, *args: Any, **kwargs: Any) -> None:
        """
        Attach a simulation to the observer. This method should be overridden by subclasses to implement specific attachment behavior.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to the observer.
        **kwargs : Any
            Keyword arguments passed to the observer.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `Observer` object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `Observer` object.
        """
        return {"name": self.__class__.__name__, "kwargs": {"interval": self.interval}}


class TextObserver(Observer):
    """
    Base class for text-based observers in a simulation. `TextObservers` are used to write output to a file or stream at specified intervals.

    Parameters
    ----------
    file : IO | Path | str
        The file or stream to write output to. This can be a file object, a string representing a file path, or a [`Path`][pathlib.Path] object.
    interval : int, optional
        The interval at which the observer will be called, by default 1.
    mode : str, optional
        The mode in which to open the file, by default "a".
    encoding : str | None, optional
        The encoding to use when opening the file, by default None. If None, default to 'utf-8' for text files stays None for binary files.

    Attributes
    ----------
    accept_stream : bool
        Whether the observer accepts a stream of data.
    file : IO
        The file or stream to write output to.
    mode : str
        The mode in which to open the file.
    encoding : str | None
        The encoding to use when opening the file.
    """

    __slots__ = ("_file", "encoding", "mode")

    accept_stream: bool = True

    def __init__(
        self,
        file: IO | Path | str,
        interval: int = 1,
        mode: str = "a",
        encoding: str | None = None,
    ) -> None:
        """Initialize the `TextObserver` object."""
        super().__init__(interval)

        self.mode: str = mode
        self.encoding: str | None = encoding or ("utf-8" if "b" not in mode else None)

        self.file = file

    def __repr__(self) -> str:
        """
        Return a representation of the `TextObserver`.

        Returns
        -------
        str
            The representation of the `TextObserver`.
        """
        return f"{self.__class__.__name__}({self.__str__()}, mode={self.mode}, encoding={self.encoding}, interval={self.interval})"

    def __str__(self) -> str:
        """
        Return a string representation of the `TextObserver`.

        Returns
        -------
        str
            The string representation of the `TextObserver`, including the file type and name.
        """
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
        """
        Get the file object associated with the `TextObserver`.

        Returns
        -------
        IO
            The file object associated with the `TextObserver`.
        """
        return self._file

    @file.setter
    def file(self, value: IO | str | Path) -> None:
        """
        Set the file object for the `TextObserver`.

        Parameters
        ----------
        value : IO | str | Path
            The file or stream to write output to. This can be a file object, a string representing a file path, or a `Path` object.
        """
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
                        f"{self.__class__.__name__} does not accept non-file streams (non-seekable). Please use a different file type."
                    )

            self._file = value
        else:
            raise TypeError(
                f"Invalid file type: {type(value)}. Expected str, Path, or file-like object."
            )

        atexit.register(self.close)

    def attach_simulation(self, file_manager: FileManager) -> None:
        """
        Attach the simulation to the `Observer` via a `FileManager`.

        Parameters
        ----------
        file_manager : FileManager
            The `FileManager` instance to attach to the observer.
        """
        file_manager.register(self.close)

    def close(self) -> None:
        """Close the file if it is not a standard stream."""
        if not hasattr(self, "_file"):
            return

        if self._file not in (sys.stdout, sys.stderr, sys.stdin):
            with suppress(OSError, AttributeError, ValueError):
                self._file.close()

                atexit.unregister(self.close)

    __del__ = close

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `TextObserver` object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `TextObserver`.
        """
        dictionary = super().to_dict()
        dictionary.setdefault("kwargs", {})

        dictionary["kwargs"]["mode"] = self.mode
        dictionary["kwargs"]["encoding"] = self.encoding

        return dictionary
