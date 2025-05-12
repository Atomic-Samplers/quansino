from __future__ import annotations

import atexit
import sys
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import IO, Any


class FileManager:
    """
    Context manager for file operations with automatic cleanup.
    This class manages file opening and closing, ensuring that files are
    properly closed when no longer needed. It also handles the case where
    a file is not specified, in which case it uses `/dev/null` for writing
    or reading from standard input. It is designed to be used as a context manager, allowing for easy
    management of file resources.

    Attributes
    ----------
    exitstack : ExitStack
        The exit stack used for managing file resources.

    Example
    -------
    ```python
    with FileManager() as fm:
        with fm.open_file("output.txt", "w") as f:
            f.write("Hello, World!")
    ```

    In this example, the file `output.txt` will be automatically closed.
    """

    def __init__(self) -> None:
        self.exitstack = ExitStack()

        atexit.register(self.close)

    def __enter__(self):
        """Enter the context manager and return self"""
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Exit the context manager and close all files"""
        self.close()

    def register(self, resource: Any) -> Any:
        """Register a resource for automatic cleanup"""
        return self.exitstack.enter_context(resource)

    def open_file(
        self, file: str | Path | None, mode: str = "w", encoding: str | None = None
    ) -> IO[Any]:
        """Open a file and register it for automatic cleanup

        Parameters
        ----------
        file : str | Path | None
            The file to open. If None, `/dev/null` is used.
        mode : str
            The mode in which to open the file. Defaults to 'w'.
        encoding : str | None
            The encoding to use for the file. Defaults to 'utf-8' if not specified.

        Returns
        -------
        IO[Any]
            The opened file object.
        """
        if file == "-":
            return sys.stdout if "w" in mode else sys.stdin

        if file is None:
            return self.register(Path("/dev/null").open(mode=mode, encoding=encoding))

        path = Path(file)
        encoding = encoding or ("utf-8" if "b" not in mode else None)
        return self.register(path.open(mode=mode, encoding=encoding))

    def close(self) -> None:
        """Close all registered resources"""
        with suppress(Exception):
            self.exitstack.close()
