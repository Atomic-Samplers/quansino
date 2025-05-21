from __future__ import annotations

import atexit
from contextlib import ExitStack, suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


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

    __del__ = __exit__

    def register(self, resource: Callable[[], None]) -> Any:
        """Register a resource for automatic cleanup"""
        return self.exitstack.callback(resource)

    def close(self) -> None:
        """Close all registered resources"""
        with suppress(OSError, AttributeError, ValueError):
            self.exitstack.close()
            atexit.unregister(self.close)
