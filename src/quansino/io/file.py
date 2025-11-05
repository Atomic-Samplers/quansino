from __future__ import annotations

import atexit
from contextlib import ExitStack, suppress
from typing import TYPE_CHECKING, Any, Final, Self
from warnings import warn

if TYPE_CHECKING:
    from collections.abc import Callable

    from quansino.io.core import Observer


class ObserverManager:
    """
    Class to automatically manage file resources. This class uses an [`ExitStack`][contextlib.ExitStack] object to register file resources that need to be closed when the program exits or when the context manager is exited. It can be used to ensure that files are properly closed, preventing resource leaks.

    Attributes
    ----------
    exitstack : ExitStack
        The exit stack used for managing file resources.

    Example
    -------
    ``` python
    from quansino.io.file import ObserverManager

    with ObserverManager() as fm:
        file = open("example.txt", "w")
        fm.register(file.close)  # Register the file close method
        file.write("Hello, World!")
    # The file will be automatically closed when exiting the context manager
    ```

    Notes
    -----
    This class is mainly used in the [`Driver`][quansino.mc.driver.Driver] class to manage file resources automatically. Users are free to use it directly for managing their own file resources as well.
    """

    def __init__(self) -> None:
        """Initialize the ObserverManager with an ExitStack for resource management."""
        self.exitstack: Final[ExitStack] = ExitStack()

        self.observers: dict[str, Observer] = {}

        atexit.register(self.close)

    def __enter__(self) -> Self:
        """
        Enter the context manager and return self.

        Returns
        -------
        Self
            The instance of the `ObserverManager`.
        """
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """
        Exit the context manager and close all files.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to the exit method.
        **kwargs : Any
            Keyword arguments passed to the exit method.
        """
        self.close()

    __del__ = __exit__

    def close(self) -> None:
        """Close all registered resources"""
        with suppress(OSError, AttributeError, ValueError):
            self.exitstack.close()
            atexit.unregister(self.close)

    def register(self, resource: Callable) -> Any:
        """
        Register a resource for automatic cleanup. This method registers a callable resource (like a file close method) to be called when the context manager exits or when the program terminates. It returns the result of the callback, which is typically `None` for close methods.

        Parameters
        ----------
        resource : Callable
            The resource to register for cleanup.

        Returns
        -------
        Any
            The result of the callback, typically `None` for close methods.
        """
        return self.exitstack.callback(resource)

    def attach_observer(self, name: str, observer: Observer) -> None:
        """
        Attach an observer to the simulation.

        Parameters
        ----------
        name : str
            The name of the observer.
        observer : Observer
            The observer object to attach. Must be a subclass of `Observer`.
        """
        self.register(observer.close)
        self.observers[name] = observer

    def detach_observer(self, name: str, close: bool = False) -> None:
        """
        Detach an observer by name.

        Parameters
        ----------
        name : str
            The name of the observer to detach.
        """
        if observer := self.observers.pop(name, None):
            if close:
                with suppress(Exception):
                    observer.close()
        else:
            warn(f"`Observer` '{name}' not found when deleting.", UserWarning, 2)
