from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any
from weakref import proxy

from ase.io.jsonio import write_json

from quansino.io.core import TextObserver

if TYPE_CHECKING:
    from pathlib import Path

    from quansino.mc.core import Driver


class RestartObserver(TextObserver):
    """
    Observer to write restart data for atomistic simulations. This observer writes the state of the simulation to a file in JSON format at specified intervals, allowing for the simulation to be restarted later.

    Parameters
    ----------
    simulation : Driver
        The simulation driver to observe and write restart data for.
    file : IO | Path | str
        The file to write restart data to.
    interval : int, optional
        The interval at which to write restart data, by default 1.
    mode : str, optional
        The mode in which to open the file, by default "a".
    write_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the JSON writer function, by default None.
    **observer_kwargs : Any
        Additional keyword arguments to pass to the parent TextObserver.

    Attributes
    ----------
    accept_stream : bool
        Whether to accept a stream of data. This is set to False by default, as this observer does not handle streaming data.
    write_kwargs : dict[str, Any]
        Additional keyword arguments to pass to the JSON writer function.
    """

    accept_stream: bool = False

    __slots__ = ("__weakref__", "simulation", "write_kwargs")

    def __init__(
        self,
        simulation: Driver,
        file: IO | Path | str,
        interval: int = 1,
        mode: str = "a",
        write_kwargs: dict[str, Any] | None = None,
        **observer_kwargs: Any,
    ) -> None:
        """
        Initialize the `RestartObserver` with a file, interval, and other parameters.
        """
        super().__init__(file=file, interval=interval, mode=mode, **observer_kwargs)

        self.simulation: Driver = proxy(simulation)
        self.write_kwargs = write_kwargs or {}

    def __call__(self) -> None:
        """Call the function to write the restart data to the file."""
        self._file.seek(0)
        self._file.truncate()

        write_json(self._file, obj=self.simulation, **self.write_kwargs)
        self._file.flush()
