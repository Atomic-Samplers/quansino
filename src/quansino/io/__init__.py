"""IO and logging utilities for quansino."""

from __future__ import annotations

from quansino.io.core import Observer, TextObserver
from quansino.io.file import FileManager
from quansino.io.logger import Logger
from quansino.io.restart import RestartObserver
from quansino.io.trajectory import TrajectoryObserver
from quansino.registry import register_class

__all__ = [
    "Logger",
    "Observer",
    "RestartObserver",
    "TextObserver",
    "TrajectoryObserver",
]

io_registry: dict[str, type[Observer]] = {
    "Logger": Logger,
    "RestartObserver": RestartObserver,
    "TextObserver": TextObserver,
    "TrajectoryObserver": TrajectoryObserver,
}

for name, cls in io_registry.items():
    register_class(cls, name)
