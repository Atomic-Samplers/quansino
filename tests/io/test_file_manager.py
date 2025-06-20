from __future__ import annotations

from io import StringIO

from quansino.io.core import TextObserver
from quansino.io.file import FileManager


def test_file_manager():
    """Test the `FileManager` context manager with multiple `TextObserver` instances."""
    observers = []

    with FileManager() as fm:
        assert fm.exitstack is not None

        for i in range(10):
            string_io = StringIO()
            text_observer = TextObserver(string_io, interval=i)
            text_observer.attach_simulation(fm)

            observers.append(text_observer)

    for i in range(10):
        assert observers[i].file.closed
