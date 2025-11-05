from __future__ import annotations

from io import StringIO

from quansino.io.core import TextObserver
from quansino.io.file import ObserverManager


def test_file_manager():
    """Test the `ObserverManager` context manager with multiple `TextObserver` instances."""
    observers: list[TextObserver] = []

    with ObserverManager() as fm:
        assert fm.exitstack is not None

        for i in range(10):
            string_io = StringIO()
            text_observer = TextObserver(string_io, interval=i)

            fm.attach_observer(str(i), text_observer)

            observers.append(text_observer)

    for i in range(10):
        assert observers[i].file.closed
