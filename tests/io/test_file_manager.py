from __future__ import annotations

from io import StringIO

from quansino.io.core import TextObserver
from quansino.io.file import FileManager


class DummyTextObserver(TextObserver):

    closed_file = 0

    def close(self) -> None:
        DummyTextObserver.closed_file += 1
        super().close()

    def __call__(self, *args, **kwargs) -> None:
        """Dummy call method to simulate writing to a file."""
        self._file.write(f"Dummy write at interval {self.interval}\n")
        self._file.flush()


def test_file_manager():
    observers = []

    with FileManager() as fm:
        assert fm.exitstack is not None

        for i in range(10):
            string_io = StringIO()
            text_observer = DummyTextObserver(string_io, interval=i)
            text_observer.attach_simulation(fm)

            observers.append(text_observer)

    for i in range(10):
        assert observers[i].file.closed

    assert DummyTextObserver.closed_file == 10
