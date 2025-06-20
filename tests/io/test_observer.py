from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tests.conftest import DummyStream

from quansino.io.core import Observer, TextObserver


def test_observers(tmp_path):
    """Test the `Observer` and `TextObserver` classes."""
    observer = Observer(interval=1)
    assert observer.interval == 1

    assert not hasattr(observer, "__dict__")

    file_path = Path(tmp_path, "test.txt")
    text_observer = TextObserver(file=file_path, interval=1)
    assert text_observer.interval == 1

    assert text_observer.file.name == str(file_path)

    assert text_observer.mode == "a"
    assert text_observer.encoding == "utf-8"
    assert text_observer.accept_stream

    text_observer.close()

    assert text_observer.file.closed

    text_observer.file = sys.stdout
    text_observer.close()

    assert not sys.stdout.closed

    with pytest.raises(TypeError):
        text_observer.file = 123  # type: ignore

    with pytest.raises(TypeError):
        text_observer.file = None  # type: ignore

    TextObserver.accept_stream = False

    with pytest.raises(ValueError):
        text_observer = TextObserver(file=DummyStream(), interval=1)  # type: ignore

    TextObserver.accept_stream = True

    assert not text_observer.file.closed

    assert text_observer.to_dict() == {
        "name": "TextObserver",
        "kwargs": {"interval": 1, "mode": "a", "encoding": "utf-8"},
    }


def test_text_observer_repr() -> None:
    """Test the `__repr__` method of the `TextObserver` class."""
    observer = TextObserver(DummyStream(), interval=1)  # type: ignore
    assert (
        repr(observer)
        == "TextObserver(Class:DummyStream, mode=a, encoding=utf-8, interval=1)"
    )

    text_observer = TextObserver(file=Path("weird_file.wrd"), interval=1)
    assert (
        repr(text_observer)
        == "TextObserver(Path:weird_file.wrd, mode=a, encoding=utf-8, interval=1)"
    )
