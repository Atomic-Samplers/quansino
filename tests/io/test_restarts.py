from __future__ import annotations

from pathlib import Path

from ase.io.jsonio import read_json

from quansino.io.restart import RestartObserver
from quansino.mc.isobaric import Isobaric


def test_restart_observer(bulk_small, tmp_path):
    """Test the `RestartObserver` class for saving and loading simulation state."""
    mc = Isobaric(bulk_small, pressure=1.0, temperature=300.0)

    restart_observer = RestartObserver(
        mc,
        file=Path(tmp_path, "restart_test.json"),
        interval=1,
        mode="w",
        encoding="utf-8",
    )

    assert not hasattr(restart_observer, "__dict__")

    restart_observer_fixed = RestartObserver(
        mc,
        file=Path(tmp_path, "restart_test_fixed.json"),
        interval=-6,
        mode="w",
        encoding="utf-8",
    )

    assert restart_observer.interval == 1
    assert restart_observer.mode == "w"
    assert restart_observer.encoding == "utf-8"
    assert restart_observer.file.name == str(Path(tmp_path, "restart_test.json"))
    assert restart_observer.accept_stream is False

    mc.file_manager.attach_observer("restart", restart_observer)
    mc.file_manager.attach_observer("restart_fixed", restart_observer_fixed)

    mc.run(10)

    data: dict = read_json(Path(tmp_path, "restart_test.json"))

    assert data["atoms"] == bulk_small
    assert data["context"]["pressure"] == 1.0
    assert data["context"]["temperature"] == 300.0
    assert data["context"]["last_potential_energy"] == mc.context.last_potential_energy
    assert data["attributes"]["step_count"] == 10

    data_fixed: dict = read_json(Path(tmp_path, "restart_test_fixed.json"))

    assert data_fixed["attributes"]["step_count"] == 6

    assert restart_observer.to_dict() == {
        "name": "RestartObserver",
        "kwargs": {"interval": 1, "mode": "w", "encoding": "utf-8"},
    }

    assert restart_observer_fixed.to_dict() == {
        "name": "RestartObserver",
        "kwargs": {"interval": -6, "mode": "w", "encoding": "utf-8"},
    }

    new_path = Path(tmp_path, "new_restart_test.json")

    restart_observer.file = new_path

    assert restart_observer.file.name == str(new_path)

    mc.run(10)

    assert Path(tmp_path, "new_restart_test.json").exists()
