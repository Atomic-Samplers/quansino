from __future__ import annotations

from pathlib import Path

from ase.atoms import Atoms
from ase.io import read
from numpy.testing import assert_allclose, assert_array_equal

from quansino.io.trajectory import TrajectoryObserver
from quansino.mc.gcmc import GrandCanonical
from quansino.moves.exchange import ExchangeMove


def test_trajectory_observer(bulk_small, tmp_path):
    """Test the `TrajectoryObserver` class for saving simulation trajectory."""
    mc = GrandCanonical(
        bulk_small,
        exchange_atoms=Atoms("Cu"),
        chemical_potential=100.0,
        temperature=10000.0,
        default_exchange_move=ExchangeMove(labels=list(range(len(bulk_small)))),
    )
    trajectory_observer = TrajectoryObserver(
        bulk_small,
        file=Path(tmp_path, "trajectory_test.xyz"),
        interval=1,
        mode="w",
        encoding="utf-8",
    )

    assert not hasattr(trajectory_observer, "__dict__")

    assert trajectory_observer.interval == 1
    assert trajectory_observer.mode == "w"
    assert trajectory_observer.encoding == "utf-8"
    assert trajectory_observer.file.name == str(Path(tmp_path, "trajectory_test.xyz"))
    assert TrajectoryObserver.accept_stream

    mc.attach_observer("trajectory", trajectory_observer)

    atoms_list: list[Atoms] = []

    for step in mc.irun(10):
        for _ in step:
            pass

        atoms_list.append(mc.atoms.copy())

    atoms_list_from_traj: list[Atoms] = read(
        Path(tmp_path, "trajectory_test.xyz"), index="1:"
    )  # type: ignore

    assert len(atoms_list_from_traj) == len(atoms_list)

    for atoms_1, atoms_2 in zip(atoms_list, atoms_list_from_traj, strict=True):
        assert_allclose(
            atoms_1.get_positions(), atoms_2.get_positions(), rtol=1e-5, atol=1e-8
        )
        assert_allclose(atoms_1.get_cell(), atoms_2.get_cell(), rtol=1e-5, atol=1e-8)
        assert_array_equal(atoms_1.get_pbc(), atoms_2.get_pbc())
        assert_array_equal(
            atoms_1.get_chemical_symbols(), atoms_2.get_chemical_symbols()
        )
