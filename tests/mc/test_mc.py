from __future__ import annotations

from pathlib import Path

import pytest

from quansino.mc.core import MonteCarlo


def test_mc_class(atoms):
    """Test that the ForceBias class works as expected."""
    mc = MonteCarlo(atoms, seed=42)

    with pytest.raises(AttributeError):
        assert MonteCarlo.__rng

    with pytest.raises(AttributeError):
        assert MonteCarlo.__seed

    assert mc._MonteCarlo__seed == 42
    assert mc._MonteCarlo__rng is not None
    assert mc.trajectory is None
    assert mc.atoms == atoms
    assert mc.converged()

    mc.nsteps = -1
    assert not mc.converged()


def test_mc_logger(atoms, tmp_path):
    mc = MonteCarlo(atoms, seed=42, logfile="-", loginterval=1)

    assert not Path("-").exists()
    assert mc.logfile is not None
    assert len(mc.observers) == 1

    mc = MonteCarlo(atoms, seed=42, logfile=tmp_path / "mc.log", append_trajectory=True)

    assert mc.logfile is not None
    assert Path(tmp_path, "mc.log").exists()

    logfile_path_str = str(Path(tmp_path, "mc_str.log"))
    mc = MonteCarlo(
        atoms, seed=42, trajectory=tmp_path / "mc.traj", logfile=logfile_path_str
    )

    assert Path(logfile_path_str).exists()
    assert Path(tmp_path, "mc.traj").exists()
    assert mc.trajectory is not None
    assert len(mc.observers) == 2


def test_todict(atoms):
    mc = MonteCarlo(atoms, seed=42)

    mc.nsteps = 1234
    assert mc.todict() == {
        "type": "monte-carlo",
        "mc-type": "MonteCarlo",
        "seed": 42,
        "rng_state": mc._MonteCarlo__rng.bit_generator.state,
        "nsteps": 1234,
    }
