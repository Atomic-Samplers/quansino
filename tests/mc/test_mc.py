from __future__ import annotations

from pathlib import Path

import pytest
from ase.calculators.calculator import Calculator

from quansino.mc.core import MonteCarlo
from quansino.mc.criteria import Criteria
from quansino.moves.displacements import DisplacementMove


def test_mc_class(bulk_small):
    mc = MonteCarlo(bulk_small, seed=42)

    assert mc._seed == 42
    assert mc._rng is not None
    assert mc.trajectory is None
    assert mc.atoms == bulk_small
    assert mc.converged()

    mc.validate_simulation()

    assert isinstance(mc.atoms.calc, Calculator)

    del mc.atoms.calc.results

    with pytest.raises(AttributeError):
        mc.validate_simulation()

    mc.atoms.calc = None
    with pytest.raises(AttributeError):
        mc.validate_simulation()

    mc.nsteps = -1
    assert not mc.converged()

    move = DisplacementMove([])

    with pytest.raises(ValueError):
        mc.add_move(move, probability=1.0)


def test_mc_yield_moves(bulk_small):
    mc = MonteCarlo(bulk_small, seed=42)

    class DummyCriteria(Criteria):
        def evaluate(self) -> bool:
            return True

    assert list(mc.yield_moves()) == []

    mc.add_move(
        DisplacementMove([]), criteria=DummyCriteria(), name="my_move", probability=1.0
    )

    assert list(mc.yield_moves()) == ["my_move"]

    mc.add_move(
        DisplacementMove([]),
        criteria=DummyCriteria(),
        name="my_move_2",
        probability=1.0,
        minimum_count=1,
    )

    assert list(mc.yield_moves()) == ["my_move_2"]

    with pytest.raises(ValueError):
        mc.add_move(
            DisplacementMove([]),
            criteria=DummyCriteria(),
            name="my_move",
            minimum_count=1,
        )

    mc.nsteps = 1

    mc.moves["my_move"].interval = 2
    del mc.moves["my_move_2"]

    assert list(mc.yield_moves()) == []

    mc.num_cycles = 100

    for i in range(20):
        mc.add_move(
            DisplacementMove([]),
            criteria=DummyCriteria(),
            name=f"move_{i}",
            minimum_count=2,
        )

    for i in range(20, 50):
        mc.add_move(
            DisplacementMove([]),
            criteria=DummyCriteria(),
            name=f"move_{i}",
            probability=0.5,
        )

    move_list = list(mc.yield_moves())
    assert len(move_list) == 100

    for i in range(20):
        assert move_list.count(f"move_{i}") >= 2


def test_mc_logger(bulk_small, tmp_path):
    mc = MonteCarlo(bulk_small, seed=42, logfile="-", loginterval=1)

    assert not Path("-").exists()
    assert mc.logfile is not None
    assert len(mc.observers) == 1

    mc = MonteCarlo(
        bulk_small, seed=42, logfile=tmp_path / "mc.log", append_trajectory=True
    )

    assert mc.logfile is not None
    assert Path(tmp_path, "mc.log").exists()

    logfile_path_str = str(Path(tmp_path, "mc_str.log"))
    mc = MonteCarlo(
        bulk_small, seed=42, trajectory=tmp_path / "mc.traj", logfile=logfile_path_str
    )

    assert Path(logfile_path_str).exists()
    assert Path(tmp_path, "mc.traj").exists()
    assert mc.trajectory is not None
    assert len(mc.observers) == 2


def test_todict(bulk_small):
    mc = MonteCarlo(bulk_small, seed=42)

    mc.nsteps = 1234
    assert mc.todict() == {
        "type": "monte-carlo",
        "mc-type": "MonteCarlo",
        "seed": 42,
        "rng_state": mc._rng.bit_generator.state,
        "nsteps": 1234,
    }
