from __future__ import annotations

import numpy as np
from ase.calculators.calculator import compare_atoms
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import DisplacementContext
from quansino.mc.core import MoveStorage
from quansino.mc.criteria import CanonicalCriteria, Criteria
from quansino.moves.displacements import DisplacementMove
from quansino.operations.displacement import Ball


def test_canonical(bulk_small, dummy_calc, tmp_path):
    """Test that the Canonical class works as expected."""
    bulk_small.calc = dummy_calc

    move = DisplacementMove[Ball, DisplacementContext](
        np.arange(len(bulk_small)), Ball(1.0)
    )

    assert move.operation.step_size == 1.0
    mc = Canonical[DisplacementMove, DisplacementContext](
        bulk_small,
        temperature=0.1,
        default_move=move,
        seed=42,
        logfile=tmp_path / "mc.log",
    )

    # bulk_small.calc = EMT()

    assert mc.temperature == 0.1
    assert mc.num_cycles == len(bulk_small)
    assert mc.atoms == bulk_small
    assert mc.last_results == {}

    np.testing.assert_allclose(mc.context.last_positions, bulk_small.get_positions())

    assert mc.todict() == {
        "type": "monte-carlo",
        "mc-type": "Canonical",
        "seed": 42,
        "rng_state": mc._rng.bit_generator.state,
        "nsteps": 0,
    }

    assert isinstance(mc.moves["default_move"].move.operation, Ball)
    assert isinstance(mc.moves["default_move"].criteria, CanonicalCriteria)

    mc.context.last_energy = -1.0

    assert mc.moves["default_move"].criteria.evaluate(mc.context)
    assert mc.moves["default_move"].probability == 1.0
    assert mc.moves["default_move"].interval == 1
    assert mc.moves["default_move"].minimum_count == 0
    assert_equal(mc.moves["default_move"].move.labels, np.arange(len(bulk_small)))
    assert isinstance(mc.moves["default_move"].move.context, DisplacementContext)
    assert mc.moves["default_move"].move.context.atoms == bulk_small
    assert mc.moves["default_move"].move.max_attempts == 10000

    energy = mc.atoms.get_potential_energy()

    mc.step()

    assert_allclose(mc.last_results["energy"], energy)

    mc.run(10)

    assert mc.nsteps == 10

    mc.temperature = 300.0
    mc.num_cycles = 1

    class DummyCriteria(Criteria):
        def evaluate(self, context: DisplacementContext) -> bool:
            return context.rng.random() < 0.5

    mc.moves["default_move"].criteria = DummyCriteria()

    acceptances = []
    for _ in mc.irun(1000):
        if mc.acceptance_rate:  # Change this ?
            assert mc.atoms.calc is not None
            assert not compare_atoms(mc.atoms.calc.atoms, mc.atoms)
            assert_allclose(mc.context.last_positions, mc.atoms.get_positions())
        else:
            assert mc.atoms.calc is not None
            assert not compare_atoms(mc.atoms.calc.atoms, mc.atoms)
            assert mc.atoms.calc.results.keys() == mc.last_results.keys()
            assert all(
                np.allclose(mc.last_results[k], mc.atoms.calc.results[k])
                for k in mc.last_results
            )
            assert_allclose(mc.context.last_positions, mc.atoms.get_positions())

        acceptances.append(mc.acceptance_rate)

    acceptance_from_log = np.loadtxt(tmp_path / "mc.log", skiprows=13, usecols=-1)

    assert_array_equal(acceptances[1:], acceptance_from_log)
    assert_allclose(np.sum(acceptances), 500, atol=100)

    move_storage = MoveStorage[DisplacementMove](
        move=move,
        interval=4,
        probability=0.4,
        minimum_count=0,
        criteria=DummyCriteria(),
    )

    mc = Canonical[DisplacementMove, DisplacementContext](
        bulk_small, default_move=move_storage, temperature=0.1, seed=42
    )

    assert mc.moves["default_move"].move == move
    assert mc.moves["default_move"].interval == 4
    assert mc.moves["default_move"].probability == 0.4
    assert mc.moves["default_move"].minimum_count == 0
    assert mc.moves["default_move"].move.operation.step_size == 1.0  # type: ignore
