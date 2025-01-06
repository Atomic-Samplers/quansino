from __future__ import annotations

import numpy as np
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from numpy.testing import assert_allclose, assert_equal

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import DisplacementContext
from quansino.moves.displacements import Ball, DisplacementMove


def test_canonical(bulk_small, rng):
    """Test that the ForceBias class works as expected."""
    mc = Canonical(bulk_small, temperature=0.1, seed=42)

    bulk_small.calc = EMT()

    assert mc.temperature == 0.1
    assert mc.num_cycles == len(bulk_small)
    assert mc.atoms == bulk_small
    assert mc.last_results == {}
    np.testing.assert_allclose(mc.last_positions, bulk_small.get_positions())
    assert mc.acceptance_rate == 0

    assert mc.todict() == {
        "temperature": 0.1,
        "type": "monte-carlo",
        "mc-type": "Canonical",
        "seed": 42,
        "rng_state": mc._rng.bit_generator.state,
        "nsteps": 0,
    }

    assert mc.get_metropolis_criteria(0.0)
    assert mc.get_metropolis_criteria(-1.0)

    assert isinstance(mc.moves["default"], DisplacementMove)

    assert mc.move_probabilities["default"] == 1.0
    assert mc.move_intervals["default"] == 1
    assert mc.move_minimum_count["default"] == 0

    assert isinstance(mc.moves["default"].move_operator, Ball)
    assert mc.moves["default"].move_operator.step_size == 1.0

    mc.moves["default"].move_operator.step_size = 0.1

    assert_equal(mc.moves["default"].candidate_indices, np.arange(len(bulk_small)))

    assert mc.moves["default"].displacements_per_move == 1
    assert isinstance(mc.moves["default"].context, DisplacementContext)

    assert mc.moves["default"].context.atoms == bulk_small

    assert mc.moves["default"].max_attempts == 10000

    energy = mc.atoms.get_potential_energy()

    mc.step()

    assert_allclose(mc.last_results["energy"], energy)

    mc.run(10)

    assert mc.nsteps == 10

    mc.temperature = 300.0
    mc.num_cycles = 1

    mc.get_metropolis_criteria = lambda energy_difference: rng.random() < 0.5

    acceptances = 0
    for _ in mc.irun(1000):
        acceptances += mc.acceptance_rate

        if mc.acceptance_rate:
            assert mc.atoms.calc is not None
            assert not compare_atoms(mc.atoms.calc.atoms, mc.atoms)
            assert_allclose(mc.last_positions, mc.atoms.get_positions())
        else:
            assert mc.atoms.calc is not None
            assert not compare_atoms(mc.atoms.calc.atoms, mc.atoms)
            assert mc.atoms.calc.results.keys() == mc.last_results.keys()
            assert all(
                np.allclose(mc.last_results[k], mc.atoms.calc.results[k])
                for k in mc.last_results
            )
            assert_allclose(mc.last_positions, mc.atoms.get_positions())

    assert_allclose(acceptances, 500, atol=100)
