from __future__ import annotations

import numpy as np
from ase.units import kB

from quansino.mc.canonical import Canonical


def test_canonical(bulk_small):
    """Test that the ForceBias class works as expected."""
    mc = Canonical(bulk_small, temperature=300.0, seed=42)

    assert mc.temperature == 300.0 * kB
    assert mc.num_cycles == len(bulk_small)
    assert mc.atoms == bulk_small
    assert mc.last_results == {}
    np.testing.assert_allclose(mc.last_positions, bulk_small.get_positions())
    assert mc.acceptance_rate == 0

    assert mc.todict() == {
        "temperature": 300.0,
        "type": "monte-carlo",
        "mc-type": "Canonical",
        "seed": 42,
        "rng_state": mc._rng.bit_generator.state,
        "nsteps": 0,
    }

    assert mc.get_metropolis_criteria(0.0)
    assert mc.get_metropolis_criteria(-1.0)

    assert mc.moves["default"].minimum_count == 0
    assert mc.moves["default"].interval == 1
    assert mc.moves["default"].probability == 1.0

    assert mc.moves["default"].move.move_type == "box"
    assert mc.moves["default"].move.delta == 0.1
    np.testing.assert_equal(
        mc.moves["default"].move.moving_indices, np.arange(len(bulk_small))
    )
    assert mc.moves["default"].move.moving_per_step == 1
    assert mc.moves["default"].move.context is not None
    assert mc.moves["default"].move._to_move is None

    mc.step()

    assert mc.last_results.get("energy")

    mc.run(10)

    assert mc.nsteps == 10
