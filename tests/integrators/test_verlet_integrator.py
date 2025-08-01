from __future__ import annotations

import numpy as np
from numpy.testing import assert_almost_equal

from quansino.integrators.displacement import Verlet
from quansino.mc.contexts import DisplacementContext


def test_verlet_integrator(bulk_small, rng):
    """
    Test the Verlet integrator with a small bulk system.

    Parameters
    ----------
    bulk_small : Atoms
        A small bulk system for testing.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """
    context = DisplacementContext(bulk_small, rng=rng)
    integrator = Verlet(dt=0.01, max_steps=20)

    bulk_small.rattle(0.01, rng=rng)

    for _ in range(20):
        energy = bulk_small.get_potential_energy()
        old_positions = bulk_small.get_positions()
        integrator.integrate(context)
        assert_almost_equal(context.atoms.get_potential_energy(), energy, decimal=4)
        assert not np.allclose(context.atoms.get_positions(), old_positions)
