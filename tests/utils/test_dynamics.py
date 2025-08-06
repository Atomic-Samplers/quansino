from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from quansino.mc.contexts import DisplacementContext
from quansino.utils.dynamics import maxwell_boltzmann_distribution


def test_maxwell_boltzmann_distribution(bulk_large, rng):
    context = DisplacementContext(bulk_large, rng=rng)
    context.temperature = 0.0

    maxwell_boltzmann_distribution(context, forced=True)

    assert context.atoms.get_momenta().shape == (len(bulk_large), 3)
    assert context.atoms.get_momenta().dtype == np.float64

    assert_allclose(bulk_large.get_temperature(), 0.0)

    maxwell_boltzmann_distribution(context, forced=False)

    context.temperature = 300.0
    temperatures = []

    for _ in range(50000):
        maxwell_boltzmann_distribution(context, forced=False)
        temperatures.append(bulk_large.get_temperature())

    average_temperature = np.mean(temperatures)

    assert_allclose(average_temperature, 300.0, atol=1.0)
