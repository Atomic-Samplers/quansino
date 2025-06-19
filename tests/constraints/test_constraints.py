from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from quansino.constraints import FixRot


def test_fix_rot(bulk_large, rng):
    """Test the FixRot constraint to ensure it removes the rotation of the system."""
    fix_rot = FixRot()
    bulk_large.set_constraint(fix_rot)

    for _ in range(100):
        momenta = rng.random((len(bulk_large), 3))
        bulk_large.set_momenta(momenta)

        positions_to_com = bulk_large.positions - bulk_large.get_center_of_mass()

        angular_momentum = np.sum(
            np.cross(positions_to_com, bulk_large.get_momenta()), axis=0
        )

        assert_allclose(angular_momentum, 0, atol=1e-9, rtol=0.0)

    assert fix_rot.get_removed_dof(bulk_large) == 3

    assert fix_rot.todict() == {"name": "FixRot", "kwargs": {}}
