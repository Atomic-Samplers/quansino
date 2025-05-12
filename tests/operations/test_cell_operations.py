from __future__ import annotations

from math import exp

import numpy as np

from quansino.mc.contexts import StrainContext
from quansino.operations.cell import (
    AnisotropicDeformation,
    IsotropicStretch,
    IsotropicVolume,
)


def test_isotropic_strain(bulk_small, rng):
    """
    Test the isotropic strain operation.
    """
    strain_operation = IsotropicStretch(max_value=1.0)
    context = StrainContext(bulk_small, rng=rng)

    for _ in range(10000):
        deformation_gradient = strain_operation.calculate(context)

        assert deformation_gradient.shape == (3, 3)
        assert (
            deformation_gradient[0, 0]
            == deformation_gradient[1, 1]
            == deformation_gradient[2, 2]
        )
        assert (
            deformation_gradient[0, 1]
            == deformation_gradient[0, 2]
            == deformation_gradient[1, 2]
            == 0
        )
        assert np.linalg.det(deformation_gradient) > 0

        assert np.all(deformation_gradient <= 2.0)
        assert np.all(deformation_gradient >= 0.0)


def test_volume_isotropic_strain(bulk_small, rng):
    """
    Test the isotropic strain operation.
    """
    strain_operation = IsotropicVolume(max_value=0.05)
    context = StrainContext(bulk_small, rng=rng)

    for _ in range(10000):
        deformation_gradient = strain_operation.calculate(context)

        assert deformation_gradient.shape == (3, 3)
        assert (
            deformation_gradient[0, 0]
            == deformation_gradient[1, 1]
            == deformation_gradient[2, 2]
        )
        assert (
            deformation_gradient[0, 1]
            == deformation_gradient[0, 2]
            == deformation_gradient[1, 2]
            == 0
        )
        assert np.linalg.det(deformation_gradient) > 0

        assert deformation_gradient[0, 0] ** (3) < exp(0.05)

        assert np.all(deformation_gradient <= exp(0.05))
        assert np.all(deformation_gradient >= 0.0)

        current_volume = bulk_small.cell.volume

        bulk_small.set_cell(
            deformation_gradient @ bulk_small.cell,
            scale_atoms=True,
            apply_constraint=False,
        )

        new_volume = bulk_small.cell.volume

        assert new_volume < current_volume * exp(0.05)


def test_anisotropic_strain(bulk_small, rng):
    """
    Test the isotropic strain operation.
    """
    strain_operation = AnisotropicDeformation(max_value=1 / 3)
    context = StrainContext(bulk_small, rng=rng)

    for _ in range(10000):
        deformation_gradient = strain_operation.calculate(context)

        assert deformation_gradient.shape == (3, 3)
        assert np.linalg.det(deformation_gradient) > 0
