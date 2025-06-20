from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from quansino.mc.contexts import DeformationContext
from quansino.operations.cell import (
    AnisotropicDeformation,
    IsotropicDeformation,
    ShapeDeformation,
)


def test_isotropic_deformation(bulk_small, rng):
    """Test isotropic deformation operation."""
    strain_operation = IsotropicDeformation(max_value=0.1)
    context = DeformationContext(bulk_small, rng=rng)

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

        assert_allclose(
            deformation_gradient - deformation_gradient.T, 0, atol=1e-12, rtol=0.0
        )


def test_anisotropic_deformation(bulk_small, rng):
    """Test anisotropic deformation operation."""
    strain_operation = AnisotropicDeformation(max_value=1 / 3)
    context = DeformationContext(bulk_small, rng=rng)

    for _ in range(10000):
        deformation_gradient = strain_operation.calculate(context)

        assert deformation_gradient.shape == (3, 3)
        assert np.linalg.det(deformation_gradient) > 0

        assert_allclose(
            deformation_gradient - deformation_gradient.T, 0.0, atol=1e-12, rtol=0.0
        )


def test_shape_deformation(bulk_small, rng):
    """Test the shape deformation operation."""
    strain_operation = ShapeDeformation(max_value=1 / 3)
    context = DeformationContext(bulk_small, rng=rng)

    for _ in range(10000):
        deformation_gradient = strain_operation.calculate(context)

        assert deformation_gradient.shape == (3, 3)
        assert_allclose(np.linalg.det(deformation_gradient), 1.0, atol=1e-12, rtol=0.0)

        assert_allclose(
            deformation_gradient - deformation_gradient.T, 0.0, atol=1e-12, rtol=0.0
        )
