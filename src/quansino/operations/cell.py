from __future__ import annotations

from math import exp

import numpy as np

from quansino.operations.core import Operation


class StrainOperation(Operation):
    """
    Base class for strain operations that modify the simulation cell.

    Parameters
    ----------
    max_value : float
        The maximum strain value for the operation, determines the magnitude of cell deformation.

    Attributes
    ----------
    max_value : float
        The maximum strain parameter controlling the deformation magnitude.
    """

    def __init__(self, max_value: float) -> None:
        self.max_value = max_value


class AnisotropicDeformation(StrainOperation):
    """
    Class for anisotropic strain operations that can deform the cell differently along each axis.

    This operation applies a strain tensor with randomly generated components to the simulation cell.
    To preserve the orientation of the unit cell, the deformation gradient tensor (F) should be
    positive definite, i.e., det(F) > 0. To ensure this, the `max_value` attribute should
    be less than 1/3.

    Parameters
    ----------
    max_value : float
        The maximum strain value for individual tensor components.

    Returns
    -------
    numpy.ndarray
        A 3x3 strain tensor to apply to the simulation cell.
    """

    def calculate(self, context):
        strain_tensor = np.eye(3)
        components = context.rng.uniform(-self.max_value, self.max_value, size=6)

        strain_tensor[0, 0] += components[0]
        strain_tensor[1, 1] += components[1]
        strain_tensor[2, 2] += components[2]

        strain_tensor[0, 1] = components[3]
        strain_tensor[0, 2] = components[4]
        strain_tensor[1, 2] = components[5]

        strain_tensor[1, 0] = strain_tensor[0, 1]
        strain_tensor[2, 0] = strain_tensor[0, 2]
        strain_tensor[2, 1] = strain_tensor[1, 2]

        return strain_tensor


class IsotropicStretch(StrainOperation):
    """
    Class for isotropic strain operations that stretch or compress the cell equally in all directions.

    This operation applies the same strain value to all diagonal components of the strain tensor,
    resulting in uniform scaling of the simulation cell.

    Parameters
    ----------
    max_value : float
        The maximum strain value to apply.

    Returns
    -------
    numpy.ndarray
        A 3x3 strain tensor with identical diagonal elements.
    """

    def calculate(self, context):
        strain = np.eye(3)
        value = context.rng.uniform(-self.max_value, self.max_value)

        strain[0, 0] += value
        strain[1, 1] += value
        strain[2, 2] += value

        return strain


class IsotropicVolume(StrainOperation):
    """
    Class for isotropic volume changes that preserve the cell shape while changing its volume.

    This operation applies an exponential transformation to ensure positive volume changes
    and provides a uniform scaling factor for the simulation cell in all directions.

    Parameters
    ----------
    max_value : float
        The maximum logarithmic volume change.

    Returns
    -------
    numpy.ndarray
        A 3x3 diagonal matrix with identical scaling factors on the diagonal.
    """

    def calculate(self, context):
        deformation_value = context.rng.uniform(-self.max_value, self.max_value)
        deformation_value = exp(deformation_value) ** (1.0 / 3.0)

        return np.eye(3) * deformation_value
