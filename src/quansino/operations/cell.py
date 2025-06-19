from __future__ import annotations

from math import exp
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import expm

from quansino.operations.core import BaseOperation

if TYPE_CHECKING:
    from quansino.mc.contexts import Context
    from quansino.type_hints import Deformation


class DeformationOperation(BaseOperation):
    """
    Base class for deformation operations that may modify the simulation cell or atoms' positions.

    Parameters
    ----------
    max_value : float
        The maximum deformation value for the operation, determines the magnitude of the deformation.

    Attributes
    ----------
    max_value : float
        The maximum deformation parameter controlling the deformation magnitude.
    """

    __slots__ = ("max_value",)

    def __init__(self, max_value: float) -> None:
        """Initialize the `DeformationOperation` object."""
        self.max_value = max_value

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the operation to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            The dictionary representation of the operation.
        """
        return {**super().to_dict(), "kwargs": {"max_value": self.max_value}}


class AnisotropicDeformation(DeformationOperation):
    """
    Class for anisotropic deformation operation that can deform the cell differently along each axis.

    This operation applies a deformation tensor with randomly generated components to the simulation cell. The tensor is constructed using exponential mapping to ensure that the resulting deformation is positive definite. If max_value is large, the true maximum component values may exceed it, use with caution.
    """

    def calculate(self, context: Context) -> Deformation:
        """
        Calculate an anisotropic deformation tensor.

        Parameters
        ----------
        context : Context
            The simulation context.

        Returns
        -------
        Deformation
            A 3x3 deformation tensor.
        """
        deformation_tensor = np.zeros((3, 3), dtype=np.float64)
        components = context.rng.uniform(-self.max_value, self.max_value, size=6)

        deformation_tensor[0, 0] = components[0]
        deformation_tensor[1, 1] = components[1]
        deformation_tensor[2, 2] = components[2]

        deformation_tensor[0, 1] = components[3]
        deformation_tensor[0, 2] = components[4]
        deformation_tensor[1, 2] = components[5]

        deformation_tensor[1, 0] = deformation_tensor[0, 1]
        deformation_tensor[2, 0] = deformation_tensor[0, 2]
        deformation_tensor[2, 1] = deformation_tensor[1, 2]

        return expm(deformation_tensor)


class ShapeDeformation(DeformationOperation):
    """
    Class for anisotropic deformation operations that can deform the cell differently along each axis.

    This operation applies a deformation tensor with randomly generated components to the simulation cell.
    The tensor is constructed using exponential mapping to ensure that the resulting deformation is positive definite. If max_value is large, the true maximum component values may exceed it, use with caution.
    """

    def calculate(self, context: Context) -> Deformation:
        """
        Calculate a volume preserving, anisotropic deformation tensor.

        Parameters
        ----------
        context : Context
            The simulation context.

        Returns
        -------
        Deformation
            A 3x3 deformation tensor.
        """
        deformation_tensor = np.zeros((3, 3), dtype=np.float64)
        components = context.rng.uniform(-self.max_value, self.max_value, size=6)

        deformation_tensor[0, 1] = components[3]
        deformation_tensor[0, 2] = components[4]
        deformation_tensor[1, 2] = components[5]

        deformation_tensor[1, 0] = deformation_tensor[0, 1]
        deformation_tensor[2, 0] = deformation_tensor[0, 2]
        deformation_tensor[2, 1] = deformation_tensor[1, 2]

        diag_mean = (components[0] + components[1] + components[2]) / 3.0

        deformation_tensor[0, 0] = components[0] - diag_mean
        deformation_tensor[1, 1] = components[1] - diag_mean
        deformation_tensor[2, 2] = components[2] - diag_mean

        return expm(deformation_tensor)


class IsotropicDeformation(DeformationOperation):
    """
    Class for isotropic deformation operations that stretch or compress the cell equally in all directions.

    This operation applies the same deformation value to all diagonal components of the deformation tensor, resulting in uniform scaling of the simulation cell.
    """

    def calculate(self, context: Context) -> Deformation:
        """
        Calculate an isotropic stretch deformation tensor.

        Parameters
        ----------
        context : Context
            The simulation context.

        Returns
        -------
        Deformation
            A 3x3 deformation tensor with identical diagonal elements.
        """
        return np.eye(3) * exp(context.rng.uniform(-self.max_value, self.max_value))
