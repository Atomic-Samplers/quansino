from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from quansino.operations.core import BaseOperation

if TYPE_CHECKING:
    from ase.atoms import Atoms

    from quansino.mc.contexts import DisplacementContext
    from quansino.type_hints import Displacement


class DisplacementOperation(BaseOperation):
    """
    Base class for displacement operations.

    Parameters
    ----------
    step_size : float, optional
        The step size for the displacement operation (default is 1.0).

    Attributes
    ----------
    step_size : float
        The step size for the displacement operation.
    """

    __slots__ = ("step_size",)

    def __init__(self, step_size: float = 1.0) -> None:
        self.step_size = step_size

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the operation to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the operation.
        """
        return {**super().to_dict(), "kwargs": {"step_size": self.step_size}}


class Box(DisplacementOperation):
    """
    Class for a box-shaped displacement operation.

    This operation generates random displacements within a box with dimensions
    determined by the step_size parameter. Displacements are uniformly
    distributed in the range [-step_size, step_size] along each axis.

    Parameters
    ----------
    step_size : float, optional
        The maximum displacement along each axis (default is 1.0).

    Returns
    -------
    Displacement
        A box-shaped displacement vector for the selected atoms.
    """

    def calculate(self, context: DisplacementContext) -> Displacement:
        return context.rng.uniform(-self.step_size, self.step_size, size=(1, 3))


class Sphere(DisplacementOperation):
    """
    Class for a spherical displacement operation that places atoms on the surface of a sphere.

    This operation generates displacements where atoms are placed exactly at a distance
    equal to step_size from the origin, creating a spherical shell pattern.

    Parameters
    ----------
    step_size : float, optional
        The radius of the sphere (default is 1.0).

    Returns
    -------
    Displacement
        A displacement vector placing atoms on the surface of a sphere.
    """

    def calculate(self, context: DisplacementContext) -> Displacement:
        """
        Calculate the spherical operation to perform on the atoms.

        Parameters
        ----------
        context : DisplacementContext
            The context to use when calculating the operation.

        Returns
        -------
        Displacement
            The spherical operation to perform on the atoms.
        """
        phi = context.rng.uniform(0, 2 * np.pi, size=1)
        cos_theta = context.rng.uniform(-1, 1, size=1)
        sin_theta = np.sqrt(1 - cos_theta**2)

        return self.step_size * np.column_stack(
            (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)
        )


class Ball(DisplacementOperation):
    """
    Class for a ball-shaped displacement operation that places atoms within a sphere.

    Unlike the Sphere operation, this operation generates displacements where atoms
    can be placed anywhere within the volume of a sphere with maximum radius equal
    to step_size.

    Parameters
    ----------
    step_size : float, optional
        The maximum radius of the ball (default is 1.0).

    Returns
    -------
    Displacement
        A displacement vector placing atoms within the volume of a sphere.
    """

    def calculate(self, context: DisplacementContext) -> Displacement:
        """
        Calculate the spherical operation to perform on the atoms.

        Parameters
        ----------
        context : DisplacementContext
            The context to use when calculating the operation.

        Returns
        -------
        Displacement
            The spherical operation to perform on the atoms.
        """
        r = context.rng.uniform(0, self.step_size, size=1)
        phi = context.rng.uniform(0, 2 * np.pi, size=1)
        cos_theta = context.rng.uniform(-1, 1, size=1)
        sin_theta = np.sqrt(1 - cos_theta**2)

        return np.column_stack(
            (r * sin_theta * np.cos(phi), r * sin_theta * np.sin(phi), r * cos_theta)
        )


class Translation(BaseOperation):
    """
    Class for a translation operation.

    This operation moves atoms to a random position within the simulation cell.
    The center of mass of the selected atoms is preserved by applying the same
    displacement to all atoms in the selection.

    Returns
    -------
    Displacement
        A displacement vector that translates the selected atoms to a random position.
    """

    def calculate(self, context: DisplacementContext) -> Displacement:
        atoms = context.atoms

        return context.rng.uniform(0, 1, (1, 3)) @ atoms.cell.array - atoms.positions[
            context._moving_indices
        ].mean(axis=0)


class Rotation(BaseOperation):
    """
    Class for a rotation operation.

    This operation rotates the selected atoms around their center of mass using
    randomly generated Euler angles. The rotation is performed around the
    center of mass of the selected atoms.

    Returns
    -------
    Displacement
        A displacement vector representing the rotational movement of the selected atoms.
    """

    def calculate(self, context: DisplacementContext) -> Displacement:
        atoms = context.atoms

        molecule = cast("Atoms", atoms[context._moving_indices])
        phi, theta, psi = context.rng.uniform(0, 2 * np.pi, 3)
        molecule.euler_rotate(phi, theta, psi, center="COM")

        return molecule.positions - context.atoms.positions[context._moving_indices]


class TranslationRotation(BaseOperation):
    """
    Class to perform both translation and rotation operations on atoms.

    This operation combines the Translation and Rotation operations,
    allowing atoms to be both translated and rotated in a single move.

    Attributes
    ----------
    translation : Translation
        The translation operation.
    rotation : Rotation
        The rotation operation.

    Returns
    -------
    Displacement
        A displacement vector combining both translation and rotation effects.
    """

    def __init__(self):
        self.translation = Translation()
        self.rotation = Rotation()

    def calculate(self, context: DisplacementContext) -> Displacement:
        return self.translation.calculate(context) + self.rotation.calculate(context)
