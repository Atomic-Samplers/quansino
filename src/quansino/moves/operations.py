"""Module for defining operations that can be performed on atoms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quansino.mc.contexts import Context, DisplacementContext, ExchangeContext
    from quansino.typing import Center, Displacement


class Operation(ABC):
    """
    Base class for all operations that can be performed on atoms.

    Methods
    -------
    calculate(context: Any) -> Any
        Calculate the operation to perform on the atoms.

    Notes
    -----
    This class is a base class for all operations that can be performed on atoms. The `calculate` method should be implemented in the subclass, calculating the operation to perform on the atoms.
    """

    @abstractmethod
    def calculate(self, context: Any) -> Any: ...

    def __add__(self, other: Operation) -> CompositeOperation:
        """
        Combine two operations into a single operation.

        Parameters
        ----------
        other : Operation
            The operation to combine with the current operation.

        Returns
        -------
        CompositeOperation
            The combined operation.

        Notes
        -----
        Works with both single operations and composite operations. If the other operation is a composite operation, the operations are combined into a single composite operation.
        """
        if isinstance(other, CompositeOperation):
            return CompositeOperation([self, *other.operations])

        return CompositeOperation([self, other])


class CompositeOperation(Operation):
    """
    Class to combine multiple operations into a single operation.

    Parameters
    ----------
    operations : list[Operation]
        The operations to combine into a single operation.

    Attributes
    ----------
    operations : list[Operation]
        The operations to combine into a single operation

    Methods
    -------
    calculate(context: Any) -> Any
        Calculate the combined operation to perform on the atoms.

    Notes
    -----
    This class is used to combine multiple operations into a single operation. The `calculate` method should be implemented in the subclass, calculating the combined operation to perform on the atoms.
    """

    def __init__(self, operations: list[Operation]) -> None:
        """Initialize the CompositeOperation object."""
        self.operations = operations

    def calculate(self, context: Any) -> Any:
        """
        Calculate the combined operation to perform on the atoms.

        Parameters
        ----------
        context : Any
            The context to use when calculating the operation.

        Returns
        -------
        Any
            The combined operation to perform on the atoms.
        """
        return np.sum([op.calculate(context) for op in self.operations], axis=0)

    def __add__(self, other: Operation) -> CompositeOperation:
        """
        Combine two operations into a single operation.

        Parameters
        ----------
        other : Operation
            The operation to combine with the current operation.

        Returns
        -------
        CompositeOperation
            The combined operation.

        Notes
        -----
        Works with both single operations and composite operations. If the other operation is a composite operation, the operations are combined into a single composite operation.
        """
        if isinstance(other, CompositeOperation):
            return CompositeOperation(self.operations + other.operations)
        else:
            return CompositeOperation([*self.operations, other])


class Translation(Operation):
    """
    Class to perform a translation operation on atoms by performing a random displacement in the cell.
    """

    def calculate(self, context: DisplacementContext) -> Displacement:
        """Calculate the translation operation to perform on the atoms.

        Parameters
        ----------
        context : DisplacementContext
            The context to use when calculating the operation.

        Returns
        -------
        Displacement
            The translation operation to perform on the atoms.
        """
        return context.rng.uniform(
            0, 1, (1, 3)
        ) @ context.atoms.cell.array - context.atoms.positions[
            context.moving_indices
        ].mean(axis=0)


class Rotation(Operation):
    """
    Class to perform a rotation operation on atoms by performing a random Euler rotation around a specified center.

    Parameters
    ----------
    center : str | Center
        The center of the rotation. Can be a string or a Center object.

    Attributes
    ----------
    center : str | Center
        The center of the rotation

    Methods
    -------
    calculate(context: DisplacementContext) -> Displacement
        Calculate the rotation operation to perform on the atoms.
    """

    def __init__(self, center: str | Center = "COM") -> None:
        """Initialize the Rotation object."""
        self.center = center

    def calculate(self, context: DisplacementContext) -> Displacement:
        """
        Calculate the rotation operation to perform on the atoms.

        Parameters
        ----------
        context : DisplacementContext
            The context to use when calculating the operation.

        Returns
        -------
        Displacement
            The rotation operation to perform on the atoms.
        """
        molecule = context.atoms[context.moving_indices]
        phi, theta, psi = context.rng.uniform(0, 2 * np.pi, 3)
        molecule.euler_rotate(phi, theta, psi, center=self.center)  # type: ignore

        return molecule.positions - context.atoms.positions[context.moving_indices]  # type: ignore


class Ball(Operation):
    """
    Class to perform a random displacement in a ball around the origin.

    Parameters
    ----------
    step_size : float
        The maximum distance to displace atoms.

    Attributes
    ----------
    step_size : float
        The maximum distance to displace atoms

    Methods
    -------
    calculate(context: Context) -> Displacement
        Calculate the displacement operation to perform on the atoms.
    """

    def __init__(self, step_size: float = 1.0) -> None:
        """Initialize the Ball object."""
        self.step_size = step_size

    def calculate(self, context: Context) -> Displacement:
        """
        Calculate the displacement operation to perform on the atoms.

        Parameters
        ----------
        context : Context
            The context to use when calculating the operation.

        Returns
        -------
        Displacement
            The displacement operation to perform on the atoms.
        """
        r = context.rng.uniform(0, self.step_size, size=1)
        phi = context.rng.uniform(0, 2 * np.pi, size=1)
        cos_theta = context.rng.uniform(-1, 1, size=1)
        sin_theta = np.sqrt(1 - cos_theta**2)

        return np.column_stack(
            (r * sin_theta * np.cos(phi), r * sin_theta * np.sin(phi), r * cos_theta)
        )


class Sphere(Operation):
    """
    Class to perform a random displacement in a sphere.

    Parameters
    ----------
    step_size : float
        The radius of the sphere.

    Attributes
    ----------
    step_size : float
        The radius of the sphere

    Methods
    -------
    calculate(context: Context) -> Displacement
        Calculate the displacement operation to perform on the atoms.
    """

    def __init__(self, step_size: float = 1.0) -> None:
        """Initialize the Sphere object."""
        self.step_size = step_size

    def calculate(self, context: Context) -> Displacement:
        """
        Calculate the displacement operation to perform on the atoms.

        Parameters
        ----------

        context : Context
            The context to use when calculating the operation.

        Returns
        -------
        Displacement
            The displacement operation to perform on the atoms.
        """
        r = self.step_size
        phi = context.rng.uniform(0, 2 * np.pi, size=1)
        cos_theta = context.rng.uniform(-1, 1, size=1)
        sin_theta = np.sqrt(1 - cos_theta**2)

        return np.column_stack(
            (r * sin_theta * np.cos(phi), r * sin_theta * np.sin(phi), r * cos_theta)
        )


class Box(Operation):
    """
    Class to perform a random displacement in a box.

    Parameters
    ----------
    step_size : float
        The maximum distance to displace atoms.

    Attributes
    ----------
    step_size : float
        The maximum distance to displace atoms

    Methods
    -------
    calculate(context: Context) -> Displacement
        Calculate the displacement operation to perform on the atoms.
    """

    def __init__(self, step_size: float = 1.0) -> None:
        """Initialize the Box object"""
        self.step_size = step_size

    def calculate(self, context: Context) -> Displacement:
        """
        Calculate the displacement operation to perform on the atoms.

        Parameters
        ----------
        context : Context
            The context to use when calculating the operation.

        Returns
        -------
        Displacement
            The displacement operation to perform on the atoms.
        """
        return context.rng.uniform(-self.step_size, self.step_size, size=(1, 3))


class TranslationRotation(Operation):
    """
    Class to perform a translation and rotation operation on atoms.

    Parameters
    ----------
    center : str | Center
        The center of the rotation. Can be a string or a Center object.

    Attributes
    ----------
    translation : Translation
        The translation operation
    rotation : Rotation
        The rotation operation

    Methods
    -------
    calculate(context: DisplacementContext) -> Displacement
        Calculate the translation and rotation operation to perform on the atoms.
    """

    def __init__(self, center: str | Center = "COM") -> None:
        """Initialize the TranslationRotation object."""
        self.translation = Translation()
        self.rotation = Rotation(center)

    def calculate(self, context: DisplacementContext) -> Displacement:
        """
        Calculate the translation and rotation operation to perform on the atoms.

        Parameters
        ----------
        context : DisplacementContext
            The context to use when calculating the operation.

        Returns
        -------
        Displacement
            The translation and rotation operation to perform on the atoms.
        """
        return self.translation.calculate(context) + self.rotation.calculate(context)


class Exchange(Operation):
    """
    Class to perform an exchange operation on atoms.

    Methods
    -------
    addition(context: ExchangeContext) -> None
        Add atoms to the atoms object.
    deletion(context: ExchangeContext) -> None
        Delete atoms from the atoms object.
    """

    def addition(self, context: ExchangeContext) -> None:
        """
        Add atoms to the atoms object.

        Parameters
        ----------
        context : ExchangeContext
            The context to use when adding atoms.
        """
        context.atoms.extend(context.addition_candidates)

    def deletion(self, context: ExchangeContext) -> None:
        """
        Delete atoms from the atoms object.

        Parameters
        ----------
        context : ExchangeContext
            The context to use when deleting atoms.
        """
        del context.atoms[context.deletion_candidates]


class ExchangeTranslation(Translation, Exchange):
    """Class to perform an exchange translation operation on atoms."""


class ExchangeTranslationRotation(TranslationRotation, Exchange):
    """Class to perform an exchange translation and rotation operation on atoms."""
