from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from quansino.mc.contexts import Context, DisplacementContext

if TYPE_CHECKING:
    from quansino.type_hints import Displacement


class Operation[ContextType: Context]:
    """
    Abstract base class for operations.
    """

    @abstractmethod
    def calculate(self, context: ContextType) -> Displacement: ...

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
        else:
            return CompositeOperation([self, other])

    def __mul__(self, n: int) -> CompositeOperation:
        """
        Multiply the displacement move by an integer to create a composite move.

        Parameters
        ----------
        n : int
            The number of times to repeat the move.

        Returns
        -------
        CompositeDisplacementMove
            The composite move.
        """
        if n < 1 or not isinstance(n, int):
            raise ValueError(
                "The number of times the move is repeated must be a positive, non-zero integer."
            )
        return CompositeOperation([self] * n)


class DisplacementOperation[ContextType: DisplacementContext](Operation[ContextType]):
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

    def __init__(self, step_size: float = 1.0) -> None:
        self.step_size = step_size


class CompositeOperation[ContextType: Context](Operation[ContextType]):
    """
    Class to combine multiple operations into a single operation.

    Parameters
    ----------
    operations : list[Operation]
        The operations to combine into a single operation.

    Attributes
    ----------
    operations : list[Operation]
        The operations to combine into a single operation.
    """

    def __init__(self, operations: list[Operation[ContextType]]) -> None:
        """Initialize the CompositeOperation object."""
        self.operations = operations

    def calculate(self, context: ContextType) -> Displacement:
        """
        Calculate the combined operation to perform on the atoms.

        Parameters
        ----------
        context : ContextType
            The context to use when calculating the operation.

        Returns
        -------
        Displacement
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

    def __mul__(self, n: int) -> CompositeOperation:
        """
        Multiply the displacement move by an integer to create a composite move.

        Parameters
        ----------
        n : int
            The number of times to repeat the move.

        Returns
        -------
        CompositeDisplacementMove
            The composite move.
        """
        if n < 1 or not isinstance(n, int):
            raise ValueError(
                "The number of times the move is repeated must be a positive, non-zero integer."
            )
        return type(self)(self.operations * n)

    def __getitem__(self, index: int) -> Operation:
        """
        Get the move at the specified index.

        Parameters
        ----------
        index : int
            The index of the move.

        Returns
        -------
        DisplacementMove
            The move at the specified index.
        """
        return self.operations[index]

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self):
        return iter(self.operations)

    __rmul__ = __mul__

    __imul__ = __mul__


class Box(DisplacementOperation[DisplacementContext]):
    """Class for a box-shaped displacement operation."""

    def calculate(self, context: DisplacementContext) -> Displacement:
        return context.rng.uniform(-self.step_size, self.step_size, size=(1, 3))


class Sphere(DisplacementOperation[DisplacementContext]):
    """Class for a spherical operation that calculates the displacement of atoms within a sphere."""

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


class Ball(DisplacementOperation[DisplacementContext]):
    """Class for a spherical operation that calculates the displacement of atoms within a sphere."""

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


class Translation(Operation[DisplacementContext]):
    """Class for a translation operation."""

    def calculate(self, context: DisplacementContext) -> Displacement:
        atoms = context.atoms

        return context.rng.uniform(0, 1, (1, 3)) @ atoms.cell.array - atoms.positions[
            context.moving_indices
        ].mean(axis=0)


class Rotation(Operation[DisplacementContext]):
    """Class for a rotation operation."""

    def calculate(self, context: DisplacementContext) -> Displacement:
        atoms = context.atoms

        molecule = atoms[context.moving_indices]
        phi, theta, psi = context.rng.uniform(0, 2 * np.pi, 3)
        molecule.euler_rotate(phi, theta, psi, center="COM")  # type: ignore

        return molecule.positions - context.atoms.positions[context.moving_indices]  # type: ignore


class TranslationRotation(Operation):
    """
    Class to perform a translation and rotation operation on atoms.

    Attributes
    ----------
    translation : Translation
        The translation operation.
    rotation : Rotation
        The rotation operation.
    """

    def __init__(self):
        self.translation = Translation()
        self.rotation = Rotation()

    def calculate(self, context: DisplacementContext) -> Displacement:
        return self.translation.calculate(context) + self.rotation.calculate(context)
