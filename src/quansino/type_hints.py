"""
Module for type hints.

This module provides semantic type hints for various data structures used
throughout the quansino package. These type aliases improve code readability
by clearly defining the expected data types, their intended scientific meaning,
and their expected shapes.

Note
----
Current Python type checkers cannot enforce numpy array shape constraints.
These type hints provide semantic clarity and documentation until native shape
typing becomes available in future Python versions without relying on
third-party libraries.
"""

from __future__ import annotations

from numpy import floating, integer
from numpy.typing import NDArray

IntegerArray = list[int] | tuple[int] | NDArray[integer]
"""1D-Array of integers with shape (N,)."""

AtomicNumbers = NDArray[integer]
"""1D-Array of atomic numbers with shape (N,)."""

UnitCell = NDArray[floating]
"""2D-Array representing a cell matrix with shape (3, 3)."""

AdjacencyMatrix = NDArray[integer]
"""2D-Array representing an adjacency matrix with shape (N, N)."""

Displacement = NDArray[floating]
"""2D-Array representing a displacement vector with shape (3,) or (1, 3)."""

Displacements = NDArray[floating]
"""2D-Array of displacements with shape (N, 3)."""

Center = list[float] | tuple[float] | NDArray[floating]
"""1D-Array representing center point with shape (3,)."""

Strain = NDArray[floating]
"""Array representing a strain tensor with shape (6,) or (3, 3)."""

Stress = NDArray[floating]
"""Array representing a stress tensor with shape (6,) or (3, 3)."""

Deformation = NDArray[floating]
"""2D-Array representing a deformation gradient tensor with shape (3, 3)."""

Forces = NDArray[floating]
"""2D-Array of forces with shape (N, 3)."""

Positions = NDArray[floating]
"""2D-Array of atomic positions with shape (N, 3)."""

Momenta = NDArray[floating]
"""2D-Array of momenta with shape (N, 3)."""

Velocities = NDArray[floating]
"""2D-Array representing velocities with shape (N, 3)."""

Masses = NDArray[floating]
"""1D-Array of masses with shape (N,)."""

ShapedMasses = NDArray[floating]
"""2D-Array of masses with shape (N, 3)."""
