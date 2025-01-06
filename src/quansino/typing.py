"""Module for type hints.

This module provides type hints for various data structures used throughout the quansino package.
These type hints help ensure type safety and improve code readability by clearly defining
the expected data types and structures.

Examples
--------
``` python
from quansino.typing import Positions, Forces, Cell

positions: Positions = np.zeros((100, 3))
forces: Forces = np.random.random((100, 3))
cell: Cell = np.eye(3) * 10.0
```
"""

from __future__ import annotations

from numpy import dtype, floating, integer, ndarray

IntegerArray = list[int] | tuple[int] | ndarray[tuple[int], dtype[integer]]
"""Type hint for an array of integers."""

AtomicNumbers = ndarray[tuple[int], dtype[integer]]
"""Type hint for an array of atomic numbers."""

Cell = ndarray[tuple[3, 3], dtype[floating]]
"""Type hint for a 3x3 array of floating point numbers representing a cell."""

Connectivity = ndarray[tuple[int, int], dtype[integer]]
"""Type hint for an array of integer pairs representing atom connectivity."""

Displacement = ndarray[tuple[3], dtype[floating]]
"""Type hint for a 3D displacement vector."""

Center = list[float] | tuple[float] | ndarray[tuple[3], dtype[floating]]
"""Type hint for a 3D center point."""

Strain = ndarray[tuple[6], dtype[floating]]
"""Type hint for a 6-element array of floating point numbers representing a strain tensor."""

Stress = ndarray[tuple[6], dtype[floating]]
"""Type hint for a 6-element array of floating point numbers representing a stress tensor."""

Forces = ndarray[tuple[int, 3], dtype[floating]]
"""Type hint for an array of 3D force vectors."""

Positions = ndarray[tuple[int, 3], dtype[floating]]
"""Type hint for an array of 3D positions."""

Velocities = ndarray[tuple[int, 3], dtype[floating]]
"""Type hint for an array of 3D velocities."""

Masses = ndarray[tuple[int], dtype[floating]] | ndarray[tuple[int, 3], dtype[floating]]
"""Type hint for an array of masses."""
