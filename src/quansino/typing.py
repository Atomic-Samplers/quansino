from __future__ import annotations

from numpy import dtype, floating, integer, ndarray

IntegerArray = list[int] | tuple[int] | ndarray[tuple[int], dtype[integer]]

AtomicNumbers = ndarray[tuple[int], dtype[integer]]

Cell = ndarray[tuple[3, 3], dtype[floating]]

Connectivity = ndarray[tuple[int, int], dtype[integer]]

Displacement = ndarray[tuple[3], dtype[floating]]

Strain = ndarray[tuple[6], dtype[floating]]
Stress = ndarray[tuple[6], dtype[floating]]

Forces = ndarray[tuple[int, 3], dtype[floating]]
Positions = ndarray[tuple[int, 3], dtype[floating]]
Velocities = ndarray[tuple[int, 3], dtype[floating]]

Masses = ndarray[tuple[int], dtype[floating]] | ndarray[tuple[int, 3], dtype[floating]]
