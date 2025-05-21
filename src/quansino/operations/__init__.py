"""
Module for quansino operations.

This module provides various operation classes that can be used in Monte Carlo simulations:

- Cell operations: For modifying the simulation cell shape and size
- Displacement operations: For moving atoms or molecules within the simulation
- Core operation classes: Base classes and composite operation support

Operations can be combined using the + operator or multiplied using the * operator
to create composite operations that perform multiple transformations in sequence.
"""

from __future__ import annotations

from quansino.operations.cell import (
    AnisotropicDeformation,
    IsotropicStretch,
    IsotropicVolume,
    StrainOperation,
)
from quansino.operations.core import CompositeOperation, Operation
from quansino.operations.displacement import (
    Ball,
    Box,
    DisplacementOperation,
    Rotation,
    Sphere,
    Translation,
    TranslationRotation,
)
from quansino.registry import register_class

__all__ = [
    "AnisotropicDeformation",
    "Ball",
    "Box",
    "CompositeOperation",
    "DisplacementOperation",
    "IsotropicStretch",
    "IsotropicVolume",
    "Operation",
    "Rotation",
    "Sphere",
    "StrainOperation",
    "Translation",
    "TranslationRotation",
]

operations_registry = {
    "Ball": Ball,
    "Box": Box,
    "Sphere": Sphere,
    "Translation": Translation,
    "Rotation": Rotation,
    "TranslationRotation": TranslationRotation,
    "IsotropicStretch": IsotropicStretch,
    "IsotropicVolume": IsotropicVolume,
    "AnisotropicDeformation": AnisotropicDeformation,
}

for name, cls in operations_registry.items():
    register_class(cls, name)
