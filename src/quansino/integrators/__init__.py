"""Module for quansino integrators."""

from __future__ import annotations

from quansino.integrators.core import BaseIntegrator
from quansino.integrators.displacement import Verlet
from quansino.registry import register_class

__all__ = ["BaseIntegrator", "Verlet"]

integrators_registry: dict[str, type[BaseIntegrator]] = {
    "BaseIntegrator": BaseIntegrator,
    "Verlet": Verlet,
}

for name, cls in integrators_registry.items():
    register_class(cls, name)
