"""Core Monte Carlo classes and functions."""

from __future__ import annotations

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import (
    Context,
    DisplacementContext,
    ExchangeContext,
    StrainContext,
)
from quansino.mc.core import MonteCarlo, MoveStorage
from quansino.mc.criteria import (
    CanonicalCriteria,
    Criteria,
    GrandCanonicalCriteria,
    IsobaricCriteria,
)
from quansino.mc.fbmc import ForceBias
from quansino.mc.gcmc import GrandCanonical
from quansino.mc.isobaric import Isobaric
from quansino.registry import register_class

__all__ = [
    "Canonical",
    "CanonicalCriteria",
    "Context",
    "Criteria",
    "DisplacementContext",
    "ExchangeContext",
    "ForceBias",
    "GrandCanonical",
    "GrandCanonicalCriteria",
    "Isobaric",
    "IsobaricCriteria",
    "MonteCarlo",
    "MoveStorage",
    "StrainContext",
]

mc_registry = {
    "Canonical": Canonical,
    "Isobaric": Isobaric,
    "GrandCanonical": GrandCanonical,
    "ForceBias": ForceBias,
    "DisplacementContext": DisplacementContext,
    "ExchangeContext": ExchangeContext,
    "StrainContext": StrainContext,
    "CanonicalCriteria": CanonicalCriteria,
    "IsobaricCriteria": IsobaricCriteria,
    "GrandCanonicalCriteria": GrandCanonicalCriteria,
    "MonteCarlo": MonteCarlo,
    "MoveStorage": MoveStorage,
}

for name, mc_class in mc_registry.items():
    register_class(mc_class, name)
