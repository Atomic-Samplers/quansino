"""Core Monte Carlo classes and functions."""

from __future__ import annotations

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import (
    Context,
    DeformationContext,
    DisplacementContext,
    ExchangeContext,
)
from quansino.mc.core import MonteCarlo
from quansino.mc.criteria import (
    BaseCriteria,
    CanonicalCriteria,
    GrandCanonicalCriteria,
    IsobaricCriteria,
)
from quansino.mc.driver import Driver
from quansino.mc.fbmc import ForceBias
from quansino.mc.gcmc import GrandCanonical
from quansino.mc.isobaric import Isobaric
from quansino.registry import register_class

__all__ = [
    "BaseCriteria",
    "Canonical",
    "CanonicalCriteria",
    "Context",
    "DeformationContext",
    "DisplacementContext",
    "Driver",
    "ExchangeContext",
    "ForceBias",
    "GrandCanonical",
    "GrandCanonicalCriteria",
    "Isobaric",
    "IsobaricCriteria",
    "MonteCarlo",
]

mc_registry = {
    "Canonical": Canonical,
    "Isobaric": Isobaric,
    "GrandCanonical": GrandCanonical,
    "ForceBias": ForceBias,
    "DisplacementContext": DisplacementContext,
    "ExchangeContext": ExchangeContext,
    "DeformationContext": DeformationContext,
    "CanonicalCriteria": CanonicalCriteria,
    "IsobaricCriteria": IsobaricCriteria,
    "GrandCanonicalCriteria": GrandCanonicalCriteria,
    "MonteCarlo": MonteCarlo,
}

for name, mc_class in mc_registry.items():
    register_class(mc_class, name)
