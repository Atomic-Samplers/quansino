"""Core Monte Carlo classes and functions."""

from __future__ import annotations

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import (
    Context,
    DeformationContext,
    DisplacementContext,
    ExchangeContext,
    HamiltonianContext,
    HamiltonianDeformationContext,
    HamiltonianDisplacementContext,
    HamiltonianExchangeContext,
)
from quansino.mc.core import MonteCarlo
from quansino.mc.criteria import (
    BaseCriteria,
    CanonicalCriteria,
    GrandCanonicalCriteria,
    IsobaricCriteria,
)
from quansino.mc.driver import Driver
from quansino.mc.fbmc import AdaptiveForceBias, ForceBias
from quansino.mc.gcmc import GrandCanonical
from quansino.mc.isobaric import Isobaric
from quansino.registry import register_class

__all__ = [
    "AdaptiveForceBias",
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
    "HamiltonianContext",
    "HamiltonianDeformationContext",
    "HamiltonianDisplacementContext",
    "HamiltonianExchangeContext",
    "Isobaric",
    "IsobaricCriteria",
    "MonteCarlo",
]

mc_registry = {
    "AdaptiveForceBias": AdaptiveForceBias,
    "Driver": Driver,
    "HamiltonianContext": HamiltonianContext,
    "HamiltonianDeformationContext": HamiltonianDeformationContext,
    "HamiltonianDisplacementContext": HamiltonianDisplacementContext,
    "HamiltonianExchangeContext": HamiltonianExchangeContext,
    "BaseCriteria": BaseCriteria,
    "Canonical": Canonical,
    "CanonicalCriteria": CanonicalCriteria,
    "Context": Context,
    "DeformationContext": DeformationContext,
    "DisplacementContext": DisplacementContext,
    "ExchangeContext": ExchangeContext,
    "Isobaric": Isobaric,
    "GrandCanonical": GrandCanonical,
    "ForceBias": ForceBias,
    "IsobaricCriteria": IsobaricCriteria,
    "GrandCanonicalCriteria": GrandCanonicalCriteria,
    "MonteCarlo": MonteCarlo,
}

for name, mc_class in mc_registry.items():
    register_class(mc_class, name)
