"""Module for quansino moves."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quansino.moves.cell import CellMove
from quansino.moves.composite import CompositeMove
from quansino.moves.core import BaseMove
from quansino.moves.displacement import CompositeDisplacementMove, DisplacementMove
from quansino.moves.exchange import ExchangeMove
from quansino.registry import register_class

if TYPE_CHECKING:
    from quansino.protocols import Move

__all__ = [
    "BaseMove",
    "CellMove",
    "CompositeDisplacementMove",
    "DisplacementMove",
    "ExchangeMove",
]

moves_registry: dict[str, type[Move]] = {
    "BaseMove": BaseMove,
    "CellMove": CellMove,
    "CompositeMove": CompositeMove,
    "DisplacementMove": DisplacementMove,
    "CompositeDisplacementMove": CompositeDisplacementMove,
    "ExchangeMove": ExchangeMove,
}

for name, cls in moves_registry.items():
    register_class(cls, name)
