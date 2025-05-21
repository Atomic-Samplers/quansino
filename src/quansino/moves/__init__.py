"""Module for quansino moves."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quansino.moves.cell import CellMove
from quansino.moves.core import BaseMove, CompositeMove
from quansino.moves.displacement import CompositeDisplacementMove, DisplacementMove
from quansino.moves.exchange import ExchangeMove
from quansino.registry import register_class

if TYPE_CHECKING:
    from quansino.moves.protocol import BaseProtocol

__all__ = [
    "BaseMove",
    "CellMove",
    "CompositeDisplacementMove",
    "DisplacementMove",
    "ExchangeMove",
]

moves_registry: dict[str, type[BaseProtocol]] = {
    "BaseMove": BaseMove,
    "CellMove": CellMove,
    "CompositeMove": CompositeMove,
    "DisplacementMove": DisplacementMove,
    "CompositeDisplacementMove": CompositeDisplacementMove,
    "ExchangeMove": ExchangeMove,
}

for name, cls in moves_registry.items():
    register_class(cls, name)
