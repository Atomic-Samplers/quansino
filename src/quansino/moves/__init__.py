"""Module for quansino moves."""

from __future__ import annotations

from quansino.moves.cell import CellMove
from quansino.moves.core import BaseMove
from quansino.moves.displacement import CompositeDisplacementMove, DisplacementMove
from quansino.moves.exchange import ExchangeMove

__all__ = [
    "BaseMove",
    "CellMove",
    "CompositeDisplacementMove",
    "DisplacementMove",
    "ExchangeMove",
]

moves_registry: dict[str, type[BaseMove]] = {
    "CellMove": CellMove,
    "DisplacementMove": DisplacementMove,
    "CompositeDisplacementMove": CompositeDisplacementMove,
    "ExchangeMove": ExchangeMove,
}
