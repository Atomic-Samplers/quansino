from __future__ import annotations

from tests.conftest import DummyOperation

from quansino.mc.contexts import Context
from quansino.moves.core import BaseMove


def test_base_move(bulk_small, rng):
    """Test the BaseMove class."""

    move = BaseMove(DummyOperation(), apply_constraints=True)
    assert move.max_attempts == 10000
    assert move.operation is not None
    assert move.apply_constraints is True
    assert not hasattr(move, "context")

    context = Context(bulk_small, rng)

    move(context)

    assert move.operation.move_count == 1

    move.check_move = lambda: False

    move(context)

    assert move.operation.move_count == 2
