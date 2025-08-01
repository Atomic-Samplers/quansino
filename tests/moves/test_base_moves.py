from __future__ import annotations

import pytest
from tests.conftest import DummyContext, DummyMove, DummyOperation

from quansino.moves.composite import CompositeMove
from quansino.moves.core import BaseMove


def test_base_move_initialization():
    """Test `BaseMove` initialization with different parameters."""
    move = DummyMove(DummyOperation())
    assert isinstance(move.operation, DummyOperation)
    assert move.operation.name == "DummyOperation"
    assert move.apply_constraints is True
    assert move.max_attempts == 10000

    custom_op = DummyOperation("CustomOperation")
    move = DummyMove(operation=custom_op, apply_constraints=False)
    assert move.operation is custom_op
    assert move.operation.name == "CustomOperation"  # type: ignore
    assert move.apply_constraints is False


def test_base_move_call():
    """Test that `BaseMove` uses the default operation if none is provided."""
    move = BaseMove(DummyOperation())

    assert not hasattr(move, "__dict__")

    with pytest.raises(NotImplementedError):
        move(DummyContext())


def test_base_move_default_operation():
    """Test that `BaseMove` uses the default operation if none is provided."""
    move = BaseMove(DummyOperation())

    with pytest.raises(NotImplementedError):
        _ = move.default_operation


def test_base_move_addition():
    """Test adding `BaseMove`s together."""
    move1 = DummyMove(DummyOperation("1"))
    move2 = DummyMove(DummyOperation("2"))

    composite = move1 + move2

    assert isinstance(composite, CompositeMove)
    assert len(composite.moves) == 2

    assert composite.moves[0].operation.name == "1"  # type: ignore
    assert composite.moves[1].operation.name == "2"  # type: ignore
