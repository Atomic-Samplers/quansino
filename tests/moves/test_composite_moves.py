from __future__ import annotations

import numpy as np
import pytest
from ase.atoms import Atoms
from ase.build import molecule
from numpy.testing import assert_allclose
from tests.conftest import DummyOperation

from quansino.mc.contexts import Context, DisplacementContext, ExchangeContext
from quansino.moves.composite import CompositeMove
from quansino.moves.core import BaseMove
from quansino.moves.displacement import CompositeDisplacementMove, DisplacementMove
from quansino.moves.exchange import CompositeExchangeMove, ExchangeMove
from quansino.operations import Sphere


def test_composite_move(bulk_small, rng):
    """Test the `CompositeMove` class."""
    move1 = BaseMove(DummyOperation(), apply_constraints=True)
    move2 = BaseMove(DummyOperation(), apply_constraints=False)

    composite_move = CompositeMove[BaseMove]([move1, move2])

    assert len(composite_move) == 2

    assert composite_move[0].max_attempts == 10000
    assert composite_move[1].max_attempts == 10000
    assert composite_move[0].operation is not None
    assert composite_move[1].operation is not None
    assert composite_move[0].apply_constraints is True
    assert not hasattr(composite_move[0], "context")
    assert not hasattr(composite_move[1], "context")

    context = Context(bulk_small, rng)

    composite_move(context)

    assert composite_move.moves[0].operation.move_count == 1

    bigger_move = composite_move * 2

    assert len(bigger_move) == 4

    for move in bigger_move:
        assert move.max_attempts == 10000
        assert move.operation is not None

    bigger_move = bigger_move + move1

    assert len(bigger_move) == 5

    bigger_move = bigger_move + composite_move

    assert len(bigger_move) == 7

    with pytest.raises(ValueError):
        composite_move * 0  # type: ignore

    with pytest.raises(TypeError):
        composite_move * "thatwontwork"  # type: ignore

    with pytest.raises(TypeError):
        composite_move + 1  # type: ignore

    with pytest.raises(TypeError):
        composite_move + "thatwontworktoo"  # type: ignore


def test_composite_move_type():
    """Test the type of composite moves."""
    move_1 = BaseMove(DummyOperation())
    move_2 = BaseMove(DummyOperation())

    composite_move = move_1 + move_2

    assert isinstance(composite_move, CompositeMove)

    move_1 = DisplacementMove[DummyOperation, DisplacementContext]([])
    move_2 = DisplacementMove([])

    composite_move = move_2 + move_1

    assert isinstance(composite_move, CompositeDisplacementMove)

    move_1 = ExchangeMove([])
    move_2 = ExchangeMove([])

    composite_move = move_1 + move_2

    assert isinstance(composite_move, CompositeExchangeMove)

    move_2 = DisplacementMove([])

    composite_move = move_1 + move_2

    assert isinstance(composite_move, CompositeMove)

    move_2 = CompositeMove([move_1, move_2])

    composite_move = move_1 + move_2
    move_1.__add__(move_2)

    assert isinstance(composite_move, CompositeMove)

    composite_move_2 = composite_move * 2

    assert isinstance(composite_move_2, CompositeMove)

    composite_move = composite_move_2 + composite_move

    assert isinstance(composite_move, CompositeMove)

    composite_move = CompositeDisplacementMove(
        [move_1] * 2
    ) + CompositeDisplacementMove([move_1] * 2)

    assert isinstance(composite_move, CompositeDisplacementMove)


def test_composite_displacement_move(bulk_medium, rng):
    """Test the `CompositeDisplacementMove` class."""
    context = DisplacementContext(bulk_medium, rng)

    move_1 = DisplacementMove(np.arange(len(bulk_medium)), Sphere(0.1))
    move_2 = DisplacementMove(np.arange(len(bulk_medium)), Sphere(0.1))

    composite_move = move_1 + move_2

    assert isinstance(composite_move, CompositeDisplacementMove)
    assert len(composite_move) == 2

    for move in composite_move:
        assert isinstance(move, DisplacementMove)

    multiple_moves = move_1 * 5
    multiple_moves_2 = 5 * move_1

    assert len(multiple_moves) == 5
    assert len(multiple_moves_2) == 5

    old_positions = bulk_medium.get_positions()

    assert composite_move(context)

    displaced = []
    for move in composite_move:
        assert move.displaced_labels is not None
        displaced.append(move.displaced_labels)

    assert len(np.unique(displaced)) == 2

    assert not np.allclose(bulk_medium.get_positions(), old_positions)
    assert_allclose(
        np.linalg.norm(bulk_medium.positions - old_positions, axis=1)[displaced], 0.1
    )

    composite_move_2 = composite_move + move_1

    assert len(composite_move_2) == 3
    assert composite_move_2[0] == composite_move[0]
    assert composite_move_2[1] == composite_move[1]
    assert composite_move_2[2] == move_1

    composite_move_3 = move_1 + composite_move

    assert len(composite_move_3) == 3
    assert composite_move_3[0] == move_1
    assert composite_move_3[1] == composite_move[0]

    composite_move_4 = composite_move + composite_move_2

    assert len(composite_move_4) == 5
    assert composite_move_4[0] == composite_move[0]
    assert composite_move_4[1] == composite_move[1]
    assert composite_move_4[2] == move_1
    assert composite_move_4[3] == composite_move_2[1]
    assert composite_move_4[4] == move_1

    old_positions = bulk_medium.get_positions()

    composite_move_4(context)

    assert not np.allclose(bulk_medium.get_positions(), old_positions)

    assert (
        np.where(((bulk_medium.get_positions() - old_positions) != 0).all(axis=1))[
            0
        ].shape[0]
        == 5
    )  # type: ignore

    bulk_medium.positions = old_positions
    composite_move_2(context)

    assert isinstance(composite_move_2, CompositeDisplacementMove)
    assert composite_move_2.number_of_moved_particles == 3
    assert not np.allclose(bulk_medium.get_positions(), old_positions)
    assert (
        np.where(((bulk_medium.get_positions() - old_positions) != 0).all(axis=1))[
            0
        ].shape[0]
        == 3
    )

    bulk_medium.positions = old_positions
    del bulk_medium[4:]

    move_1.set_labels([-1, -1, -1, -1])
    move_2.set_labels([-1, -1, -1, 0])

    composite_move(context)

    assert composite_move.displaced_labels == [None, 0]
    assert composite_move.number_of_moved_particles == 1

    move_2.check_move = lambda *args, **kwargs: False

    composite_move(context)

    assert composite_move.displaced_labels == [None, None]
    assert composite_move.number_of_moved_particles == 0

    composite_move_5 = composite_move * 5

    assert len(composite_move_5) == 10

    move_1.set_labels([0, 1, 2, 3])
    move_2.set_labels([0, 1, 2, 3])

    move_2.check_move = lambda *args, **kwargs: True

    composite_move_5(context)

    assert composite_move_5.number_of_moved_particles == 4
    assert sorted(composite_move_5.displaced_labels[:4]) == [0, 1, 2, 3]  # type: ignore
    assert composite_move_5.displaced_labels[4:] == [None] * 6

    move_1.set_labels([-1, 0, -1, 1])
    move_2.set_labels([2, -1, -1, -1])

    composite_move_5(context)

    assert composite_move_5.number_of_moved_particles == 3

    assert np.setdiff1d([0, 1, 2], composite_move_5.displaced_labels[:3]).size == 0  # type: ignore

    with pytest.raises(ValueError):
        move_1 * -2  # type: ignore

    with pytest.raises(ValueError):
        move_1 * 0  # type: ignore

    with pytest.raises(ValueError):
        move_1 * 4.3  # type: ignore

    with pytest.raises(ValueError):
        composite_move * 0  # type: ignore

    with pytest.raises(ValueError):
        composite_move * -1  # type: ignore


def test_composite_exchange_move(empty_atoms, rng):
    """Test the `CompositeExchangeMove` class."""
    empty_atoms.set_cell(np.eye(3) * 10.0)

    context = ExchangeContext(empty_atoms, rng)
    context.exchange_atoms = Atoms("H")

    move_1 = ExchangeMove([])
    move_2 = ExchangeMove([])

    composite_move = move_1 + move_2

    assert isinstance(composite_move, CompositeExchangeMove)
    assert len(composite_move) == 2

    composite_move.bias_towards_insert = 1.0

    composite_move(context)

    assert len(context.atoms) == 2

    assert not np.allclose(context.atoms[0].position, context.atoms[1].position)

    for i in range(50):
        composite_move(context)
        assert len(context.atoms) == 2 + 2 * (i + 1)

    composite_move = composite_move * 2

    assert isinstance(composite_move, CompositeExchangeMove)
    assert len(composite_move) == 4

    for move in composite_move:
        assert isinstance(move, ExchangeMove)

        move.set_labels(np.arange(len(empty_atoms)))

    composite_move.bias_towards_insert = 0.0

    composite_move(context)

    assert len(context.atoms) == 98

    for i in range(20):
        for move in composite_move:
            move.set_labels(np.arange(len(empty_atoms)))

        composite_move(context)

        assert len(context.atoms) == 98 - 4 * (i + 1)

    context.exchange_atoms = molecule("H2O")

    composite_move.bias_towards_insert = 1.0

    composite_move(context)

    assert len(context.atoms) == 18 + 4 * len(context.exchange_atoms)

    for i in range(50):
        composite_move(context)

        assert len(context.atoms) == 30 + 4 * (i + 1) * len(context.exchange_atoms)

    labels = np.full(len(empty_atoms), -1, dtype=int)
    labels[18:] = np.repeat(np.arange(51 * 4), 3)

    for move in composite_move:
        move.set_labels(labels)

    composite_move.bias_towards_insert = 0.0

    composite_move(context)

    assert len(context.atoms) == 630 - 4 * len(context.exchange_atoms)

    for i in range(50):
        for move in composite_move:
            labels = np.full(len(empty_atoms), -1, dtype=int)
            labels[18:] = np.repeat(np.arange(50 * 4 - i * 4), 3)
            move.set_labels(labels)

        composite_move(context)

        assert len(context.atoms) == 618 - 4 * (i + 1) * len(context.exchange_atoms)
        assert context._deleted_indices is not None
        assert context._deleted_atoms is not None

    for move in composite_move:
        move.set_labels(np.full(len(empty_atoms), -1, dtype=int))

    atoms_count = len(context.atoms)

    for _ in range(50):
        composite_move(context)
        assert len(context.atoms) == atoms_count
