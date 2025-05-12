from __future__ import annotations

from copy import copy

import numpy as np
import pytest
from ase.build import molecule
from ase.constraints import FixAtoms
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from tests.conftest import DummyOperation

from quansino.mc.contexts import DisplacementContext
from quansino.moves.displacement import CompositeDisplacementMove, DisplacementMove
from quansino.operations import Box, Rotation, Sphere


def test_displacement_move(bulk_small, rng):
    move = DisplacementMove(np.arange(len(bulk_small)))
    context = DisplacementContext(bulk_small, rng)

    dummy_operation = DummyOperation()

    move.operation = dummy_operation

    move.attach_simulation(context)

    assert move.context is not None
    assert move.context.atoms is not None
    assert move.context.rng is not None

    move.to_displace_labels = 0
    move.context.moving_indices = [0, 1]

    move.register_success()

    assert move.to_displace_labels is None
    assert_array_equal(move.context.moving_indices, [0, 1])
    assert move.displaced_labels == 0

    move.labels = [1, 1, 1, 0]

    move.operation.calculate(move.context)

    assert_allclose(bulk_small.get_positions(), np.zeros((len(bulk_small), 3)))

    move_copy = copy(move)

    assert move_copy.context is not None
    assert_array_equal(move_copy.labels, move.labels)
    assert move_copy.operation == move.operation
    assert move_copy.context == move.context


def test_displacement_move_2(bulk_small, rng):
    context = DisplacementContext(bulk_small, rng)

    move = DisplacementMove(np.arange(len(bulk_small)), Sphere(0.1))

    assert not hasattr(move, "context")
    assert move.operation.step_size == 0.1

    move.attach_simulation(context)

    assert move.context is not None
    assert_array_equal(move.labels, np.arange(len(bulk_small)))
    assert_array_equal(move.unique_labels, np.arange(len(bulk_small)))

    old_positions = bulk_small.get_positions()

    assert move()

    norm = np.linalg.norm(bulk_small.get_positions() - old_positions)

    assert_allclose(norm, 0.1)

    old_positions = bulk_small.get_positions()

    move.set_labels([-1, -1, -1, -1])

    assert not move()

    assert_allclose(bulk_small.get_positions(), old_positions)

    move.set_labels([0, 0, 0, 0])

    assert move()

    assert len(np.unique(np.round(bulk_small.get_positions() - old_positions, 5))) == 3
    assert len(move.unique_labels) == 1

    old_positions = bulk_small.get_positions()

    move.context.atoms.set_constraint(FixAtoms([0, 1, 2, 3]))

    random_int = rng.integers(0, len(bulk_small))
    exchangeable_labels = [-1, -1, -1, -1]
    exchangeable_labels[random_int] = 0

    move.labels = exchangeable_labels

    assert move()

    assert move.displaced_labels is not None
    assert_array_equal(move.displaced_labels, 0)

    trues = np.full(len(bulk_small), True)
    trues[random_int] = False

    assert_allclose(bulk_small.get_positions()[trues], old_positions[trues])


def test_displacement_move_3(bulk_small, rng):
    context = DisplacementContext(bulk_small, rng)

    move = DisplacementMove(np.arange(len(bulk_small)), Box(0.1))
    move.attach_simulation(context)

    old_positions = bulk_small.get_positions()

    move.set_labels([0, -1, -1, -1])
    bulk_small.set_constraint(FixAtoms([0]))

    assert move()
    assert_allclose(bulk_small.positions, old_positions)

    move.apply_constraints = False

    assert move()
    new_positions = bulk_small.get_positions()
    assert not np.allclose(new_positions, old_positions)
    assert_allclose(new_positions[1:], old_positions[1:])

    assert_array_less(np.abs(bulk_small.positions[0] - old_positions[0]), 0.1)

    def check_move() -> bool:
        return move.apply_constraints

    move.check_move = check_move
    assert not move()

    assert_allclose(bulk_small.get_positions(), new_positions)

    move.apply_constraints = True
    move.set_labels([-1, 1, 2, -1])

    assert move()

    assert not np.allclose(bulk_small.get_positions(), new_positions)


def test_displacement_move_molecule(rng):
    water = molecule("H2O", vacuum=10)

    context = DisplacementContext(water, rng)

    move = DisplacementMove([0, 0, 0], Sphere(0.1))

    move.context = context

    old_positions = water.get_positions()
    distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    for _ in range(10):
        move()

    assert not np.allclose(water.positions, old_positions)

    old_positions = water.get_positions()
    new_distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    assert_allclose(distances, new_distances)

    move = DisplacementMove(move.labels, Sphere(0.1))
    move.context = context

    move()

    new_distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    assert_allclose(distances, new_distances)
    assert not np.allclose(water.positions, old_positions)

    assert_allclose(np.linalg.norm(water.positions - old_positions, axis=-1), 0.1)

    water += molecule("H2O", vacuum=10)

    old_positions = water.get_positions()
    distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    move.labels = np.array([0, 0, 0, -1, -1, -1])

    for _ in range(1000):
        move()

    new_distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)
    assert not np.allclose(water.positions, old_positions)
    assert_allclose(distances[3:, 3:], new_distances[3:, 3:])
    assert_allclose(distances[:3, :3], new_distances[:3, :3])
    assert_allclose(old_positions[3:], water.positions[3:])

    move_1 = DisplacementMove([0, 0, 0, -1, -1, -1], Rotation())
    move_2 = DisplacementMove([0, 0, 0, -1, -1, -1], Rotation())
    composite_move = move_1 + move_2
    composite_move.context = context

    old_positions = water.get_positions()

    for _ in range(10):
        move()
        new_distances = np.linalg.norm(
            water.positions[:, None] - water.positions, axis=-1
        )
        assert_allclose(distances[3:, 3:], new_distances[3:, 3:])
        assert_allclose(distances[:3, :3], new_distances[:3, :3])

    assert not np.allclose(old_positions, water.positions)


def test_composite_move(bulk_medium, rng):
    context = DisplacementContext(bulk_medium, rng)

    move_1 = DisplacementMove(np.arange(len(bulk_medium)), Sphere(0.1))
    move_2 = DisplacementMove(np.arange(len(bulk_medium)), Sphere(0.1))

    composite_move = move_1 + move_2

    assert isinstance(composite_move, CompositeDisplacementMove)
    assert len(composite_move) == 2

    for move in composite_move:
        move.attach_simulation(context)
        assert isinstance(move, DisplacementMove)
        assert move.context is not None

    composite_move.attach_simulation(DisplacementContext(bulk_medium, rng))

    multiple_moves = move_1 * 5
    multiple_moves_2 = 5 * move_1

    assert len(multiple_moves) == 5
    assert len(multiple_moves_2) == 5

    assert multiple_moves[0].context == multiple_moves_2[0].context

    old_positions = bulk_medium.get_positions()

    assert composite_move()

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

    composite_move_4()

    assert not np.allclose(bulk_medium.get_positions(), old_positions)

    assert (
        np.where(((bulk_medium.get_positions() - old_positions) != 0).all(axis=1))[
            0
        ].shape[0]
        == 5
    )

    bulk_medium.positions = old_positions
    composite_move_2()

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

    composite_move()

    assert composite_move.displaced_labels == [None, 0]
    assert composite_move.number_of_moved_particles == 1

    move_2.check_move = lambda: False

    composite_move()

    assert composite_move.displaced_labels == [None, None]
    assert composite_move.number_of_moved_particles == 0

    composite_move_5 = composite_move * 5

    assert len(composite_move_5) == 10

    move_1.set_labels([0, 1, 2, 3])
    move_2.set_labels([0, 1, 2, 3])

    move_2.check_move = lambda: True

    composite_move_5()

    assert composite_move_5.number_of_moved_particles == 4
    assert sorted(composite_move_5.displaced_labels[:4]) == [0, 1, 2, 3]  # type: ignore
    assert composite_move_5.displaced_labels[4:] == [None] * 6

    composite_move_5.with_replacement = True

    composite_move_5()

    assert composite_move_5.number_of_moved_particles == 10

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
