from __future__ import annotations

from copy import copy

import numpy as np
from ase.build import molecule
from ase.constraints import FixAtoms
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from tests.conftest import DummyOperation

from quansino.mc.contexts import DisplacementContext
from quansino.moves.displacement import DisplacementMove
from quansino.operations import Box, Sphere
from quansino.registry import register_class


def test_displacement_move(bulk_small, rng):
    """Test the `DisplacementMove` class."""
    move = DisplacementMove[DummyOperation, DisplacementContext](
        np.arange(len(bulk_small))
    )
    context = DisplacementContext(bulk_small, rng)

    dummy_operation = DummyOperation()

    register_class(DummyOperation, "DummyOperation")

    move.operation = dummy_operation

    move.to_displace_labels = 0
    context._moving_indices = [0, 1]

    move.register_success()

    assert move.to_displace_labels is None
    assert_array_equal(context._moving_indices, [0, 1])
    assert move.displaced_labels == 0

    move.labels = [1, 1, 1, 0]

    move.operation.calculate(context)

    assert_allclose(bulk_small.get_positions(), np.zeros((len(bulk_small), 3)))

    move_copy = copy(move)

    assert_array_equal(move_copy.labels, move.labels)
    assert isinstance(move_copy.operation, DummyOperation)


def test_displacement_move_2(bulk_small, rng):
    """Test the `DisplacementMove` class with a sphere operation."""
    context = DisplacementContext(bulk_small, rng)

    move = DisplacementMove(np.arange(len(bulk_small)), Sphere(0.1))

    assert not hasattr(move, "context")
    assert move.operation.step_size == 0.1

    assert_array_equal(move.labels, np.arange(len(bulk_small)))
    assert_array_equal(move.unique_labels, np.arange(len(bulk_small)))

    old_positions = bulk_small.get_positions()

    assert move(context)

    norm = np.linalg.norm(bulk_small.get_positions() - old_positions)

    assert_allclose(norm, 0.1)

    old_positions = bulk_small.get_positions()

    move.set_labels([-1, -1, -1, -1])

    assert not move(context)

    assert_allclose(bulk_small.get_positions(), old_positions)

    move.set_labels([0, 0, 0, 0])

    assert move(context)

    assert len(np.unique(np.round(bulk_small.get_positions() - old_positions, 5))) == 3
    assert len(move.unique_labels) == 1

    old_positions = bulk_small.get_positions()

    context.atoms.set_constraint(FixAtoms([0, 1, 2, 3]))

    random_int = rng.integers(0, len(bulk_small))
    labels = [-1, -1, -1, -1]
    labels[random_int] = 0

    move.labels = labels

    assert move(context)

    assert move.displaced_labels is not None
    assert_array_equal(move.displaced_labels, 0)

    trues = np.full(len(bulk_small), True)
    trues[random_int] = False

    assert_allclose(bulk_small.get_positions()[trues], old_positions[trues])


def test_displacement_move_3(bulk_small, rng):
    """Test the `DisplacementMove` class with a Box operation."""
    context = DisplacementContext(bulk_small, rng)

    move = DisplacementMove(np.arange(len(bulk_small)), Box(0.1))

    old_positions = bulk_small.get_positions()

    move.set_labels([0, -1, -1, -1])
    bulk_small.set_constraint(FixAtoms([0]))

    assert move(context)
    assert_allclose(bulk_small.positions, old_positions)

    move.apply_constraints = False

    assert move(context)
    new_positions = bulk_small.get_positions()
    assert not np.allclose(new_positions, old_positions)
    assert_allclose(new_positions[1:], old_positions[1:])

    assert_array_less(np.abs(bulk_small.positions[0] - old_positions[0]), 0.1)

    def check_move() -> bool:
        return move.apply_constraints

    move.check_move = check_move
    assert not move(context)

    assert_allclose(bulk_small.get_positions(), new_positions)

    move.apply_constraints = True
    move.set_labels([-1, 1, 2, -1])

    assert move(context)

    assert not np.allclose(bulk_small.get_positions(), new_positions)


def test_displacement_move_molecule(rng):
    """Test the `DisplacementMove` class with a molecule."""
    water = molecule("H2O", vacuum=10)

    context = DisplacementContext(water, rng)

    move = DisplacementMove([0, 0, 0], Sphere(0.1))

    old_positions = water.get_positions()
    distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    for _ in range(10):
        move(context)

    assert not np.allclose(water.positions, old_positions)

    old_positions = water.get_positions()
    new_distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    assert_allclose(distances, new_distances)

    move = DisplacementMove(move.labels, Sphere(0.1))

    move(context)

    new_distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    assert_allclose(distances, new_distances)
    assert not np.allclose(water.positions, old_positions)

    assert_allclose(np.linalg.norm(water.positions - old_positions, axis=-1), 0.1)

    water += molecule("H2O", vacuum=10)

    old_positions = water.get_positions()
    distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    move.labels = np.array([0, 0, 0, -1, -1, -1])  # type: ignore[shape]

    for _ in range(1000):
        move(context)

    new_distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)
    assert not np.allclose(water.positions, old_positions)
    assert_allclose(distances[3:, 3:], new_distances[3:, 3:])
    assert_allclose(distances[:3, :3], new_distances[:3, :3])
    assert_allclose(old_positions[3:], water.positions[3:])

    old_positions = water.get_positions()

    for _ in range(10):
        move(context)
        new_distances = np.linalg.norm(
            water.positions[:, None] - water.positions, axis=-1
        )
        assert_allclose(distances[3:, 3:], new_distances[3:, 3:])
        assert_allclose(distances[:3, :3], new_distances[:3, :3])

    assert not np.allclose(old_positions, water.positions)
