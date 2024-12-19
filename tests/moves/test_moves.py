from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.constraints import FixAtoms
from numpy.testing import assert_allclose, assert_array_equal

from quansino.mc.core import MonteCarloContext
from quansino.moves.atomic import AtomicMove
from quansino.moves.base import BaseMove
from quansino.moves.molecular import MolecularMove


def test_base_move(bulk_small, rng):
    context = MonteCarloContext(bulk_small, rng)

    def dummy_move():
        pass

    move = BaseMove(
        moving_indices=[0, 1],
        moving_per_step=1,
        move_type="dummy",
        move_functions={"dummy": dummy_move},
    )

    move.context = context

    assert move.moving_indices == [0, 1]
    assert move.moving_per_step == 1
    assert move.move_type == "dummy"
    assert move.move_functions == {"dummy": dummy_move}
    assert move.state.to_move is None

    move()

    assert move.state.to_move is None
    assert move.state.moved is not None

    move.update_indices([0, 1, 2])

    print(move.moving_indices)
    assert move.moving_indices == [0, 1, 2]


def test_simple_move(bulk_small, rng):
    context = MonteCarloContext(bulk_small, rng)
    move = AtomicMove(0.1, [1, 2])
    move.context = context

    assert move.move_type == "box"
    assert move.delta == 0.1
    assert move.moving_indices == [1, 2]
    assert move.moving_per_step == 1
    assert move.context is not None
    assert move.state is not None
    assert move.state.to_move is None

    move.state.to_move = [0]
    old_positions = bulk_small.get_positions()
    move()

    assert move.state.to_move is None
    assert_array_equal(np.array(move.state.moved), [0])

    new_positions = bulk_small.get_positions()

    assert_allclose(new_positions[0], old_positions[0], atol=0.1)
    assert_allclose(new_positions[1:], old_positions[1:])

    move.state.to_move = None
    move()
    displacement = bulk_small.get_positions() - old_positions

    assert np.all(displacement < 0.1)


def test_single_moves(rng):
    single_atom = Atoms("H", positions=[[0, 0, 0]])
    context = MonteCarloContext(single_atom, rng)
    move = AtomicMove(0.1, [0], moving_per_step=1, move_type="sphere")
    move.context = context
    move()

    assert np.isclose(np.linalg.norm(single_atom.get_positions()), 0.1)


def test_edge_cases(bulk_small, rng):
    context = MonteCarloContext(bulk_small, rng)
    move = AtomicMove(0.1, np.arange(len(bulk_small)), moving_per_step=0)
    move.context = context
    old_positions = bulk_small.get_positions()

    move()
    assert_allclose(bulk_small.get_positions(), old_positions)

    move.moving_indices = [0]
    move.moving_per_step = 1
    bulk_small.set_constraint(FixAtoms([0]))
    move()
    assert_allclose(bulk_small.get_positions(), old_positions)

    move.apply_constraints = False
    move()
    new_positions = bulk_small.get_positions()
    assert not np.allclose(new_positions, old_positions)

    def check_move() -> bool:
        return move.apply_constraints

    move.check_move = check_move
    move()

    assert_allclose(bulk_small.get_positions(), new_positions)

    move.apply_constraints = True
    move.moving_indices = [1, 2]

    move()

    assert not np.allclose(bulk_small.get_positions(), new_positions)


def test_ball_move(rng):
    single_atom = Atoms("H", positions=[[0, 0, 0]])
    context = MonteCarloContext(single_atom, rng)

    move = AtomicMove(0.1, [0], move_type="ball")
    move.context = context
    move()

    assert np.linalg.norm(single_atom.positions) < 0.1


def test_molecular_move(rng):
    water = molecule("H2O", vacuum=10)

    context = MonteCarloContext(water, rng)

    move = MolecularMove([0, 0, 0], 0.1)

    assert isinstance(move.molecule_ids, dict)
    assert len(move.molecule_ids) == 1
    assert_array_equal(move.moving_indices, [0, 1, 2])

    move.context = context

    old_positions = water.get_positions()
    distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    for _ in range(10):
        move()

    assert not np.allclose(water.positions, old_positions)

    old_positions = water.get_positions()
    new_distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    assert_allclose(distances, new_distances)

    move.move_type = "sphere"

    move()

    new_distances2 = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    assert_allclose(distances, new_distances2)
    assert not np.allclose(water.positions, old_positions)

    assert_allclose(np.linalg.norm(water.positions - old_positions, axis=-1), 0.1)

    water += molecule("H2O", vacuum=10)

    old_positions = water.get_positions()
    distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    move.molecule_ids = np.array([0, 0, 0, -1, -1, -1])

    assert len(move.molecule_ids) == 1
    assert move.molecule_ids.get(-1) is None
    assert_array_equal(move.molecule_ids[0], [0, 1, 2])

    for _ in range(1000):
        move()

    new_distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)
    assert not np.allclose(water.positions, old_positions)
    assert_allclose(distances[3:, 3:], new_distances[3:, 3:])
    assert_allclose(distances[:3, :3], new_distances[:3, :3])
    assert_allclose(old_positions[3:], water.positions[3:])
