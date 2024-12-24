from __future__ import annotations

import numpy as np
import pytest
from ase import Atom, Atoms
from ase.build import molecule
from ase.constraints import FixAtoms
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chisquare

from quansino.mc.core import MonteCarloContext, MoveStore
from quansino.moves.atomic import AtomicMove
from quansino.moves.base import BaseMove
from quansino.moves.exchange import (
    AtomicExchangeMove,
    ExchangeContext,
    MolecularExchangeMove,
)
from quansino.moves.molecular import MolecularMove


def test_base_move(bulk_small, rng):
    context = MonteCarloContext(bulk_small, rng)

    def dummy_move():
        move.apply_constraints = False

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
    assert move.state.moved is None
    assert not move.apply_constraints

    move.update_indices([0, 1, 2])
    assert_array_equal(move.moving_indices, [0, 1, 0, 1, 2])

    move.update_indices(None, 1)
    assert_array_equal(move.moving_indices, [0, 0, 1, 2])

    move.update_indices(None, [0, 1])
    assert_array_equal(move.moving_indices, [1, 2])

    with pytest.raises(ValueError):
        move.update_indices(None, None)

    with pytest.raises(ValueError):
        move.update_indices([1, 2], [1, 2])

    move.check_move = lambda: False

    assert not move()

    move.check_move = lambda: True

    move.max_attempts = 0

    assert not move()

    class DummyContext:
        _atoms = bulk_small
        _rng = rng

    with pytest.raises(AttributeError):
        move.context = DummyContext()  # type: ignore


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
    original_positions = rng.uniform(-50, 50, (1, 3))
    single_atom = Atoms("H", positions=original_positions)
    context = MonteCarloContext(single_atom, rng)
    move = AtomicMove(0.1, [0], moving_per_step=1, move_type="sphere")
    move.context = context
    move()

    assert np.isclose(
        np.linalg.norm(single_atom.get_positions() - original_positions), 0.1
    )

    single_atom.center(vacuum=20)

    move.move_type = "translation"

    positions_recording = []

    for _ in range(10000):
        assert move()
        assert np.all(single_atom.get_scaled_positions() < 1)
        positions_recording.append(single_atom.get_positions())

    positions_recording = np.array(positions_recording).flatten()

    histogram = np.histogram(positions_recording, bins=10)[0]

    assert chisquare(histogram, f_exp=np.ones_like(histogram) * 3000)[1] > 0.05


def test_edge_cases(bulk_small, rng):
    context = MonteCarloContext(bulk_small, rng)
    move = AtomicMove(0.1, np.arange(len(bulk_small)), moving_per_step=0)
    move.context = context
    old_positions = bulk_small.get_positions()

    assert move()
    assert_allclose(bulk_small.get_positions(), old_positions)

    move.moving_indices = [0]
    move.moving_per_step = 1
    bulk_small.set_constraint(FixAtoms([0]))
    assert move()
    assert_allclose(bulk_small.get_positions(), old_positions)

    move.apply_constraints = False
    assert move()
    new_positions = bulk_small.get_positions()
    assert not np.allclose(new_positions, old_positions)

    def check_move() -> bool:
        return move.apply_constraints

    move.check_move = check_move
    assert not move()

    assert_allclose(bulk_small.get_positions(), new_positions)

    move.apply_constraints = True
    move.moving_indices = [1, 2]

    assert move()

    assert not np.allclose(bulk_small.get_positions(), new_positions)


def test_ball_move(rng):
    single_atom = Atoms("H", positions=[[0, 0, 0]])
    context = MonteCarloContext(single_atom, rng)

    move = AtomicMove(0.1, [0], move_type="ball")
    move.context = context
    assert move()
    assert move.state.moved == 0

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

    move.update_indices([3, 4, 5])


def test_molecular_on_atoms(rng):
    atom = Atoms("H", positions=[[0, 0, 0]])

    context = MonteCarloContext(atom, rng)
    move = MolecularMove([0], 0.1)
    move.context = context

    assert move()

    assert_allclose(atom.positions, [[0, 0, 0]])

    move.move_type = "sphere"

    assert move()
    assert_allclose(np.linalg.norm(atom.positions), 0.1)
    assert move.state.moved == 0

    move.move_type = "translation_rotation"

    assert move()

    assert_allclose(atom.positions, np.zeros((1, 3)))

    random_cell = rng.uniform(-10, 10, (3, 3))

    atom.set_cell(random_cell)

    for _ in range(50):
        assert move()
        assert np.all(atom.get_scaled_positions() < 1)


def test_atomic_exchange_move(rng):
    atoms = Atoms()

    initial_position = rng.uniform(-50, 50, 3)

    exchange_atom = Atom("H", position=initial_position)

    move = AtomicExchangeMove(exchange_atom=exchange_atom)

    move_store = MoveStore(move, 1, 1.0, 0)

    context = ExchangeContext(atoms, rng, {"default": move_store})

    move.context = context

    move.bias_towards_insert = 1.0

    assert move()

    assert len(atoms) == 1
    assert_allclose(atoms.get_positions(), np.zeros((1, 3)))

    atoms.set_cell(np.eye(3) * 100)

    for _ in range(1000):
        assert move()
        move.revert_move()

    assert len(atoms) == 1
    assert_allclose(atoms.get_positions(), np.zeros((1, 3)))

    for _ in range(100):
        assert move()
        move.update_moves()

    assert move.exchange_state.last_added is not None
    assert move.exchange_state.last_deleted is None
    assert isinstance(move.moving_indices, np.ndarray)
    assert len(move.moving_indices) == 100
    assert len(atoms) == 101

    move.bias_towards_insert = 0.0

    old_atoms = atoms.copy()

    for _ in range(100):
        assert move()
        move.revert_move()

    assert len(atoms) == 101

    assert atoms == old_atoms

    while len(move.moving_indices) > 0:
        size = (
            rng.integers(1, 5)
            if len(move.moving_indices) > 5
            else len(move.moving_indices)
        )
        move.state.to_move = rng.choice(move.moving_indices, size=size)
        assert move()
        move.update_moves()
        assert len(set(move.moving_indices)) == len(move.moving_indices)
