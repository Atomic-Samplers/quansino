from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.constraints import FixAtoms
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chisquare

from quansino.mc.contexts import DisplacementContext, ExchangeContext
from quansino.moves.core import BaseMove
from quansino.moves.displacements import DisplacementMove
from quansino.moves.exchange import ExchangeMove
from quansino.moves.operations import (
    Ball,
    Box,
    Operation,
    Rotation,
    Sphere,
    Translation,
    TranslationRotation,
)

if TYPE_CHECKING:
    from numpy.random import Generator

    from quansino.typing import Displacement


def test_base_move(bulk_small, rng):
    context = DisplacementContext(bulk_small, rng)

    class DummyOperation(Operation):
        def calculate(self, context):
            context.atoms.set_positions(np.zeros((len(context.atoms), 3)))

    move = BaseMove(DummyOperation(), apply_constraints=True)

    move.attach_simulation(context)

    assert move.context is not None
    move.context = cast(DisplacementContext, move.context)

    assert move.context.atoms is not None
    assert move.context.rng is not None

    move.context.selected_candidates = [0, 1, 2]
    move.context.moving_indices = [0, 1]

    move.context.register_success()

    assert move.context.selected_candidates is None

    assert move.context.moving_indices is None
    assert move.context.moved_candidates is not None
    assert_array_equal(move.context.moved_candidates, [0, 1, 2])

    move.context.register_failure()

    assert move.context.selected_candidates is None

    @dataclass
    class DummyContext:
        atoms: Atoms
        rng: Generator

    move.attach_simulation(DummyContext(bulk_small, rng))  # type: ignore

    assert move.context is not None

    move.move_operator.calculate(move.context)

    assert_allclose(bulk_small.get_positions(), np.zeros((len(bulk_small), 3)))


def test_simple_move(bulk_small, rng):
    context = DisplacementContext(bulk_small, rng)

    ball_operation = Sphere(0.1)
    move = DisplacementMove(ball_operation)

    with pytest.raises(AttributeError):
        assert move.context is None

    assert move.move_operator == ball_operation
    assert_allclose(ball_operation.step_size, 0.1)

    move.attach_simulation(context)

    assert move.context is not None
    assert_array_equal(move.candidate_indices, np.arange(len(bulk_small)))

    old_positions = bulk_small.get_positions()

    assert move()

    norm = np.linalg.norm(bulk_small.get_positions() - old_positions)

    assert_allclose(norm, 0.1)

    old_positions = bulk_small.get_positions()

    move._number_of_available_candidates = 0

    assert not move()

    assert_allclose(bulk_small.get_positions(), old_positions)

    move.set_candidate_indices([0, 0, 0, 0])

    assert move()

    assert len(np.unique(np.round(bulk_small.get_positions() - old_positions, 5))) == 3
    assert move._number_of_available_candidates == 1

    move.displacements_per_move = 4

    old_positions = bulk_small.get_positions()

    assert move()

    assert len(np.unique(np.round(bulk_small.get_positions() - old_positions, 5))) == 3

    old_positions = bulk_small.get_positions()
    move.set_candidate_indices([-1, -1, -1, -1])

    assert not move()

    assert_allclose(bulk_small.get_positions(), old_positions)

    random_int = rng.integers(0, len(bulk_small))
    candidate_indices = [-1, -1, -1, -1]
    candidate_indices[random_int] = 0

    move.set_candidate_indices(candidate_indices)

    assert move()

    assert move.context.moved_candidates is not None
    assert_array_equal(move.context.moved_candidates, [0])

    trues = np.full(len(bulk_small), True)
    trues[random_int] = False

    assert_allclose(bulk_small.get_positions()[trues], old_positions[trues])


def test_translation_operation(rng):
    original_positions = rng.uniform(-50, 50, (1, 3))

    single_atom = Atoms("H", positions=original_positions)
    single_atom.center(vacuum=20)
    context = DisplacementContext(single_atom, rng)
    move = DisplacementMove(Translation())
    move.attach_simulation(context)

    positions_recording = []

    for _ in range(10000):
        assert move()
        assert np.all(single_atom.get_scaled_positions() < 1)
        positions_recording.append(single_atom.get_positions())

    positions_recording = np.array(positions_recording).flatten()

    histogram = np.histogram(positions_recording, bins=10)[0]

    assert chisquare(histogram, f_exp=np.ones_like(histogram) * 3000)[1] > 0.02


def test_edge_cases(bulk_small, rng):
    context = DisplacementContext(bulk_small, rng)
    move = DisplacementMove(
        Box(0.1), np.arange(len(bulk_small)), displacements_per_move=0
    )
    move.attach_simulation(context)
    old_positions = bulk_small.get_positions()

    assert not move()
    assert_allclose(bulk_small.get_positions(), old_positions)

    move.displacements_per_move = 1

    move.set_candidate_indices([0, -1, -1, -1])
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
    move.set_candidate_indices([-1, 1, 2, -1])

    assert move()

    assert not np.allclose(bulk_small.get_positions(), new_positions)


def test_ball_move(rng):
    single_atom = Atoms("H", positions=[[0, 0, 0]])
    context = DisplacementContext(single_atom, rng)

    move = DisplacementMove(Ball(0.1), [0])
    move.context = context

    assert move()
    assert move.context.moved_candidates == [0]

    assert np.linalg.norm(single_atom.positions) > 0
    assert np.linalg.norm(single_atom.positions) < 0.1


def test_molecular_move(rng):
    water = molecule("H2O", vacuum=10)

    context = DisplacementContext(water, rng)

    move = DisplacementMove(Rotation(), [0, 0, 0])

    assert isinstance(move.move_operator, Rotation)
    assert move.move_operator.center == "COM"

    move.context = context

    old_positions = water.get_positions()
    distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    for _ in range(10):
        move()

    assert not np.allclose(water.positions, old_positions)

    old_positions = water.get_positions()
    new_distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    assert_allclose(distances, new_distances)

    move.move_operator = Sphere(0.1)

    move()

    new_distances2 = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    assert_allclose(distances, new_distances2)
    assert not np.allclose(water.positions, old_positions)

    assert_allclose(np.linalg.norm(water.positions - old_positions, axis=-1), 0.1)

    water += molecule("H2O", vacuum=10)

    old_positions = water.get_positions()
    distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)

    move.candidate_indices = np.array([0, 0, 0, -1, -1, -1])

    for _ in range(1000):
        move()

    new_distances = np.linalg.norm(water.positions[:, None] - water.positions, axis=-1)
    assert not np.allclose(water.positions, old_positions)
    assert_allclose(distances[3:, 3:], new_distances[3:, 3:])
    assert_allclose(distances[:3, :3], new_distances[:3, :3])
    assert_allclose(old_positions[3:], water.positions[3:])

    move.set_candidate_indices([0, 0, 0, 1, 1, 1])

    move.displacements_per_move = 2

    old_positions = water.get_positions()

    for _ in range(10):
        move()

    assert not np.allclose(old_positions[:3], water.positions[:3])
    assert not np.allclose(old_positions[3:], water.positions[3:])


def test_molecular_on_atoms(rng):
    atom = Atoms("H", positions=[[0, 0, 0]])

    context = DisplacementContext(atom, rng)
    move = DisplacementMove(Rotation(), [0])
    move.context = context

    assert move()

    assert_allclose(atom.positions, [[0, 0, 0]])

    move.move_operator = TranslationRotation()

    assert move()

    assert_allclose(atom.positions, np.zeros((1, 3)))

    random_cell = rng.uniform(-10, 10, (3, 3))

    atom.set_cell(random_cell)

    for _ in range(50):
        assert move()
        assert np.all(atom.get_scaled_positions() < 1)


def test_exchange_move(rng):
    atoms = Atoms()

    initial_position = rng.uniform(-50, 50, (1, 3))

    exchange_atoms = Atoms("H", positions=initial_position)

    move = ExchangeMove(exchange_atoms=exchange_atoms)

    context = ExchangeContext(atoms, rng, moves={"default": move})

    move.attach_simulation(context)

    move.bias_towards_insert = 1.0

    assert move()

    assert len(atoms) == 1
    assert_allclose(atoms.get_positions(), np.zeros((1, 3)))

    move.update_moves()
    assert_array_equal(move.candidate_indices, [0])

    atoms.set_cell(np.eye(3) * 100)

    for _ in range(1000):
        assert move()
        move.context.revert_move()

    assert len(atoms) == 1
    assert_allclose(atoms.get_positions(), np.zeros((1, 3)))

    for _ in range(100):
        assert move()
        move.update_moves()

    assert move.context.last_added_indices is not None
    assert move.context.last_deleted_indices is None
    assert len(atoms) == 101
    assert_array_equal(move.candidate_indices, np.arange(101))

    move.bias_towards_insert = 0.0

    old_atoms = atoms.copy()

    for _ in range(100):
        assert move()
        move.context.revert_move()

    assert len(atoms) == 101
    assert atoms == old_atoms

    while len(move.candidate_indices) > 0:
        size = (
            rng.integers(1, 5)
            if len(move._unique_candidates) > 5
            else len(move._unique_candidates)
        )
        move.context.deletion_candidates = rng.choice(
            move._unique_candidates, size=size
        )
        assert move()
        move.update_moves()

        assert move.context.last_deleted_indices is not None
        assert move.context.deletion_candidates is not None

        assert not np.isin(
            move.candidate_indices, move.context.deletion_candidates
        ).any()


def test_molecular_exchange_move(rng):
    atoms = Atoms()

    exchange_atoms = molecule("H2O", vacuum=10)

    old_distance = np.linalg.norm(
        exchange_atoms.positions[:, None] - exchange_atoms.positions, axis=-1
    )

    move = ExchangeMove(exchange_atoms=exchange_atoms)

    context = ExchangeContext(atoms, rng, moves={"default": move})

    move.attach_simulation(context)

    move.bias_towards_insert = 1.0

    move.context.selected_candidates = []
    assert move()
    move.update_moves()

    assert len(atoms) == 3

    new_distance = np.linalg.norm(atoms.positions[:, None] - atoms.positions, axis=-1)

    assert_allclose(new_distance, old_distance)
    assert_array_equal(move.candidate_indices, [0, 0, 0])

    atoms.set_cell(np.eye(3) * 100)

    old_positions = atoms.get_positions()

    for _ in range(1000):
        assert move()
        move.context.revert_move()

    assert len(atoms) == 3
    assert_allclose(atoms.get_positions(), old_positions)

    for _ in range(100):
        assert move()
        move.update_moves()

    assert move.context.last_added_indices is not None
    assert move.context.last_deleted_atoms is None
    assert len(atoms) == 303

    move.bias_towards_insert = 0.0

    old_atoms = atoms.copy()

    for _ in range(100):
        assert move()
        move.context.revert_move()

    assert len(atoms) == 303
    assert atoms == old_atoms

    while len(move.candidate_indices) > 0:
        size = (
            rng.integers(1, 5)
            if len(move._unique_candidates) > 5
            else len(move._unique_candidates)
        )
        move.context.deletion_candidates = rng.choice(
            move._unique_candidates, size=size
        )
        assert move()
        move.update_moves()

        assert move.context.last_deleted_indices is not None
        assert move.context.deletion_candidates is not None

        assert not np.isin(
            move.candidate_indices, move.context.deletion_candidates
        ).any()


def test_addition_operations(bulk_small, rng):
    class DummyTranslation(Operation):
        def calculate(self, context: DisplacementContext) -> Displacement:
            return np.ones((1, 3))

    double_displacement = DummyTranslation() + DummyTranslation()

    move = DisplacementMove(double_displacement, displacements_per_move=4)
    context = DisplacementContext(bulk_small, rng)
    move.attach_simulation(context)

    old_positions = bulk_small.get_positions()

    assert move()
    assert_allclose(bulk_small.get_positions(), old_positions + 2)
