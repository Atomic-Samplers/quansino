from __future__ import annotations

from copy import copy, deepcopy
from types import MethodType
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.constraints import FixAtoms
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from scipy.stats import chisquare

from quansino.mc.contexts import DisplacementContext, ExchangeContext
from quansino.mc.core import AcceptanceCriteria, MoveStorage
from quansino.moves.composite import CompositeDisplacementMove
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


def test_displacement_move(bulk_small, rng):
    move = DisplacementMove[Any, DisplacementContext](np.arange(len(bulk_small)))

    context = DisplacementContext(bulk_small, rng)

    def dummy_calculate(self: DisplacementMove, energy_difference) -> None:
        self.context.atoms.set_positions(np.zeros((len(self.context.atoms), 3)))

    move.calculate = MethodType(dummy_calculate, move)

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

    move.calculate(move.context)

    assert_allclose(bulk_small.get_positions(), np.zeros((len(bulk_small), 3)))

    move_copy = copy(move)

    assert move_copy.context is not None
    assert_array_equal(move_copy.labels, move.labels)
    assert move_copy.operation == move.operation
    assert move_copy.context == move.context


def test_sphere_move(bulk_small, rng):
    context = DisplacementContext(bulk_small, rng)

    move = DisplacementMove(np.arange(len(bulk_small)), Sphere(0.1))

    assert not hasattr(move, "context")
    assert move.operation.step_size == 0.1  # type: ignore

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


def test_translation_move(single_atom, rng):
    original_positions = rng.uniform(-50, 50, (1, 3))

    single_atom.positions = original_positions
    single_atom.center(vacuum=20)

    context = DisplacementContext(single_atom, rng)
    move = DisplacementMove(np.arange(len(single_atom)), Translation())
    move.attach_simulation(context)

    positions_recording = []

    for _ in range(10000):
        assert move()
        assert np.all(single_atom.get_scaled_positions() < 1)
        positions_recording.append(single_atom.get_positions())

    positions_recording = np.array(positions_recording).flatten()

    histogram = np.histogram(positions_recording, bins=10)[0]

    assert chisquare(histogram, f_exp=np.ones_like(histogram) * 3000)[1] > 0.001


def test_box_move(bulk_small, rng):
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


def test_ball_move(single_atom, rng):
    single_atom = Atoms("H", positions=[[0, 0, 0]])
    context = DisplacementContext(single_atom, rng)

    move = DisplacementMove([0], Ball(0.1))
    move.context = context

    assert move()
    assert move.displaced_labels == 0

    assert 0.1 > np.linalg.norm(single_atom.positions) > 0


def test_molecular_move(rng):
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


def test_translation_rotation_move(rng):
    atom = Atoms("H", positions=[[0, 0, 0]])

    context = DisplacementContext(atom, rng)
    move = DisplacementMove([0], Rotation())
    move.context = context

    assert move()

    assert_allclose(atom.positions, [[0, 0, 0]])

    move.operation = TranslationRotation()  # type: ignore

    assert move()

    assert_allclose(atom.positions, np.zeros((1, 3)))

    random_cell = rng.uniform(-10, 10, (3, 3))

    atom.set_cell(random_cell)

    for _ in range(50):
        assert move()
        assert np.all(atom.get_scaled_positions() < 1)


def test_exchange_move(empty_atoms, rng):
    initial_position = rng.uniform(-50, 50, (1, 3))

    exchange_atoms = Atoms("H", positions=initial_position)

    move = ExchangeMove[Operation, ExchangeContext](
        exchange_atoms=exchange_atoms, exchangeable_labels=[]
    )

    class DummyCriteria(AcceptanceCriteria):
        def evaluate(self, context, energy_difference):
            return energy_difference < 0

    move_storage = MoveStorage[DisplacementMove, ExchangeContext](
        move, 0, 0, 0, DummyCriteria()
    )

    context = ExchangeContext(empty_atoms, rng, moves={"default": move_storage})

    move.attach_simulation(context)

    move.bias_towards_insert = 1.0

    assert move()

    assert len(empty_atoms) == 1
    assert_allclose(empty_atoms.positions, np.zeros((1, 3)))

    old_positions = empty_atoms.get_positions()

    context.save_state()
    assert_array_equal(move.labels, [0])

    empty_atoms.set_cell(np.eye(3) * 100)

    old_atoms = empty_atoms.copy()

    move.check_move = lambda: False

    for _ in range(10):
        assert not move()
        assert empty_atoms == old_atoms

    move.check_move = lambda: True

    for _ in range(100):
        assert move()
        move.context.revert_state()
        assert empty_atoms == old_atoms

    assert len(empty_atoms) == 1
    assert len(move.labels) == len(empty_atoms)
    assert_allclose(empty_atoms.get_positions(), old_positions)

    for _ in range(100):
        assert move()
        move.context.save_state()

    assert move.context.added_indices is not None
    assert len(move.context.deleted_indices) == 0
    assert len(empty_atoms) == 101
    assert len(move.labels) == len(empty_atoms)
    assert_array_equal(move.labels, list(np.arange(101)))

    move.bias_towards_insert = 0.0

    old_atoms = empty_atoms.copy()

    move.to_delete_indices = 0
    assert move()

    assert len(move.context.added_indices) == 0
    assert len(move.context.deleted_indices) == 1
    assert move.context.deleted_indices == 0
    assert len(move.context.deleted_atoms) == 1

    old_deleted_atoms = deepcopy(move.context.deleted_atoms)

    move.context.deleted_atoms = Atoms()

    with pytest.raises(ValueError):
        move.context.revert_state()

    move.context.deleted_atoms = old_deleted_atoms

    move.context.revert_state()
    assert empty_atoms == old_atoms

    move.set_labels(np.full(101, -1))

    for _ in range(100):
        assert not move()
        assert empty_atoms == old_atoms

    move.set_labels(np.arange(101))

    for _ in range(100):
        assert move()
        move.context.revert_state()
        assert empty_atoms == old_atoms

    while len(move.labels) > 1:
        if rng.random() < 0.7:
            to_delete = rng.choice(move.unique_labels).astype(int)
            move.to_delete_indices = to_delete
            is_deletion = True
        else:
            move.to_add_atoms = Atoms("He", positions=[[0, 0, 0]], cell=np.eye(3) * 20)
            is_deletion = False

        assert move()
        move.context.save_state()

        if is_deletion:
            assert move.context.deleted_indices is not None
            assert move.to_delete_indices is None

            assert not np.isin(move.labels, to_delete).any()  # type: ignore
        else:
            assert_allclose(move.context.atoms.cell.array, np.eye(3) * 100)
            assert move.context.added_indices is not None
            assert move.to_add_atoms is None


def test_displacement_move_with_exchange_context(bulk_small, rng):
    move = DisplacementMove[Operation, DisplacementContext](
        np.arange(len(bulk_small)), Ball(0.1)
    )

    class DummyCriteria(AcceptanceCriteria[ExchangeContext]):
        def evaluate(self, context, energy_difference):
            return energy_difference < 0

    move_storage = MoveStorage[DisplacementMove, ExchangeContext](
        move, 0, 0, 0, DummyCriteria()
    )

    context = ExchangeContext(bulk_small, rng, moves={"default": move_storage})
    move.context = context

    old_atoms = bulk_small.copy()

    for _ in range(1000):
        assert move()
        move.context.revert_state()
        assert bulk_small == old_atoms


def test_molecular_exchange_move(rng):
    atoms = Atoms()

    move = ExchangeMove[Operation, ExchangeContext]("H2O", [])

    exchange_atoms = move.exchange_atoms

    old_distance = np.linalg.norm(
        exchange_atoms.positions[:, None] - exchange_atoms.positions, axis=-1
    )

    class DummyCriteria(AcceptanceCriteria):
        def evaluate(self, context, energy_difference):
            return energy_difference < 0

    move_storage = MoveStorage[
        DisplacementMove[Operation, ExchangeContext], ExchangeContext
    ](move, 0, 0, 0, DummyCriteria())

    context = ExchangeContext(atoms, rng, moves={"default": move_storage})

    move.attach_simulation(context)

    move.bias_towards_insert = 1.0

    move.to_delete_indices = -1

    assert not move()
    assert len(atoms) == 0

    move.to_add_atoms = exchange_atoms

    assert move()
    assert len(atoms) == 3
    move.context.save_state()

    new_distance = np.linalg.norm(atoms.positions[:, None] - atoms.positions, axis=-1)

    assert_allclose(new_distance, old_distance)
    assert_array_equal(move.labels, [0, 0, 0])
    assert len(move.unique_labels) == 1

    atoms.set_cell(np.eye(3) * 100)

    old_atoms = atoms.copy()

    for _ in range(1000):
        assert move()
        move.context.revert_state()
        assert atoms == old_atoms

    for _ in range(100):
        assert move()
        move.context.save_state()

    assert move.context.added_indices is not None
    assert len(move.context.deleted_atoms) == 0
    assert len(atoms) == 303

    move.bias_towards_insert = 0.0

    old_atoms = atoms.copy()

    for _ in range(100):
        assert move()
        move.context.revert_state()
        assert atoms == old_atoms

    assert len(atoms) == 303

    while len(move.unique_labels) > 0:
        to_delete = rng.choice(move.unique_labels)
        move.to_delete_indices = to_delete
        assert move()
        move.context.save_state()

        assert move.context.deleted_indices is not None
        assert move.to_delete_indices is None
        assert not np.isin(move.unique_labels, to_delete).any()


def test_displacement_calculator_consistency(bulk_small, rng):
    context = DisplacementContext(bulk_small, rng)

    move = DisplacementMove[Operation, DisplacementContext](
        np.arange(len(bulk_small)), Sphere(0.1)
    )
    move.attach_simulation(context)

    for _ in range(100):
        assert move()
        bulk_small.get_potential_energy()
        move.context.save_state()
        assert context.last_results == bulk_small.calc.results
        assert move()
        move.context.revert_state()
        assert len(bulk_small.calc.check_state(bulk_small)) == 0
        assert (
            bulk_small.calc.get_property("energy", bulk_small, allow_calculation=False)
            is not None
        )


def test_exchange_calculator_consistency(empty_atoms, rng):
    exchange_atoms = Atoms("H", positions=[[0, 0, 0]])

    empty_atoms.set_cell(np.eye(3) * 100)

    for _ in range(100):
        empty_atoms.extend(exchange_atoms)

    empty_atoms.set_positions(rng.uniform(-50, 50, (100, 3)))

    move = ExchangeMove[Operation, ExchangeContext](
        exchange_atoms, np.arange(100), Translation()
    )

    class DummyCriteria(AcceptanceCriteria):
        def evaluate(self, context, energy_difference):
            return energy_difference < 0

    move_storage = MoveStorage[DisplacementMove, ExchangeContext](
        move, 0, 0, 0, DummyCriteria()
    )

    context = ExchangeContext(empty_atoms, rng, moves={"default": move_storage})

    move.attach_simulation(context)

    for _ in range(100):
        assert move()
        empty_atoms.get_potential_energy()
        move.context.save_state()
        assert context.last_results == empty_atoms.calc.results
        assert move()
        move.context.revert_state()
        assert len(empty_atoms.calc.check_state(empty_atoms)) == 0
        assert (
            empty_atoms.calc.get_property(
                "energy", empty_atoms, allow_calculation=False
            )
            is not None
        )


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


def test_operations(single_atom, rng):
    context = DisplacementContext(single_atom, rng)

    sphere = Sphere(0.1)
    assert sphere.calculate(context).shape == (1, 3)
    assert_allclose(np.linalg.norm(sphere.calculate(context)), 0.1)

    box = Box(0.1)
    assert box.calculate(context).shape == (1, 3)
    assert (box.calculate(context) > -0.1).all()
    assert (box.calculate(context) < 0.1).all()

    ball = Ball(0.1)
    assert ball.calculate(context).shape == (1, 3)
    assert (np.linalg.norm(ball.calculate(context)) < 0.1).all()
    assert (np.linalg.norm(ball.calculate(context)) > 0).all()

    single_atom.set_cell(np.eye(3) * 100)
    context.moving_indices = [0]

    translation = Translation()
    assert translation.calculate(context).shape == (1, 3)
    assert (translation.calculate(context) < 100).all()
    assert (translation.calculate(context) > 0).all()

    old_positions = single_atom.get_positions()

    rotation = Rotation()
    assert rotation.calculate(context).shape == (1, 3)
    single_atom.positions += rotation.calculate(context)
    assert_allclose(context.atoms.positions, old_positions)

    context.atoms = molecule("H2O", vacuum=10)

    old_distances = np.linalg.norm(
        context.atoms.positions[:, None] - context.atoms.positions, axis=-1
    )

    context.moving_indices = [0, 1, 2]

    old_positions = context.atoms.get_positions()
    assert not np.allclose(rotation.calculate(context), 0)
    context.atoms.positions += rotation.calculate(context)
    assert not np.allclose(context.atoms.get_positions(), old_positions)

    new_distances = np.linalg.norm(
        context.atoms.positions[:, None] - context.atoms.positions, axis=-1
    )

    assert_allclose(old_distances, new_distances)

    translation_rotation = TranslationRotation()
    assert translation_rotation.calculate(context).shape == (3, 3)
    assert not np.allclose(translation_rotation.calculate(context), 0)

    new_distances_2 = np.linalg.norm(
        context.atoms.positions[:, None] - context.atoms.positions, axis=-1
    )

    assert_allclose(old_distances, new_distances_2)

    composite_operation = sphere + box

    assert len(composite_operation) == 2

    assert composite_operation[0] == sphere
    assert composite_operation[1] == box

    assert composite_operation.calculate(context).shape == (1, 3)
    assert (np.linalg.norm(composite_operation.calculate(context)) > 0.0).all()

    composite_operation_2 = composite_operation + ball
    composite_operation_3 = ball + composite_operation

    assert len(composite_operation_2) == 3
    assert len(composite_operation_3) == 3

    assert composite_operation_2[0] == sphere
    assert composite_operation_2[1] == box
    assert composite_operation_2[2] == ball

    assert composite_operation_3[0] == ball
    assert composite_operation_3[1] == sphere
    assert composite_operation_3[2] == box

    composite_operation_4 = composite_operation + composite_operation_2

    assert len(composite_operation_4) == 5
    assert composite_operation_4[0] == sphere
    assert composite_operation_4[1] == box
    assert composite_operation_4[2] == sphere
    assert composite_operation_4[3] == box
    assert composite_operation_4[4] == ball

    composite_operation_5 = sphere * 5

    assert len(composite_operation_5) == 5

    for move in composite_operation_5:
        assert move == sphere

    assert composite_operation_5.calculate(context).shape == (1, 3)
    assert (np.linalg.norm(composite_operation_5.calculate(context)) > 0.0).all()

    with pytest.raises(ValueError):
        sphere * -2  # type: ignore

    with pytest.raises(ValueError):
        sphere * 4.2  # type: ignore

    with pytest.raises(ValueError):
        composite_operation_2 * 0  # type: ignore

    with pytest.raises(ValueError):
        composite_operation_2 * -1  # type: ignore

    with pytest.raises(ValueError):
        composite_operation_2 * 4.2  # type: ignore

    composite_operation_6 = composite_operation_2 * 2

    assert len(composite_operation_6) == 6

    for move in composite_operation_6:
        assert isinstance(move, Operation)
