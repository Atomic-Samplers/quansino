from __future__ import annotations

import math
from copy import copy, deepcopy

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.units import _e, _hplanck, _Nav, kB
from numpy.testing import assert_allclose, assert_array_equal
from tests.conftest import DummyCalculator, DummyCriteria

from quansino.mc.contexts import ExchangeContext
from quansino.mc.core import MoveStorage
from quansino.mc.gcmc import GrandCanonical, GrandCanonicalCriteria
from quansino.moves.displacement import DisplacementMove
from quansino.moves.exchange import ExchangeMove
from quansino.operations.displacement import Translation


def test_grand_canonical(bulk_large):
    bulk_large.calc = EMT()

    energy_full = bulk_large.get_potential_energy()

    del bulk_large[0]

    energy_minus_one = bulk_large.get_potential_energy()

    energy_difference = energy_full - energy_minus_one

    exchange_move = ExchangeMove(Atoms("Cu"), np.arange(len(bulk_large)), Translation())

    gcmc = GrandCanonical(
        bulk_large,
        temperature=1000,
        chemical_potential=energy_difference,
        max_cycles=1,
        number_of_exchange_particles=len(bulk_large),
    )

    gcmc.add_move(exchange_move)

    assert gcmc.context.temperature == 1000
    assert gcmc.context.chemical_potential == energy_difference
    assert gcmc.max_cycles == 1
    assert gcmc.atoms == bulk_large

    gcmc.chemical_potential = 0.0

    assert gcmc.chemical_potential == 0.0
    assert gcmc.context.chemical_potential == 0.0

    gcmc.temperature = 2000

    assert gcmc.temperature == 2000
    assert gcmc.context.temperature == 2000

    gcmc.chemical_potential = energy_difference * 1.56

    del bulk_large[20]
    del bulk_large[10]
    del bulk_large[5]
    del bulk_large[2]

    gcmc.number_of_exchange_particles = len(bulk_large)

    assert gcmc.context.number_of_exchange_particles == len(bulk_large)
    assert gcmc.number_of_exchange_particles == len(bulk_large)

    exchange_move.set_labels(np.arange(len(bulk_large)))

    current_atoms_count = len(bulk_large)
    labels = np.array(gcmc.moves["default"].move.labels)

    for _ in gcmc.irun(10):
        assert gcmc.temperature == 2000
        assert gcmc.context.temperature == 2000
        assert gcmc.chemical_potential == energy_difference * 1.56
        assert gcmc.context.chemical_potential == energy_difference * 1.56

        if len(gcmc.atoms) != current_atoms_count:
            assert bool(gcmc.context.added_atoms) ^ bool(gcmc.context.deleted_atoms)
            assert len(bulk_large) == gcmc.context.number_of_exchange_particles
            assert len(bulk_large) == current_atoms_count + gcmc.context.particle_delta

            assert gcmc.context.accessible_volume == bulk_large.cell.volume

            if gcmc.context.added_atoms:
                assert gcmc.context.particle_delta == len(gcmc.context.added_atoms)
                assert len(gcmc.context.added_indices) == len(gcmc.context.added_atoms)

                assert len(labels) + len(gcmc.context.added_indices) == len(
                    gcmc.moves["default"].move.labels
                )
                assert max(labels) + 1 in gcmc.moves["default"].move.labels
            elif gcmc.context.deleted_atoms:
                assert gcmc.context.particle_delta == -len(gcmc.context.deleted_atoms)
                assert len(gcmc.context.deleted_indices) == len(
                    gcmc.context.deleted_atoms
                )

                assert (
                    labels[gcmc.context.deleted_indices]
                    not in gcmc.moves["default"].move.labels
                )

            current_atoms_count = len(gcmc.atoms)
            labels = np.array(gcmc.moves["default"].move.labels)
        else:
            assert_array_equal(labels, gcmc.moves["default"].move.labels)
            assert current_atoms_count == len(gcmc.atoms)


def test_grand_canonical_simulation(bulk_medium):
    energy_full = bulk_medium.get_potential_energy()

    del bulk_medium[0]

    energy_minus_one = bulk_medium.get_potential_energy()

    del bulk_medium[20]
    del bulk_medium[10]
    del bulk_medium[5]
    del bulk_medium[2]

    energy_difference = energy_full - energy_minus_one

    move_storage = MoveStorage(
        DisplacementMove(np.arange(len(bulk_medium))),
        10,
        0.5,
        1,
        GrandCanonicalCriteria(),
    )

    gcmc = GrandCanonical(
        bulk_medium,
        temperature=1000,
        chemical_potential=energy_difference,
        max_cycles=1,
        default_exchange_move=move_storage,
        number_of_exchange_particles=len(bulk_medium),
    )

    assert gcmc.moves["default_exchange_move"].move == move_storage.move
    assert gcmc.moves["default_exchange_move"].interval == 10
    assert gcmc.moves["default_exchange_move"].probability == 0.5
    assert gcmc.moves["default_exchange_move"].minimum_count == 1

    displacement_move = DisplacementMove(np.arange(len(bulk_medium)))

    labels = np.arange(len(bulk_medium))
    labels[: len(labels) // 2] = -1
    exchange_move = ExchangeMove(Atoms("Cu"), labels)

    gcmc = GrandCanonical(
        bulk_medium,
        temperature=1000,
        chemical_potential=energy_difference,
        max_cycles=1,
        default_exchange_move=exchange_move,
        default_displacement_move=displacement_move,
        number_of_exchange_particles=len(bulk_medium),
    )

    for index in range(10):
        gcmc.add_move(copy(displacement_move), name=f"displacement_{index}")

    gcmc.moves["displacement_0"].move.set_labels(labels)
    gcmc.moves["displacement_0"].move.default_label = -1

    assert gcmc.context.temperature == 1000
    assert gcmc.context.chemical_potential == energy_difference
    assert gcmc.max_cycles == 1
    assert gcmc.atoms == bulk_medium

    gcmc.chemical_potential = 0.0

    assert gcmc.chemical_potential == 0.0
    assert gcmc.context.chemical_potential == 0.0

    gcmc.temperature = 2000

    assert gcmc.temperature == 2000
    assert gcmc.context.temperature == 2000

    gcmc.chemical_potential = energy_difference * 1.56

    assert gcmc.context.number_of_exchange_particles == len(bulk_medium)
    assert gcmc.number_of_exchange_particles == len(bulk_medium)

    current_atoms_count = len(bulk_medium)
    labels = np.array(gcmc.moves["default_exchange_move"].move.labels)

    old_atoms = bulk_medium.copy()
    old_energy = bulk_medium.get_potential_energy()

    for _ in gcmc.irun(10):
        assert gcmc.temperature == 2000
        assert gcmc.context.temperature == 2000
        assert gcmc.chemical_potential == energy_difference * 1.56
        assert gcmc.context.chemical_potential == energy_difference * 1.56

        if len(gcmc.atoms) != current_atoms_count:
            assert bool(gcmc.context.added_atoms) ^ bool(gcmc.context.deleted_atoms)
            assert len(bulk_medium) == gcmc.context.number_of_exchange_particles
            assert len(bulk_medium) == current_atoms_count + gcmc.context.particle_delta

            assert gcmc.context.accessible_volume == bulk_medium.cell.volume

            if gcmc.context.added_atoms:
                assert gcmc.context.particle_delta == len(gcmc.context.added_atoms)
                assert len(gcmc.context.added_indices) == len(gcmc.context.added_atoms)

                assert len(labels) + len(gcmc.context.added_indices) == len(
                    gcmc.moves["default_exchange_move"].move.labels
                )
                assert (
                    max(labels) + 1 in gcmc.moves["default_exchange_move"].move.labels
                )
            elif gcmc.context.deleted_atoms:
                assert gcmc.context.particle_delta == -len(gcmc.context.deleted_atoms)
                assert len(gcmc.context.deleted_indices) == len(
                    gcmc.context.deleted_atoms
                )

                assert (
                    labels[gcmc.context.deleted_indices]
                    not in gcmc.moves["default_exchange_move"].move.labels
                )

            current_atoms_count = len(gcmc.atoms)
            labels = np.array(gcmc.moves["default_exchange_move"].move.labels)
        else:
            assert_array_equal(labels, gcmc.moves["default_exchange_move"].move.labels)
            assert current_atoms_count == len(gcmc.atoms)

        assert gcmc.moves["displacement_0"].move.default_label == -1

        if gcmc.acceptance_rate:
            assert bulk_medium != old_atoms
            if "energy" in gcmc.last_results:
                assert not np.allclose(
                    gcmc.last_results["energy"], old_energy, atol=1e-8, rtol=0
                )
        else:
            assert bulk_medium == old_atoms
            if "energy" in gcmc.last_results:
                assert_allclose(gcmc.last_results["energy"], old_energy)

        old_atoms = bulk_medium.copy()
        old_energy = bulk_medium.get_potential_energy()


def test_grand_canonical_atomic_simulation(empty_atoms, rng):
    initial_position = rng.uniform(-50, 50, (1, 3))
    exchange_atoms = Atoms("H", positions=initial_position)

    move = ExchangeMove(exchange_atoms=exchange_atoms, exchangeable_labels=[])
    move_storage = MoveStorage(move, 0, 0, 0, DummyCriteria())

    gcmc = GrandCanonical(
        empty_atoms,
        temperature=1000,
        chemical_potential=0.0,
        max_cycles=1,
        number_of_exchange_particles=len(empty_atoms),
        default_exchange_move=move_storage,
    )

    move.bias_towards_insert = 1.0

    assert move()

    assert len(empty_atoms) == 1
    assert_allclose(empty_atoms.positions, np.zeros((1, 3)))

    old_positions = empty_atoms.get_positions()

    gcmc.save_state()
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
        gcmc.revert_state()
        assert empty_atoms == old_atoms

    assert len(empty_atoms) == 1
    assert len(move.labels) == len(empty_atoms)
    assert_allclose(empty_atoms.get_positions(), old_positions)

    for _ in range(100):
        assert move()
        gcmc.save_state()

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
        gcmc.revert_state()

    move.context.deleted_atoms = old_deleted_atoms

    gcmc.revert_state()
    assert empty_atoms == old_atoms

    move.set_labels(np.full(101, -1))

    for _ in range(100):
        assert not move()
        assert empty_atoms == old_atoms

    move.set_labels(np.arange(101))

    for _ in range(100):
        assert move()
        gcmc.revert_state()
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
        gcmc.save_state()

        if is_deletion:
            assert move.context.deleted_indices is not None
            assert move.to_delete_indices is None

            assert not np.isin(move.labels, to_delete).any()  # type: ignore
        else:
            assert_allclose(move.context.atoms.cell.array, np.eye(3) * 100)
            assert move.context.added_indices is not None
            assert move.to_add_atoms is None


def test_grand_canonical_molecular_simulation(empty_atoms, rng):
    move = ExchangeMove("H2O", [])
    exchange_atoms = move.exchange_atoms

    old_distance = np.linalg.norm(
        exchange_atoms.positions[:, None] - exchange_atoms.positions, axis=-1
    )

    move_storage = MoveStorage(move, 0, 0, 0, DummyCriteria())

    gcmc = GrandCanonical(
        empty_atoms,
        temperature=1000,
        chemical_potential=0.0,
        max_cycles=1,
        number_of_exchange_particles=len(empty_atoms),
        default_exchange_move=move_storage,
    )

    move.bias_towards_insert = 1.0

    move.to_delete_indices = -1

    assert not move()
    assert len(empty_atoms) == 0

    move.to_add_atoms = exchange_atoms

    assert move()
    assert len(empty_atoms) == 3
    gcmc.save_state()

    new_distance = np.linalg.norm(
        empty_atoms.positions[:, None] - empty_atoms.positions, axis=-1
    )

    assert_allclose(new_distance, old_distance)
    assert_array_equal(move.labels, [0, 0, 0])
    assert len(move.unique_labels) == 1

    empty_atoms.set_cell(np.eye(3) * 100)

    old_atoms = empty_atoms.copy()

    for _ in range(1000):
        assert move()
        gcmc.revert_state()
        assert empty_atoms == old_atoms

    for _ in range(100):
        assert move()
        gcmc.save_state()

    assert move.context.added_indices is not None
    assert len(move.context.deleted_atoms) == 0
    assert len(empty_atoms) == 303

    move.bias_towards_insert = 0.0

    old_atoms = empty_atoms.copy()

    for _ in range(100):
        assert move()
        gcmc.revert_state()
        assert empty_atoms == old_atoms

    assert len(empty_atoms) == 303

    while len(move.unique_labels) > 0:
        to_delete = rng.choice(move.unique_labels)
        move.to_delete_indices = to_delete
        assert move()
        gcmc.save_state()

        assert move.context.deleted_indices is not None
        assert move.to_delete_indices is None
        assert not np.isin(move.unique_labels, to_delete).any()


def test_displacement_consistency_manually(bulk_small):
    move = DisplacementMove(np.arange(len(bulk_small)))

    gcmc = GrandCanonical(
        bulk_small,
        temperature=1000,
        chemical_potential=0.0,
        max_cycles=1,
        number_of_exchange_particles=len(bulk_small),
        default_displacement_move=move,
    )

    for _ in range(100):
        assert move()
        bulk_small.get_potential_energy()
        gcmc.save_state()
        assert gcmc.last_results == bulk_small.calc.results
        assert move()
        gcmc.revert_state()
        assert len(bulk_small.calc.check_state(bulk_small)) == 0
        assert (
            bulk_small.calc.get_property("energy", bulk_small, allow_calculation=False)
            is not None
        )


def test_displacement_consistency_manually_2(bulk_small):
    move = DisplacementMove(np.arange(len(bulk_small)))

    gcmc = GrandCanonical(
        bulk_small,
        temperature=1000,
        chemical_potential=0.0,
        max_cycles=5,
        number_of_exchange_particles=len(bulk_small),
        default_displacement_move=move,
    )

    old_atoms = bulk_small.copy()

    for _ in range(100):
        assert move()
        gcmc.revert_state()
        assert bulk_small == old_atoms


def test_exchange_consistency_manually(empty_atoms, rng):
    exchange_atoms = Atoms("H", positions=[[0, 0, 0]])

    empty_atoms.set_cell(np.eye(3) * 100)

    for _ in range(100):
        empty_atoms.extend(exchange_atoms)

    empty_atoms.set_positions(rng.uniform(-50, 50, (100, 3)))

    move = ExchangeMove(exchange_atoms, np.arange(100), Translation())

    move_storage = MoveStorage(move, 0, 0, 0, DummyCriteria())

    gcmc = GrandCanonical(
        empty_atoms,
        temperature=1000,
        chemical_potential=0.0,
        max_cycles=1,
        number_of_exchange_particles=len(empty_atoms),
        default_exchange_move=move_storage,
    )

    for _ in range(100):
        assert move()
        empty_atoms.get_potential_energy()
        gcmc.save_state()
        assert gcmc.last_results == empty_atoms.calc.results
        assert move()
        gcmc.revert_state()
        assert len(empty_atoms.calc.check_state(empty_atoms)) == 0
        assert (
            empty_atoms.calc.get_property(
                "energy", empty_atoms, allow_calculation=False
            )
            is not None
        )


def test_grand_canonical_criteria(rng):
    context = ExchangeContext(Atoms(), rng)
    context.last_energy = 0.0

    dummy_calc = DummyCalculator()
    dummy_calc.dummy_value = -np.inf
    context.atoms.calc = dummy_calc

    context.number_of_exchange_particles = 1

    assert np.isnan(context.chemical_potential)

    context.chemical_potential = 0.0
    context.accessible_volume = 10

    context.particle_delta = 1

    context.temperature = 298

    with pytest.raises(ValueError):
        GrandCanonicalCriteria().evaluate(context)

    context.added_atoms = Atoms("Cu")

    criteria = GrandCanonicalCriteria()

    assert criteria.evaluate(context)

    dihydrogen = molecule("H2")

    debroglie_wavelength = (
        math.sqrt(
            _hplanck**2
            / (
                2
                * np.pi
                * dihydrogen.get_masses().sum()
                * kB
                * context.temperature
                / _Nav
                * 1e-3
                * _e
            )
        )
        * 1e10
    )

    assert np.allclose(debroglie_wavelength, 0.71228)  # Wikipedia

    context.added_atoms = dihydrogen

    probability = context.accessible_volume / (
        debroglie_wavelength**3
        * (context.number_of_exchange_particles + context.particle_delta)
    )

    assert np.allclose(probability, 13.83662789)

    context.chemical_potential = -10

    exponential = math.exp(context.chemical_potential / (context.temperature * kB))
    exponential *= math.exp(9.9 / (context.temperature * kB))

    probability *= exponential

    assert np.allclose(probability, 0.28172731)

    dummy_calc.dummy_value = -9.9

    criteria.evaluate(context)

    successes = sum(criteria.evaluate(context) for _ in range(100000))
    assert np.allclose(successes / 100000, probability, atol=0.01)
