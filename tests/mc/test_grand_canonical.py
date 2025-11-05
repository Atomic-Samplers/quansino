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
from quansino.mc.criteria import GrandCanonicalCriteria
from quansino.mc.gcmc import GrandCanonical
from quansino.moves.core import CompositeMove
from quansino.moves.displacement import DisplacementMove
from quansino.moves.exchange import ExchangeMove
from quansino.operations.displacement import Translation
from quansino.utils.moves import MoveStorage


def test_grand_canonical(bulk_large):
    """Test the `GrandCanonical` class."""
    bulk_large.calc = EMT()

    energy_full = bulk_large.get_potential_energy()

    del bulk_large[0]

    energy_minus_one = bulk_large.get_potential_energy()

    energy_difference = energy_full - energy_minus_one

    exchange_move = ExchangeMove(np.arange(len(bulk_large)), Translation())

    gcmc = GrandCanonical(
        bulk_large,
        Atoms("Cu"),
        temperature=1000,
        chemical_potential=energy_difference,
        max_cycles=1,
        number_of_exchange_particles=len(bulk_large),
    )

    gcmc.add_move(exchange_move)

    assert isinstance(gcmc.moves["default"].criteria, GrandCanonicalCriteria)

    assert gcmc.context.temperature == 1000
    assert gcmc.context.chemical_potential == energy_difference
    assert gcmc.max_cycles == 1
    assert gcmc.atoms == bulk_large

    gcmc.chemical_potential = 100.00

    assert gcmc.chemical_potential == 100.0
    assert gcmc.context.chemical_potential == 100.0

    gcmc.temperature = 2000

    assert gcmc.temperature == 2000
    assert gcmc.context.temperature == 2000

    del bulk_large[20]
    del bulk_large[10]
    del bulk_large[5]
    del bulk_large[2]

    gcmc.number_of_exchange_particles = len(bulk_large)

    assert gcmc.context.number_of_exchange_particles == len(bulk_large)
    assert gcmc.number_of_exchange_particles == len(bulk_large)

    exchange_move.set_labels(np.arange(len(bulk_large)))

    current_atoms_count = len(bulk_large)

    assert isinstance(gcmc.moves["default"].move, ExchangeMove)

    labels = np.array(gcmc.moves["default"].move.labels)

    for _ in gcmc.srun(10):

        assert gcmc.temperature == 2000
        assert gcmc.context.temperature == 2000

        if len(gcmc.atoms) != current_atoms_count:
            assert not bool(gcmc.context._added_atoms)
            assert not bool(gcmc.context._deleted_atoms)
            assert len(bulk_large) == gcmc.context.number_of_exchange_particles
            assert abs(current_atoms_count - len(bulk_large)) == 1
            assert gcmc.context.accessible_volume == bulk_large.cell.volume

            current_atoms_count = len(gcmc.atoms)
            labels = np.array(gcmc.moves["default"].move.labels)
        else:
            assert_array_equal(labels, gcmc.moves["default"].move.labels)
            assert current_atoms_count == len(gcmc.atoms)


def test_grand_canonical_simulation(bulk_medium):
    """Test the `GrandCanonical` class with a simulation."""
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
        GrandCanonicalCriteria(),
        10,
        0.5,
        1,
    )

    gcmc = GrandCanonical[DisplacementMove, GrandCanonicalCriteria](
        bulk_medium,
        Atoms("Cu"),
        temperature=1000,
        chemical_potential=energy_difference,
        max_cycles=1,
        number_of_exchange_particles=len(bulk_medium),
    )

    gcmc.moves["default_exchange_move"] = move_storage

    assert gcmc.moves["default_exchange_move"].move == move_storage.move
    assert gcmc.moves["default_exchange_move"].interval == 10
    assert gcmc.moves["default_exchange_move"].probability == 0.5
    assert gcmc.moves["default_exchange_move"].minimum_count == 1

    displacement_move = DisplacementMove(np.arange(len(bulk_medium)))

    labels = np.arange(len(bulk_medium))
    labels[: len(labels) // 2] = -1
    exchange_move = ExchangeMove(labels)

    gcmc = GrandCanonical(
        bulk_medium,
        Atoms("Cu"),
        temperature=1000,
        chemical_potential=energy_difference,
        max_cycles=1,
        default_exchange_move=exchange_move,
        default_displacement_move=displacement_move,
        number_of_exchange_particles=len(bulk_medium),
    )

    for index in range(10):
        gcmc.add_move(copy(displacement_move), name=f"displacement_{index}")

    assert isinstance(gcmc.moves["displacement_0"].move, DisplacementMove)

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

    for _ in gcmc.srun(10):
        assert gcmc.temperature == 2000
        assert gcmc.context.temperature == 2000
        assert gcmc.chemical_potential == energy_difference * 1.56
        assert gcmc.context.chemical_potential == energy_difference * 1.56

        if len(gcmc.atoms) != current_atoms_count:
            assert not bool(gcmc.context._added_atoms)
            assert not bool(gcmc.context._deleted_atoms)
            assert len(bulk_medium) == gcmc.context.number_of_exchange_particles
            assert abs(len(bulk_medium) - current_atoms_count) == 1

            assert gcmc.context.accessible_volume == bulk_medium.cell.volume

            assert len(gcmc.context._deleted_indices) == 0
            assert len(gcmc.context._added_indices) == 0
            assert len(gcmc.context._added_atoms) == 0
            assert len(gcmc.context._deleted_atoms) == 0

            current_atoms_count = len(gcmc.atoms)
            labels = np.array(gcmc.moves["default_exchange_move"].move.labels)
        else:
            assert_array_equal(labels, gcmc.moves["default_exchange_move"].move.labels)
            assert current_atoms_count == len(gcmc.atoms)

        assert gcmc.moves["displacement_0"].move.default_label == -1

        if gcmc.acceptance_rate:
            assert bulk_medium != old_atoms
            if "energy" in gcmc.context.last_results:
                assert not np.allclose(
                    gcmc.context.last_results["energy"], old_energy, atol=1e-8, rtol=0
                )
        else:
            assert bulk_medium == old_atoms
            if "energy" in gcmc.context.last_results:
                assert_allclose(gcmc.context.last_results["energy"], old_energy)

        old_atoms = bulk_medium.copy()
        old_energy = bulk_medium.get_potential_energy()


def test_grand_canonical_atomic_simulation(empty_atoms, rng):
    """Test the `GrandCanonical` class with single atom insertion and deletion."""
    initial_position = rng.uniform(-50, 50, (1, 3))
    exchange_atoms = Atoms("H", positions=initial_position)

    move = ExchangeMove(labels=[])

    gcmc = GrandCanonical(
        empty_atoms,
        exchange_atoms,
        temperature=1000,
        chemical_potential=0.0,
        max_cycles=1,
        number_of_exchange_particles=len(empty_atoms),
        default_exchange_move=move,
    )

    gcmc.moves["default_exchange_move"].criteria = DummyCriteria()

    move.bias_towards_insert = 1.0

    assert move(gcmc.context)

    assert len(empty_atoms) == 1
    assert_allclose(empty_atoms.positions, np.zeros((1, 3)))

    old_positions = empty_atoms.get_positions()

    gcmc.save_state()
    assert_array_equal(move.labels, [0])

    empty_atoms.set_cell(np.eye(3) * 100)

    old_atoms = empty_atoms.copy()

    move.check_move = lambda *args, **kwargs: False

    for _ in range(10):
        assert not move(gcmc.context)
        assert empty_atoms == old_atoms

    move.check_move = lambda *args, **kwargs: True

    for _ in range(100):
        assert move(gcmc.context)
        gcmc.revert_state()
        assert empty_atoms == old_atoms

    assert len(empty_atoms) == 1
    assert len(move.labels) == len(empty_atoms)
    assert_allclose(empty_atoms.get_positions(), old_positions)

    for _ in range(100):
        assert move(gcmc.context)
        gcmc.save_state()

    assert gcmc.context._added_indices is not None
    assert len(gcmc.context._deleted_indices) == 0
    assert len(empty_atoms) == 101
    assert len(move.labels) == len(empty_atoms)
    assert_array_equal(move.labels, list(np.arange(101)))

    move.bias_towards_insert = 0.0

    old_atoms = empty_atoms.copy()

    move.to_delete_label = 0

    assert move(gcmc.context)

    assert len(gcmc.context._added_indices) == 0
    assert len(gcmc.context._deleted_indices) == 1
    assert gcmc.context._deleted_indices == 0
    assert len(gcmc.context._deleted_atoms) == 1

    old__deleted_atoms = deepcopy(gcmc.context._deleted_atoms)

    gcmc.context._deleted_atoms = Atoms()

    with pytest.raises(ValueError):
        gcmc.revert_state()

    gcmc.context._deleted_atoms = old__deleted_atoms

    gcmc.revert_state()
    assert empty_atoms == old_atoms

    move.set_labels(np.full(101, -1))

    for _ in range(100):
        assert not move(gcmc.context)
        assert empty_atoms == old_atoms

    move.set_labels(np.arange(101))

    for _ in range(100):
        assert move(gcmc.context)
        gcmc.revert_state()
        assert empty_atoms == old_atoms

    while len(move.labels) > 1:
        if rng.random() < 0.7:
            to_delete = rng.choice(move.unique_labels).astype(int)
            move.to_delete_label = to_delete
            is_deletion = True
        else:
            move.to_add_atoms = Atoms("C", positions=[[0, 0, 0]], cell=np.eye(3) * 20)
            is_deletion = False

        assert move(gcmc.context)
        gcmc.save_state()

        if is_deletion:
            assert gcmc.context._deleted_indices is not None
            assert move.to_delete_label is None

            assert not np.isin(move.labels, to_delete).any()  # type: ignore
        else:
            assert_allclose(gcmc.context.atoms.cell.array, np.eye(3) * 100)
            assert gcmc.context._added_indices is not None
            assert move.to_add_atoms is None


def test_grand_canonical_molecular_simulation(empty_atoms, rng):
    """Test the `GrandCanonical` class with molecular insertion and deletion."""
    exchange_atoms = molecule("H2O")
    move = ExchangeMove([])

    old_distance = np.linalg.norm(
        exchange_atoms.positions[:, None] - exchange_atoms.positions, axis=-1
    )

    move_storage = MoveStorage(move, DummyCriteria(), 0, 0, 0)

    gcmc = GrandCanonical(
        empty_atoms,
        exchange_atoms,
        temperature=1000,
        chemical_potential=0.0,
        max_cycles=1,
        number_of_exchange_particles=len(empty_atoms),
    )

    gcmc.moves["default_exchange_move"] = move_storage

    move.bias_towards_insert = 0.0

    move.to_delete_label = -1

    assert not move(gcmc.context)
    assert len(empty_atoms) == 0

    move.to_add_atoms = exchange_atoms

    assert move(gcmc.context)
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

    move.bias_towards_insert = 1.0

    for _ in range(1000):
        assert move(gcmc.context)
        gcmc.revert_state()
        assert empty_atoms == old_atoms

    for _ in range(100):
        assert move(gcmc.context)
        gcmc.save_state()

    assert gcmc.context._added_indices is not None
    assert len(gcmc.context._deleted_atoms) == 0
    assert len(empty_atoms) == 303

    move.bias_towards_insert = 0.0

    old_atoms = empty_atoms.copy()

    for _ in range(100):
        assert move(gcmc.context)
        gcmc.revert_state()
        assert empty_atoms == old_atoms

    assert len(empty_atoms) == 303

    while len(move.unique_labels) > 0:
        to_delete = rng.choice(move.unique_labels)
        move.to_delete_label = to_delete
        assert move(gcmc.context)
        gcmc.save_state()

        assert gcmc.context._deleted_indices is not None
        assert move.to_delete_label is None
        assert not np.isin(move.unique_labels, to_delete).any()


def test_displacement_consistency_manually(bulk_small):
    """Test that the DisplacementMove works correctly with manual state management."""
    move = DisplacementMove(np.arange(len(bulk_small)))

    gcmc = GrandCanonical(
        bulk_small,
        Atoms("H"),
        temperature=1000,
        chemical_potential=0.0,
        max_cycles=1,
        number_of_exchange_particles=len(bulk_small),
        default_displacement_move=move,
    )

    for _ in range(100):
        assert move(gcmc.context)
        bulk_small.get_potential_energy()
        gcmc.save_state()
        assert gcmc.context.last_results == bulk_small.calc.results
        assert move(gcmc.context)
        gcmc.revert_state()
        assert len(bulk_small.calc.check_state(bulk_small)) == 0
        assert (
            bulk_small.calc.get_property("energy", bulk_small, allow_calculation=False)
            is not None
        )


def test_displacement_consistency_manually_2(bulk_small):
    """Test that the DisplacementMove works correctly with manual state management."""
    move = DisplacementMove(np.arange(len(bulk_small)))

    gcmc = GrandCanonical(
        bulk_small,
        exchange_atoms=Atoms("H"),
        temperature=1000,
        chemical_potential=0.0,
        max_cycles=5,
        number_of_exchange_particles=len(bulk_small),
        default_displacement_move=move,
    )

    old_atoms = bulk_small.copy()

    for _ in range(100):
        assert move(gcmc.context)
        gcmc.revert_state()
        assert bulk_small == old_atoms


def test_exchange_consistency_manually(empty_atoms, rng):
    """Test that the ExchangeMove works correctly with manual state management."""
    exchange_atoms = Atoms("H", positions=[[0, 0, 0]])

    empty_atoms.set_cell(np.eye(3) * 100)

    for _ in range(100):
        empty_atoms.extend(exchange_atoms)

    empty_atoms.set_positions(rng.uniform(-50, 50, (100, 3)))

    move = ExchangeMove(np.arange(100), Translation())

    GrandCanonical.default_criteria[ExchangeMove] = DummyCriteria

    gcmc = GrandCanonical(
        empty_atoms,
        exchange_atoms,
        temperature=1000,
        chemical_potential=0.0,
        max_cycles=1,
        number_of_exchange_particles=len(empty_atoms),
        default_exchange_move=move,
    )

    for _ in range(100):
        assert move(gcmc.context)
        empty_atoms.get_potential_energy()
        gcmc.save_state()
        assert gcmc.context.last_results == empty_atoms.calc.results
        assert move(gcmc.context)
        gcmc.revert_state()
        assert len(empty_atoms.calc.check_state(empty_atoms)) == 0
        assert (
            empty_atoms.calc.get_property(
                "energy", empty_atoms, allow_calculation=False
            )
            is not None
        )


def test_grand_canonical_criteria(rng):
    """Test the `GrandCanonicalCriteria` class."""
    context = ExchangeContext(Atoms(), rng)
    context.last_potential_energy = 0.0

    dummy_calc = DummyCalculator()
    dummy_calc.dummy_value = -np.inf
    context.atoms.calc = dummy_calc

    context.number_of_exchange_particles = 1

    assert np.isnan(context.chemical_potential)

    context.chemical_potential = 0.0
    context.accessible_volume = 10

    context.temperature = 298

    context._added_atoms = Atoms("Cu")

    criteria = GrandCanonicalCriteria()
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

    context.exchange_atoms = dihydrogen
    context.particle_delta = 1

    probability = context.accessible_volume / (
        debroglie_wavelength**3 * (context.number_of_exchange_particles + 1)
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


def test_grand_canonical_restart(bulk_small, tmp_path, rng):
    """Test the restart functionality of the `GrandCanonical` Monte Carlo class."""
    displacement_move = DisplacementMove([0, 1, -1, -1]) + DisplacementMove(
        [2, 3, -1, -1]
    )
    exchange_move = ExchangeMove(np.arange(len(bulk_small)), Translation())

    move = displacement_move + exchange_move

    assert type(move) is CompositeMove

    mc = GrandCanonical(
        bulk_small,
        Atoms("Cu"),
        temperature=10000.0,
        chemical_potential=100,
        max_cycles=1,
        logfile=tmp_path / "mc.log",
        trajectory=tmp_path / "mc.xyz",
        restart_file=tmp_path / "mc_restart.json",
        logging_interval=5,
    )

    mc.add_move(move, criteria=GrandCanonicalCriteria(), name="default")

    data = mc.to_dict()

    reconstructed_mc = GrandCanonical.from_dict(data)
    reconstructed_mc.atoms.calc = bulk_small.calc  # calc are not serialized...

    assert reconstructed_mc.atoms == mc.atoms
    assert reconstructed_mc.temperature == mc.temperature
    assert reconstructed_mc.chemical_potential == mc.chemical_potential
    assert reconstructed_mc.max_cycles == mc.max_cycles
    assert reconstructed_mc.step_count == mc.step_count

    assert reconstructed_mc.context.last_results == mc.context.last_results
    assert np.isnan(reconstructed_mc.context.last_potential_energy)
    assert_allclose(reconstructed_mc.context.last_positions, mc.context.last_positions)
    assert reconstructed_mc.moves.keys() == mc.moves.keys()

    assert mc.default_logger is not None
    assert reconstructed_mc.default_logger is None

    assert mc.default_trajectory is not None
    assert reconstructed_mc.default_trajectory is None

    assert mc.default_restart is not None
    assert reconstructed_mc.default_restart is None

    assert reconstructed_mc._seed == mc._seed  # type: ignore
    assert reconstructed_mc._rng.bit_generator.state == mc._rng.bit_generator.state
    assert reconstructed_mc._rng.random() == mc._rng.random()
    assert reconstructed_mc._rng.integers(0, 100) == mc._rng.integers(0, 100)

    energies = []

    mc.context.last_potential_energy = np.nan
    reconstructed_mc.context.last_potential_energy = np.nan

    energies = [mc.context.last_potential_energy for _ in mc.irun(20)]

    energies_reconstructed = [
        reconstructed_mc.context.last_potential_energy
        for _ in reconstructed_mc.irun(20)
    ]

    assert_allclose(energies, energies_reconstructed)

    energies = []

    for _ in mc.srun(20):
        assert len(mc.move_history) == 1
        assert mc.move_history[0][0] == "default"

        energies.append(mc.context.last_potential_energy)

    energies_reconstructed = []

    i = 0
    current_mc = reconstructed_mc
    while i < 20:
        for _ in range(20 - i):
            for _ in current_mc.step():
                ...
            i += 1
            energies_reconstructed.append(current_mc.context.last_potential_energy)

            if rng.random() < 0.5:
                break

        current_mc = GrandCanonical.from_dict(current_mc.to_dict())
        current_mc.atoms.calc = bulk_small.calc

    assert_allclose(energies, energies_reconstructed)

    mc_reconstructed = GrandCanonical.from_dict(
        mc.to_dict(),
        logfile=tmp_path / "mc.log",
        trajectory=tmp_path / "mc.xyz",
        restart_file=tmp_path / "mc_restart.json",
        logging_interval=6,
        max_cycles=10,
    )

    assert str(mc_reconstructed.default_logger) == str(mc.default_logger)
    assert str(mc_reconstructed.default_trajectory) == str(mc.default_trajectory)
    assert str(mc_reconstructed.default_restart) == str(mc.default_restart)
    assert mc_reconstructed.logging_interval == 6
    assert mc_reconstructed.max_cycles == 10
