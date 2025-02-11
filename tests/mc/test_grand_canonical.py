from __future__ import annotations

import math
from copy import copy

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.units import _e, _hplanck, _Nav, kB
from numpy.testing import assert_array_equal

from quansino.mc.contexts import ExchangeContext
from quansino.mc.gcmc import GrandCanonical, GrandCanonicalCriteria
from quansino.moves.displacements import DisplacementMove
from quansino.moves.exchange import ExchangeMove
from quansino.moves.operations import Translation


def test_grand_canonical(bulk_large):
    bulk_large.calc = EMT()

    energy_full = bulk_large.get_potential_energy()

    del bulk_large[0]

    energy_minus_one = bulk_large.get_potential_energy()

    energy_difference = energy_full - energy_minus_one

    exchange_move = ExchangeMove(Atoms("Cu"), np.arange(len(bulk_large)), Translation())

    gcmc = GrandCanonical[DisplacementMove, ExchangeContext](
        bulk_large,
        temperature=1000,
        chemical_potential=energy_difference,
        num_cycles=1,
        number_of_particles=len(bulk_large),
        seed=42,
    )

    gcmc.add_move(exchange_move)

    assert gcmc.context.temperature == 1000
    assert gcmc.context.chemical_potential == energy_difference
    assert gcmc.num_cycles == 1
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

    gcmc.number_of_particles = len(bulk_large)

    assert gcmc.context.number_of_particles == len(bulk_large)
    assert gcmc.number_of_particles == len(bulk_large)

    exchange_move.set_labels(np.arange(len(bulk_large)))

    current_atoms_count = len(bulk_large)
    labels = np.array(gcmc.context.moves["default"].move.labels)

    for _ in gcmc.irun(1000):
        assert gcmc.temperature == 2000
        assert gcmc.context.temperature == 2000
        assert gcmc.chemical_potential == energy_difference * 1.56
        assert gcmc.context.chemical_potential == energy_difference * 1.56

        if len(gcmc.atoms) != current_atoms_count:
            assert bool(gcmc.context.added_atoms) ^ bool(gcmc.context.deleted_atoms)
            assert len(bulk_large) == gcmc.context.number_of_particles
            assert len(bulk_large) == current_atoms_count + gcmc.context.particle_delta

            assert gcmc.context.accessible_volume == bulk_large.cell.volume

            if gcmc.context.added_atoms:
                assert gcmc.context.particle_delta == len(gcmc.context.added_atoms)
                assert len(gcmc.context.added_indices) == len(gcmc.context.added_atoms)

                assert len(labels) + len(gcmc.context.added_indices) == len(
                    gcmc.moves["default"].move.labels
                )
                assert max(labels) + 1 in gcmc.context.moves["default"].move.labels
            elif gcmc.context.deleted_atoms:
                assert gcmc.context.particle_delta == -len(gcmc.context.deleted_atoms)
                assert len(gcmc.context.deleted_indices) == len(
                    gcmc.context.deleted_atoms
                )

                assert (
                    labels[gcmc.context.deleted_indices]
                    not in gcmc.context.moves["default"].move.labels
                )

            current_atoms_count = len(gcmc.atoms)
            labels = np.array(gcmc.context.moves["default"].move.labels)
        else:
            assert_array_equal(labels, gcmc.context.moves["default"].move.labels)
            assert current_atoms_count == len(gcmc.atoms)

    assert len(gcmc.atoms) == 901


def test_grand_canonical_defaults(bulk_medium):
    bulk_medium.calc = EMT()

    energy_full = bulk_medium.get_potential_energy()

    del bulk_medium[0]

    energy_minus_one = bulk_medium.get_potential_energy()

    del bulk_medium[20]
    del bulk_medium[10]
    del bulk_medium[5]
    del bulk_medium[2]

    energy_difference = energy_full - energy_minus_one

    displacement_move = DisplacementMove(np.arange(len(bulk_medium)))

    labels = np.arange(len(bulk_medium))
    labels[: len(labels) // 2] = -1
    exchange_move = ExchangeMove(Atoms("Cu"), labels)

    gcmc = GrandCanonical[DisplacementMove, ExchangeContext](
        bulk_medium,
        temperature=1000,
        chemical_potential=energy_difference,
        num_cycles=1,
        default_exchange_move=exchange_move,
        default_displacement_move=displacement_move,
        number_of_particles=len(bulk_medium),
        seed=42,
    )

    for index in range(10):
        gcmc.add_move(copy(displacement_move), name=f"displacement_{index}")

    gcmc.moves["displacement_0"].move.set_labels(labels)
    gcmc.moves["displacement_0"].move.default_label = -1

    assert gcmc.context.temperature == 1000
    assert gcmc.context.chemical_potential == energy_difference
    assert gcmc.num_cycles == 1
    assert gcmc.atoms == bulk_medium

    gcmc.chemical_potential = 0.0

    assert gcmc.chemical_potential == 0.0
    assert gcmc.context.chemical_potential == 0.0

    gcmc.temperature = 2000

    assert gcmc.temperature == 2000
    assert gcmc.context.temperature == 2000

    gcmc.chemical_potential = energy_difference * 1.56

    assert gcmc.context.number_of_particles == len(bulk_medium)
    assert gcmc.number_of_particles == len(bulk_medium)

    current_atoms_count = len(bulk_medium)
    labels = np.array(gcmc.context.moves["default_exchange"].move.labels)

    for _ in gcmc.irun(1000):
        assert gcmc.temperature == 2000
        assert gcmc.context.temperature == 2000
        assert gcmc.chemical_potential == energy_difference * 1.56
        assert gcmc.context.chemical_potential == energy_difference * 1.56

        if len(gcmc.atoms) != current_atoms_count:
            assert bool(gcmc.context.added_atoms) ^ bool(gcmc.context.deleted_atoms)
            assert len(bulk_medium) == gcmc.context.number_of_particles
            assert len(bulk_medium) == current_atoms_count + gcmc.context.particle_delta

            assert gcmc.context.accessible_volume == bulk_medium.cell.volume

            if gcmc.context.added_atoms:
                assert gcmc.context.particle_delta == len(gcmc.context.added_atoms)
                assert len(gcmc.context.added_indices) == len(gcmc.context.added_atoms)

                assert len(labels) + len(gcmc.context.added_indices) == len(
                    gcmc.moves["default_exchange"].move.labels
                )
                assert (
                    max(labels) + 1
                    in gcmc.context.moves["default_exchange"].move.labels
                )
            elif gcmc.context.deleted_atoms:
                assert gcmc.context.particle_delta == -len(gcmc.context.deleted_atoms)
                assert len(gcmc.context.deleted_indices) == len(
                    gcmc.context.deleted_atoms
                )

                assert (
                    labels[gcmc.context.deleted_indices]
                    not in gcmc.context.moves["default_exchange"].move.labels
                )

            current_atoms_count = len(gcmc.atoms)
            labels = np.array(gcmc.context.moves["default_exchange"].move.labels)
        else:
            assert_array_equal(
                labels, gcmc.context.moves["default_exchange"].move.labels
            )
            assert current_atoms_count == len(gcmc.atoms)

        assert gcmc.context.moves["displacement_0"].move.default_label == -1


def test_grand_canonical_criteria(rng):
    context = ExchangeContext(Atoms(), rng, {})

    context.number_of_particles = 1
    context.accessible_volume = 10

    context.particle_delta = 1

    context.temperature = 298

    context.added_atoms = Atoms("Cu")

    criteria = GrandCanonicalCriteria()

    assert criteria.evaluate(context, -np.inf)

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
        debroglie_wavelength**3 * (context.number_of_particles + context.particle_delta)
    )

    assert np.allclose(probability, 13.83662789)

    context.chemical_potential = -10

    exponential = math.exp(context.chemical_potential / (context.temperature * kB))
    exponential *= math.exp(9.9 / (context.temperature * kB))

    probability *= exponential

    assert np.allclose(probability, 0.28172731)

    criteria.evaluate(context, -9.9)

    successes = sum(criteria.evaluate(context, -9.9) for _ in range(100000))

    assert np.allclose(successes / 100000, probability, atol=0.01)
