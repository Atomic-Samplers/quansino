from __future__ import annotations

import numpy as np
from ase.atoms import Atoms
from ase.build import molecule
from ase.constraints import FixAtoms, FixCom, FixedPlane
from numpy.testing import assert_allclose, assert_array_equal

from quansino.utils.atoms import has_constraint, reinsert_atoms, search_molecules


def test_has_constraint(bulk_large):
    bulk_large.set_constraint(FixAtoms(indices=[0]))

    assert has_constraint(bulk_large, FixAtoms)
    assert has_constraint(bulk_large, "FixAtoms")

    bulk_large.set_constraint(FixCom())

    assert has_constraint(bulk_large, FixCom)
    assert has_constraint(bulk_large, "FixCom")

    bulk_large.set_constraint(FixedPlane([0, 1], (0, 0, 1)))

    assert has_constraint(bulk_large, FixedPlane)
    assert has_constraint(bulk_large, "FixedPlane")
    assert not has_constraint(bulk_large, "FixConstraint")
    assert not has_constraint(bulk_large, "FixCartesian")
    assert not has_constraint(bulk_large, "FixBondLength")


def test_search_molecules(bulk_large, rng):
    bulk_large.rattle(0.1)
    molecules = search_molecules(bulk_large, 2.9, required_size=0)

    assert_array_equal(molecules, np.full(len(bulk_large), -1))

    molecules = search_molecules(bulk_large, 2.9, required_size=1)

    assert_array_equal(molecules, np.full(len(bulk_large), -1))

    molecules = search_molecules(bulk_large, 2.9, required_size=(0, 10000))

    assert_array_equal(molecules, np.full(len(bulk_large), 0))

    atoms = Atoms()
    atoms.set_cell(np.eye(3) * 100)

    choices = ["H2", "H2O", "CH4"]

    counts = {choice: 0 for choice in choices}

    X, Y, Z = np.meshgrid(
        np.arange(0, 100, 10), np.arange(0, 100, 10), np.arange(0, 100, 10)
    )

    grid = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    for _ in grid:
        to_add = rng.choice(choices)
        counts[to_add] += 1

        to_add = molecule(to_add)

        to_add.translate(_)

        atoms += to_add

    molecules = search_molecules(atoms, 2.9, required_size=2)
    assert len(np.unique(molecules[molecules != -1])) == counts["H2"]

    molecules = search_molecules(atoms, 2.9, required_size=3)
    assert len(np.unique(molecules[molecules != -1])) == counts["H2O"]

    molecules = search_molecules(atoms, 2.9, required_size=4)
    assert len(molecules[molecules != -1]) == 0

    molecules = search_molecules(atoms, 2.9, required_size=5)
    assert len(np.unique(molecules[molecules != -1])) == counts["CH4"]
    assert len(molecules) == len(atoms)

    molecules = search_molecules(atoms, 2.9, required_size=(2, 3))
    assert len(np.unique(molecules[molecules != -1])) == counts["H2"] + counts["H2O"]

    molecules = search_molecules(atoms, 2.9)
    assert len(np.unique(molecules[molecules != -1])) == sum(counts.values())

    molecules = search_molecules(
        atoms, {("C", "H"): 2.1, ("H", "O"): 1.2}, required_size=(3, 5)
    )

    assert len(np.unique(molecules[molecules != -1])) == counts["H2O"] + counts["CH4"]


def test_reinsert_atoms(bulk_small, rng):
    other_bulk_atoms = bulk_small.copy()

    other_bulk_atoms.set_initial_charges(rng.random(len(other_bulk_atoms)))

    reinsert_atoms(bulk_small, other_bulk_atoms, [0, 1, 2, 3])

    assert len(bulk_small) == 2 * len(other_bulk_atoms)

    assert_allclose(
        bulk_small.get_initial_charges(),
        np.concatenate(
            [other_bulk_atoms.get_initial_charges(), np.zeros(len(other_bulk_atoms))]
        ),
    )

    assert_allclose(
        bulk_small.positions[: len(other_bulk_atoms)], other_bulk_atoms.positions
    )

    bulk_small.get_masses()

    reinsert_atoms(bulk_small, other_bulk_atoms, [-4, -3, -2, -1])

    assert len(bulk_small) == 3 * len(other_bulk_atoms)

    assert_allclose(
        bulk_small.get_initial_charges(),
        np.concatenate(
            [
                other_bulk_atoms.get_initial_charges(),
                np.zeros(len(other_bulk_atoms)),
                other_bulk_atoms.get_initial_charges(),
            ]
        ),
    )

    assert_allclose(
        bulk_small.positions[: len(other_bulk_atoms)], other_bulk_atoms.positions
    )
