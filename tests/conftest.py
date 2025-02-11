from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.eam import EAM
from ase.calculators.emt import EMT


@pytest.fixture(name="eam_calc")
def eam_calc_fixture():
    path = Path(__file__).parent
    return EAM(
        potential=str(
            Path(path, ".potentials/cu_eam_mishin2001.alloy").resolve().expanduser()
        )
    )


@pytest.fixture(name="bulk_small")
def bulk_small_fixture(eam_calc):
    atoms = bulk("Cu", cubic=True)
    atoms.calc = eam_calc
    return atoms


@pytest.fixture(name="bulk_medium")
def bulk_medium_fixture(eam_calc):
    atoms = bulk("Cu", cubic=True) * (3, 3, 3)
    atoms.calc = eam_calc
    return atoms


@pytest.fixture(name="bulk_large")
def bulk_large_fixture(eam_calc):
    atoms = bulk("Cu", cubic=True) * (6, 6, 6)
    atoms.calc = eam_calc
    return atoms


@pytest.fixture(name="single_atom")
def single_atom_fixture():
    atom = Atoms("H", positions=[[0, 0, 0]])
    atom.calc = EMT()
    return atom


@pytest.fixture(name="empty_atoms")
def empty_atoms_fixture():
    empty_atoms = Atoms()
    empty_atoms.calc = EMT()
    return empty_atoms


@pytest.fixture(name="rng")
def rng_fixture():
    return np.random.default_rng()
