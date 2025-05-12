from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.eam import EAM
from ase.calculators.emt import EMT

from quansino.mc.criteria import Criteria
from quansino.operations.core import Operation


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


class DummyOperation(Operation):

    def calculate(self, context):
        context.atoms.set_positions(np.zeros((len(context.atoms), 3)))


class DummyCalculator:
    def __init__(self):
        self.dummy_value = -1.0
        self.results = {"energy": self.dummy_value}

    def get_potential_energy(self, *_args, **_kwargs):
        self.results["energy"] = self.dummy_value
        return self.dummy_value


class DummyCriteria(Criteria):
    @staticmethod
    def evaluate():
        return True


class DummySimulation:
    def __init__(self, atoms, context, moves):
        self.atoms = atoms
        self.context = context
        self.moves = moves
