from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.eam import EAM


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
