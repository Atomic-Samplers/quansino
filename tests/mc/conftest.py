from __future__ import annotations

import pytest
from ase.build import bulk, molecule
from ase.calculators.emt import EMT


@pytest.fixture(name="atoms", params=[bulk("Cu"), molecule("H2O", vacuum=5)])
def atoms_fixture(request):
    request.param.calc = EMT()
    return request.param
