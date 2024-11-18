from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT


@pytest.fixture(name="atoms", params=[bulk("Cu") * (3, 3, 3)])
def atoms_fixture(request):
    request.param.calc = EMT()
    return request.param
