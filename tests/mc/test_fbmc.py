from __future__ import annotations

import numpy as np
import pytest
from ase.constraints import FixAtoms, FixCom
from numpy.testing import assert_allclose

from quansino.mc.fbmc import ForceBias


@pytest.fixture(name="fbmc")
def fbmc_fixture(bulk_small, tmp_path):
    return ForceBias(
        bulk_small,
        delta=0.01,
        logfile=tmp_path / "test.log",
        trajectory=tmp_path / "test.traj",
        seed=42,
        temperature=5000,
    )


def test_force_bias(fbmc):
    displacements = np.abs([fbmc.atoms.get_velocities() for _ in fbmc.irun(200)]).mean(
        axis=0
    )  # type: ignore

    assert_allclose(displacements, 0.01 / 3, atol=0.01 / 9)


def test_force_bias_mass_setter(fbmc):
    fbmc.masses_scaling_power = 0.5

    assert_allclose(fbmc.masses_scaling_power, 0.5)

    fbmc.atoms[1].symbol = "Au"

    fbmc.masses_scaling_power = {"Cu": 0.5, "Au": 0.1}

    expected = np.full((len(fbmc.atoms), 3), 0.5)
    expected[1, :] = 0.1

    expected_masses = np.full((len(fbmc.atoms), 3), 63.546)
    expected_masses[1, :] = 196.96657
    fbmc.update_masses()

    assert_allclose(fbmc._masses, expected_masses)
    assert_allclose(fbmc.masses_scaling_power, expected)

    rng = np.random.default_rng()
    random = rng.random((len(fbmc.atoms), 3))

    fbmc.masses_scaling_power = random

    assert_allclose(fbmc.masses_scaling_power, random)

    with pytest.raises(ValueError):
        fbmc.masses_scaling_power = [0.5, 0.5]


def test_todict(fbmc):
    dictionary = fbmc.todict()

    assert dictionary["type"] == "monte-carlo"
    assert dictionary["mc-type"] == "ForceBias"
    assert dictionary["seed"] == 42
    assert dictionary["nsteps"] == 0
    assert dictionary["temperature"] == 5000
    assert dictionary["delta"] == 0.01
    assert_allclose(
        dictionary["masses_scaling_power"], np.full((len(fbmc.atoms), 3), 0.25)
    )
    assert dictionary["rng_state"] is not None


def test_constraints(fbmc):
    fbmc.atoms.set_constraint(FixAtoms(indices=[0, 1]))

    fbmc.run(1)

    displacements = np.abs([fbmc.atoms.get_velocities() for _ in fbmc.irun(50)]).mean(
        axis=0
    )  # type: ignore

    assert_allclose(displacements[:2], 0)
    assert_allclose(displacements[2:], 0.01 / 3, atol=0.001)


def test_warning(bulk_small, recwarn):
    with pytest.warns(UserWarning):
        ForceBias(bulk_small, delta=0.01, seed=42, temperature=5000, logfile="-")

    bulk_small.set_constraint(FixCom())
    ForceBias(bulk_small, delta=0.01, seed=42, temperature=5000, logfile="-")

    assert len(recwarn) == 0
