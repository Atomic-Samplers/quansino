from __future__ import annotations

import numpy as np
import pytest
from ase.constraints import FixAtoms
from numpy.testing import assert_allclose

from quansino.mc.fbmc import ForceBias


def test_force_bias(bulk_small, tmp_path):
    """Test the `ForceBias` Monte Carlo class."""
    fbmc = ForceBias(
        bulk_small,
        delta=0.01,
        logfile=tmp_path / "test.log",
        trajectory=tmp_path / "test.traj",
        seed=42,
        temperature=5000,
    )
    displacements = np.abs([fbmc.atoms.get_velocities() for _ in fbmc.irun(200)]).mean(
        axis=0
    )

    assert_allclose(displacements, 0.01 / 3, atol=0.01 / 9)


def test_force_bias_mass_setter(bulk_small, tmp_path):
    """Test the `ForceBias` Monte Carlo class with mass scaling."""
    fbmc = ForceBias(
        bulk_small,
        delta=0.01,
        logfile=tmp_path / "test.log",
        trajectory=tmp_path / "test.traj",
        seed=42,
        temperature=5000,
    )

    assert_allclose(fbmc.masses_scaling_power, 0.25)

    fbmc.atoms.symbols[1] = "Au"

    fbmc.masses_scaling_power = {"Cu": 0.5, "Au": 0.1}

    expected = np.full((len(fbmc.atoms), 3), 0.5)
    expected[1, :] = 0.1

    expected_masses = np.full((len(fbmc.atoms), 3), 63.546)
    expected_masses[1, :] = 196.96657
    fbmc.update_masses()

    assert_allclose(fbmc.shaped_masses, expected_masses)

    a = fbmc.masses_scaling_power
    assert_allclose(a, expected)

    rng = np.random.default_rng()
    random = rng.random((len(fbmc.atoms), 3))

    fbmc.masses_scaling_power = random

    assert_allclose(fbmc.masses_scaling_power, random)

    with pytest.raises(ValueError):
        fbmc.masses_scaling_power = [0.5, 0.5]  # type: ignore[test-type]


def test_fbmc_restart(bulk_small, tmp_path):
    """Test the restart functionality of the `ForceBias` Monte Carlo class."""
    fbmc = ForceBias(
        bulk_small,
        delta=0.01,
        logfile=tmp_path / "test.log",
        trajectory=tmp_path / "test.traj",
        seed=42,
        temperature=5000,
    )

    dictionary = fbmc.to_dict()

    assert dictionary["name"] == "ForceBias"
    assert dictionary["kwargs"]["seed"] == 42
    assert dictionary["attributes"]["step_count"] == 0
    assert dictionary["kwargs"]["temperature"] == 5000
    assert dictionary["kwargs"]["delta"] == 0.01
    assert_allclose(
        dictionary["attributes"]["masses_scaling_power"],
        np.full((len(fbmc.atoms), 3), 0.25),
    )
    assert dictionary["rng_state"] is not None


def test_constraints(bulk_small, tmp_path):
    """Test the `ForceBias` Monte Carlo class with constraints."""
    fbmc = ForceBias(
        bulk_small,
        delta=0.01,
        logfile=tmp_path / "test.log",
        trajectory=tmp_path / "test.traj",
        seed=42,
        temperature=5000,
    )
    fbmc.atoms.set_constraint(FixAtoms(indices=[0, 1]))

    fbmc.run(1)

    displacements = np.abs([fbmc.atoms.get_velocities() for _ in fbmc.irun(50)]).mean(
        axis=0
    )

    assert_allclose(displacements[:2], 0)
    assert_allclose(displacements[2:], 0.01 / 3, atol=0.001)


def test_warning(bulk_small):
    """Test COM warning in `ForceBias`."""
    with pytest.warns(
        UserWarning, match="can lead to sustained drift of the center of mass."
    ):
        ForceBias(bulk_small, delta=0.01, seed=42, temperature=5000, logfile=None)
