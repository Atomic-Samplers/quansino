from __future__ import annotations

import numpy as np
import pytest
from ase.constraints import FixAtoms
from numpy.testing import assert_allclose
from tests.conftest import EMTUncertaintyReadyCalculator

from quansino.mc.fbmc import AdaptiveForceBias, ForceBias


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
        fbmc.masses_scaling_power = [0.5, 0.5]  # type: ignore


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


def test_afbmc_forces_scheme(bulk_small, tmp_path):
    """Test the `AdaptiveForceBias` Monte Carlo class with a different scheme."""
    bulk_small.calc = EMTUncertaintyReadyCalculator(len(bulk_small))

    fbmc = AdaptiveForceBias(
        bulk_small,
        min_delta=0.05,
        max_delta=0.1,
        seed=42,
        temperature=5000,
        logfile=tmp_path / "afbmc.log",
    )

    for _ in fbmc.irun(200):
        assert isinstance(fbmc.delta, np.ndarray)

        wrong_values = (fbmc.delta < 0.05) | (fbmc.delta > 0.1)
        assert not np.any(wrong_values)

        uncertainties = fbmc.atoms.calc.results.get(  # type: ignore
            AdaptiveForceBias.forces_variance_keyword, None
        )
        assert uncertainties is not None

        assert_allclose(
            fbmc.delta,
            fbmc.min_delta
            + (fbmc.max_delta - fbmc.min_delta)
            * fbmc.update_functions[fbmc.update_function](fbmc.variation_coef),
        )

        assert fbmc.get_forces_variation_coef(fbmc.atoms).shape == (len(fbmc.atoms), 3)

        assert_allclose(
            fbmc.get_forces_variation_coef(fbmc.atoms),
            np.std(uncertainties, axis=0) / np.mean(uncertainties, axis=0),
        )

    assert fbmc.tanh_update(fbmc.reference_variance) == pytest.approx(0.5)
    assert fbmc.tanh_update(0) == pytest.approx(1.0)
    assert fbmc.tanh_update(1 / 3) == pytest.approx(0.050074186913)
    assert fbmc.tanh_update(np.inf) == pytest.approx(0.0)

    assert fbmc.exp_update(fbmc.reference_variance) == pytest.approx(0.5)
    assert fbmc.exp_update(0) == pytest.approx(1.0)
    assert fbmc.exp_update(1 / 3) == pytest.approx(0.099212565748)
    assert fbmc.exp_update(np.inf) == pytest.approx(0.0)

    with open(tmp_path / "afbmc.log") as f:
        lines = f.readlines()

        assert "MeanDelta" in lines[0]
        assert "MinDelta" in lines[0]
        assert "MaxDelta" in lines[0]
        assert "MeanForcesVar" in lines[0]


def test_afbmc_energy_scheme(bulk_small, tmp_path):
    """Test the `AdaptiveForceBias` Monte Carlo class with a different scheme."""
    bulk_small.calc = EMTUncertaintyReadyCalculator(len(bulk_small))

    fbmc = AdaptiveForceBias(
        bulk_small,
        min_delta=0.05,
        max_delta=0.1,
        seed=42,
        temperature=5000,
        logfile=tmp_path / "afbmc.log",
        scheme="energy",
    )

    for _ in fbmc.irun(200):
        assert isinstance(fbmc.delta, float)

        assert fbmc.min_delta <= fbmc.delta <= fbmc.max_delta

        uncertainties = fbmc.atoms.calc.results.get(  # type: ignore
            AdaptiveForceBias.energies_variance_keyword, None
        )
        assert uncertainties is not None

        assert fbmc.delta == pytest.approx(
            fbmc.min_delta
            + (fbmc.max_delta - fbmc.min_delta)
            * fbmc.update_functions[fbmc.update_function](fbmc.variation_coef)
        )

        assert isinstance(fbmc.get_energy_variation_coef(fbmc.atoms), float)

        assert fbmc.get_energy_variation_coef(fbmc.atoms) == pytest.approx(
            np.std(uncertainties, axis=0) / len(fbmc.atoms)
        )

    with open(tmp_path / "afbmc.log") as f:
        lines = f.readlines()

        assert "Delta" in lines[0]
        assert "EnergyVar" in lines[0]


def test_afbmc_warnings(bulk_small):
    """Test warnings in `AdaptiveForceBias`."""
    afbmc = AdaptiveForceBias(
        bulk_small, min_delta=0.05, max_delta=0.1, seed=42, temperature=5000
    )

    with pytest.warns(UserWarning, match="No committee forces available"):
        afbmc.run(1)

    afbmc = AdaptiveForceBias(
        bulk_small,
        min_delta=0.05,
        max_delta=0.1,
        seed=42,
        temperature=5000,
        scheme="energy",
    )

    with pytest.warns(UserWarning, match="No committee energies available"):
        afbmc.run(1)
