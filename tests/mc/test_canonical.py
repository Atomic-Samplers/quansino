from __future__ import annotations

import numpy as np
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from tests.conftest import DummyCalculator

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import DisplacementContext
from quansino.mc.criteria import BaseCriteria, CanonicalCriteria
from quansino.moves.displacement import DisplacementMove
from quansino.operations.displacement import Ball
from quansino.utils.moves import MoveStorage


def test_canonical(bulk_small, tmp_path):
    """Test the `Canonical` class."""
    move = DisplacementMove[Ball, DisplacementContext](
        np.arange(len(bulk_small)), Ball(1.0)
    )

    assert move.operation.step_size == 1.0

    mc = Canonical[DisplacementMove, BaseCriteria](
        bulk_small,
        temperature=10.0,
        default_displacement_move=move,
        logfile=tmp_path / "mc.log",
    )

    bulk_small.calc = EMT()

    assert mc.temperature == 10.0
    assert mc.max_cycles == len(bulk_small)
    assert mc.atoms == bulk_small
    assert mc.last_results == {}

    assert_allclose(mc.context.last_positions, bulk_small.get_positions())

    data = mc.to_dict()

    assert data["name"] == "Canonical"
    assert data["attributes"]["step_count"] == 0
    assert data["context"]["temperature"] == 10.0
    assert data["kwargs"]["seed"] == mc._MonteCarlo__seed  # type: ignore
    assert data["rng_state"] == mc._rng.bit_generator.state
    assert (
        data["moves"]["default_displacement_move"]["kwargs"]["move"]["name"]
        == "DisplacementMove"
    )

    assert isinstance(mc.moves["default_displacement_move"].move.operation, Ball)
    assert isinstance(mc.moves["default_displacement_move"].criteria, CanonicalCriteria)

    mc.context.last_energy = 0.0

    assert mc.moves["default_displacement_move"].probability == 1.0
    assert mc.moves["default_displacement_move"].interval == 1
    assert mc.moves["default_displacement_move"].minimum_count == 0
    assert_equal(
        mc.moves["default_displacement_move"].move.labels, np.arange(len(bulk_small))
    )
    assert mc.moves["default_displacement_move"].move.max_attempts == 10000

    energy = mc.atoms.get_potential_energy()

    mc.validate_simulation()

    for _ in mc.step():
        pass

    if mc.acceptance_rate:
        assert not np.allclose(mc.last_results["energy"], energy)
    else:
        assert_allclose(mc.last_results["energy"], energy)

    mc.run(10)

    assert mc.step_count == 10

    mc.temperature = 300.0
    mc.max_cycles = 1

    class DummyCriteria(BaseCriteria):
        def evaluate(self, context: DisplacementContext) -> bool:
            return context.rng.random() < 0.5

    mc.moves["default_displacement_move"].criteria = DummyCriteria()

    assert mc.default_logger is not None

    mc.default_logger.file.seek(0)
    mc.default_logger.file.truncate()

    acceptances = []
    for _ in mc.srun(1000):
        assert mc.atoms.calc is not None
        assert not compare_atoms(mc.atoms.calc.atoms, mc.atoms)

        if mc.acceptance_rate:
            pass
        else:
            assert mc.atoms.calc.results.keys() == mc.last_results.keys()
            assert all(
                np.allclose(mc.last_results[k], mc.atoms.calc.results[k])
                for k in mc.last_results
            )

        assert_allclose(mc.context.last_positions, mc.atoms.get_positions())
        acceptances.append(mc.acceptance_rate)

    acceptance_from_log = np.loadtxt(tmp_path / "mc.log", skiprows=0, usecols=-1)

    assert_array_equal(acceptances, acceptance_from_log)
    assert_allclose(np.sum(acceptances), 500, atol=100)

    move_storage = MoveStorage(
        move=move,
        interval=4,
        probability=0.4,
        minimum_count=0,
        criteria=DummyCriteria(),
    )

    mc = Canonical[DisplacementMove, DummyCriteria](
        bulk_small, temperature=0.1, seed=42
    )

    mc.moves["default_displacement_move"] = move_storage

    assert mc.moves["default_displacement_move"].move == move
    assert mc.moves["default_displacement_move"].interval == 4
    assert mc.moves["default_displacement_move"].probability == 0.4
    assert mc.moves["default_displacement_move"].minimum_count == 0
    assert mc.moves["default_displacement_move"].move.operation.step_size == 1.0

    mc.atoms.calc = DummyCalculator()

    mc.revert_state()
    mc.save_state()


def test_canonical_restart(bulk_small, tmp_path):
    """Test the `Canonical` class with restart functionality."""
    move = DisplacementMove(np.arange(len(bulk_small)), Ball(0.1))

    mc = Canonical(
        bulk_small,
        temperature=2000.0,
        default_displacement_move=move,
        logfile=tmp_path / "mc.log",
        trajectory=tmp_path / "mc.xyz",
        restart_file=tmp_path / "mc_restart.json",
        logging_interval=5,
    )

    mc.context.last_energy = 0.0

    data = mc.to_dict()

    reconstructed_mc = Canonical.from_dict(data)
    reconstructed_mc.atoms.calc = bulk_small.calc  # calc are not serialized...

    assert reconstructed_mc.atoms == mc.atoms
    assert reconstructed_mc.temperature == mc.temperature
    assert reconstructed_mc.max_cycles == mc.max_cycles
    assert reconstructed_mc.step_count == mc.step_count

    assert reconstructed_mc.last_results == mc.last_results
    assert reconstructed_mc.context.last_energy == mc.context.last_energy
    assert_allclose(reconstructed_mc.context.last_positions, mc.context.last_positions)
    assert reconstructed_mc.moves.keys() == mc.moves.keys()
    assert (
        reconstructed_mc.moves["default_displacement_move"].interval
        == mc.moves["default_displacement_move"].interval
    )
    assert (
        reconstructed_mc.moves["default_displacement_move"].probability
        == mc.moves["default_displacement_move"].probability
    )
    assert (
        reconstructed_mc.moves["default_displacement_move"].minimum_count
        == mc.moves["default_displacement_move"].minimum_count
    )
    assert (
        reconstructed_mc.moves["default_displacement_move"].move.operation.step_size
        == mc.moves["default_displacement_move"].move.operation.step_size
    )
    assert_allclose(
        reconstructed_mc.moves["default_displacement_move"].move.labels,
        mc.moves["default_displacement_move"].move.labels,
    )
    assert (
        reconstructed_mc.moves["default_displacement_move"].move.max_attempts
        == mc.moves["default_displacement_move"].move.max_attempts
    )
    assert isinstance(reconstructed_mc.context, DisplacementContext)
    assert isinstance(
        reconstructed_mc.moves["default_displacement_move"].criteria, CanonicalCriteria
    )
    assert isinstance(
        reconstructed_mc.moves["default_displacement_move"].move.operation, Ball
    )

    assert mc.default_logger is not None
    assert reconstructed_mc.default_logger is None

    assert mc.default_trajectory is not None
    assert reconstructed_mc.default_trajectory is None

    assert mc.default_restart is not None
    assert reconstructed_mc.default_restart is None

    assert reconstructed_mc._MonteCarlo__seed == mc._MonteCarlo__seed  # type: ignore
    assert reconstructed_mc._rng.bit_generator.state == mc._rng.bit_generator.state
    assert reconstructed_mc._rng.random() == mc._rng.random()
    assert reconstructed_mc._rng.integers(0, 100) == mc._rng.integers(0, 100)

    energies = []

    mc.context.last_energy = np.nan
    reconstructed_mc.context.last_energy = np.nan

    energies = [mc.context.last_energy for _ in mc.irun(20)]

    energies_reconstructed = [
        reconstructed_mc.context.last_energy for _ in reconstructed_mc.irun(20)
    ]

    assert_allclose(energies, energies_reconstructed)

    energies = []

    external_rng = default_rng(42)

    for _ in mc.srun(20):
        energies.append(mc.context.last_energy)

    energies_reconstructed = []

    i = 0
    current_mc = reconstructed_mc
    while i < 20:
        for _ in range(20 - i):
            for _ in current_mc.step():
                ...
            i += 1
            energies_reconstructed.append(current_mc.context.last_energy)

            if external_rng.random() < 0.5:
                break

        current_mc = Canonical.from_dict(current_mc.to_dict())
        current_mc.atoms.calc = bulk_small.calc

    assert_allclose(energies, energies_reconstructed)

    mc_reconstructed = Canonical.from_dict(
        mc.to_dict(),
        logfile=tmp_path / "mc.log",
        trajectory=tmp_path / "mc.xyz",
        restart_file=tmp_path / "mc_restart.json",
        logging_interval=6,
        max_cycles=10,
    )

    assert str(mc_reconstructed.default_logger) == str(mc.default_logger)
    assert str(mc_reconstructed.default_trajectory) == str(mc.default_trajectory)
    assert str(mc_reconstructed.default_restart) == str(mc.default_restart)
    assert mc_reconstructed.logging_interval == 6
    assert mc_reconstructed.max_cycles == 10
