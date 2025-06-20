from __future__ import annotations

import numpy as np
from ase.calculators.calculator import compare_atoms
from ase.units import bar
from numpy.testing import assert_allclose, assert_array_equal

from quansino.mc.contexts import DeformationContext
from quansino.mc.criteria import BaseCriteria
from quansino.mc.isobaric import Isobaric
from quansino.moves import CellMove, DisplacementMove
from quansino.operations.cell import IsotropicDeformation
from quansino.operations.displacement import Ball


def test_isobaric(bulk_small, rng, tmp_path):
    """Test the `Isobaric` class."""
    mc = Isobaric(
        bulk_small,
        temperature=0.1,
        pressure=1.0 * bar,
        max_cycles=10,
        default_displacement_move=DisplacementMove([0, 1, 2, 3], Ball(0.1), rng),
        default_cell_move=CellMove(IsotropicDeformation(0.05), rng),
        logfile=tmp_path / "mc.log",
        trajectory=tmp_path / "mc.traj",
    )

    class DummyCriteria(BaseCriteria):
        def evaluate(self, context: DeformationContext) -> bool:
            return context.rng.random() < 0.5

    mc.moves["default_displacement_move"].criteria = DummyCriteria()
    mc.moves["default_cell_move"].criteria = DummyCriteria()

    assert mc.atoms == bulk_small
    assert mc.temperature == 0.1
    assert mc.pressure == 1.0 * bar
    assert mc.max_cycles == 10

    assert isinstance(mc.context, DeformationContext)

    assert isinstance(mc.moves["default_displacement_move"].move, DisplacementMove)
    assert isinstance(mc.moves["default_cell_move"].move, CellMove)

    assert isinstance(mc.moves["default_displacement_move"].move.operation, Ball)
    assert isinstance(
        mc.moves["default_cell_move"].move.operation, IsotropicDeformation
    )

    assert mc.moves["default_cell_move"].probability == 1 / (len(bulk_small) + 1)
    assert mc.moves["default_displacement_move"].probability == 1 / (
        1 + 1 / len(bulk_small)
    )

    assert_allclose(mc.context.last_cell, bulk_small.cell)
    assert_allclose(mc.context.last_positions, bulk_small.get_positions())
    assert np.isnan(mc.context.last_energy)
    assert mc.last_results == {}

    old_cell = mc.atoms.cell.copy()
    old_positions = mc.atoms.get_positions().copy()

    for _ in mc.step():
        pass

    assert mc.atoms.calc is not None

    accepted_moves = [move[0] for move in mc.move_history if move[1]]

    if "default_cell_move" in accepted_moves:
        assert not np.allclose(mc.context.last_cell, old_cell)
        assert not np.allclose(mc.atoms.cell, old_cell)
        assert not np.allclose(mc.context.last_positions, old_positions)
        assert not np.allclose(mc.atoms.get_positions(), old_positions)
    else:
        assert_allclose(mc.context.last_cell, old_cell)
        assert_allclose(mc.atoms.cell, old_cell)

    assert_allclose(mc.context.last_positions, mc.atoms.get_positions())
    assert_allclose(mc.context.last_cell, mc.atoms.cell)

    acceptances = []

    for step in mc.irun(100):
        for move_name in step:
            assert move_name in mc.moves
            assert mc.atoms.calc is not None

            if mc.move_history and mc.move_history[-1][1]:
                assert_allclose(mc.context.last_positions, mc.atoms.get_positions())
                assert_allclose(mc.context.last_cell, mc.atoms.cell)
            else:
                assert mc.atoms.calc.results.keys() == mc.last_results.keys()
                assert all(
                    np.allclose(mc.last_results[k], mc.atoms.calc.results[k])
                    for k in mc.last_results
                    if isinstance(mc.last_results[k], str | float | int)
                )

            assert not compare_atoms(mc.atoms.calc.atoms, mc.atoms)
            assert_allclose(mc.context.last_positions, mc.atoms.get_positions())

        acceptances.append(mc.acceptance_rate)

    acceptance_from_log = np.loadtxt(tmp_path / "mc.log", skiprows=1, usecols=-1)

    assert_array_equal(acceptances, acceptance_from_log[1:])
    assert_allclose(np.sum(acceptances), 50, atol=20)


def test_isobaric_simulation(bulk_small, rng):
    """Test the `Isobaric` class with a simulation."""
    mc = Isobaric(
        bulk_small,
        temperature=300.0,
        pressure=1.0 * bar,
        max_cycles=10,
        default_displacement_move=None,
        default_cell_move=CellMove(IsotropicDeformation(0.05)),
    )

    last_cell = mc.atoms.cell.copy()

    for step in mc.irun(10):
        for move_name in step:
            assert_allclose(mc.context.last_cell, mc.atoms.cell)
            assert_allclose(mc.context.last_positions, mc.atoms.get_positions())
            assert mc.moves[move_name].probability == 0.2

        any_accepted = any(history[1] for history in mc.move_history)

        if any_accepted:
            assert not np.allclose(mc.atoms.cell, last_cell)
            last_cell = mc.atoms.cell.copy()
        else:
            assert_allclose(mc.atoms.cell, last_cell)
