from __future__ import annotations

import numpy as np
from ase.units import bar
from numpy.testing import assert_allclose

from quansino.mc.contexts import DeformationContext
from quansino.mc.criteria import IsotensionCriteria
from quansino.mc.isotension import Isotension
from quansino.moves import CellMove, DisplacementMove
from quansino.operations.cell import AnisotropicDeformation, IsotropicDeformation
from quansino.operations.displacement import Ball


def test_isotension(bulk_small):
    """Test the `Isobaric` class."""
    mc = Isotension(
        bulk_small,
        temperature=0.1,
        pressure=1.0 * bar,
        max_cycles=10,
        default_displacement_move=DisplacementMove([0, 1, 2, 3], Ball(0.1)),
        default_cell_move=CellMove(IsotropicDeformation(0.05)),
    )

    assert mc.atoms == bulk_small
    assert mc.temperature == 0.1
    assert mc.pressure == 1.0 * bar
    assert mc.max_cycles == 10

    assert isinstance(mc.context, DeformationContext)

    assert isinstance(mc.moves["default_displacement_move"].move, DisplacementMove)
    assert isinstance(mc.moves["default_cell_move"].move, CellMove)

    assert isinstance(mc.moves["default_cell_move"].criteria, IsotensionCriteria)


def test_isotension_simulation(bulk_small):
    """Test the `Isobaric` class with a simulation."""
    mc = Isotension(
        bulk_small,
        temperature=300.0,
        pressure=1.0 * bar,
        external_stress=np.array([[1000000, 0, 0], [0, -1000000, 0], [0, 0, 1000000]])
        * bar,
        max_cycles=10,
        default_displacement_move=None,
        default_cell_move=CellMove(AnisotropicDeformation(0.025)),
    )

    last_cell = mc.atoms.cell.copy()
    first_cell = mc.atoms.cell.copy()

    for step in mc.irun(50):
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

    assert not np.allclose(mc.atoms.cell, first_cell)
    assert mc.atoms.cell.array[0, 0] < first_cell.array[0, 0]
    assert mc.atoms.cell.array[1, 1] > first_cell.array[1, 1]
    assert mc.atoms.cell.array[2, 2] < first_cell.array[2, 2]


def test_isotension_simulation_with_mask(bulk_small):
    """Test the `Isobaric` class with a simulation."""
    mc = Isotension(
        bulk_small,
        temperature=300.0,
        pressure=1.0 * bar,
        external_stress=np.array([[1000000, 0, 0], [0, -1000000, 0], [0, 0, 1000000]])
        * bar,
        max_cycles=10,
        default_displacement_move=None,
        default_cell_move=CellMove(
            AnisotropicDeformation(
                0.025,
                mask=np.array(
                    [[True, False, False], [False, True, False], [False, False, True]]
                ),
            )
        ),
    )

    last_cell = mc.atoms.cell.copy()
    first_cell = mc.atoms.cell.copy()

    for step in mc.irun(50):
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

    assert not np.allclose(mc.atoms.cell, first_cell)
    assert mc.atoms.cell.array[0, 0] < first_cell.array[0, 0]
    assert mc.atoms.cell.array[1, 1] > first_cell.array[1, 1]
    assert mc.atoms.cell.array[2, 2] < first_cell.array[2, 2]
