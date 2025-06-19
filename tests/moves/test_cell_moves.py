from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from quansino.mc.contexts import DeformationContext
from quansino.moves.cell import CellMove
from quansino.operations.cell import IsotropicDeformation


def test_cell_move(bulk_small, rng):
    """Test the `CellMove` class."""
    move = CellMove(IsotropicDeformation(0.1))

    context = DeformationContext(bulk_small, rng=rng)

    old_cell = context.atoms.cell.copy()
    old_positions = context.atoms.positions.copy()

    assert move.attempt_deformation(context)

    assert context.atoms.cell.array.shape == (3, 3)

    assert np.all(np.diag(context.atoms.cell) != np.diag(old_cell))
    assert np.any(context.atoms.positions != old_positions)

    old_positions = context.atoms.positions.copy()

    move.scale_atoms = False

    assert move.attempt_deformation(context)

    assert_allclose(context.atoms.positions, old_positions)

    move.scale_atoms = True
    move.check_move = lambda: False

    old_cell = context.atoms.cell.copy()

    assert not move.attempt_deformation(context)

    assert_allclose(context.atoms.cell, old_cell)
    assert_allclose(context.atoms.positions, old_positions)
