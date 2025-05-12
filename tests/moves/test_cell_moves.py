from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from quansino.mc.contexts import StrainContext
from quansino.moves.cell import CellMove
from quansino.operations.cell import IsotropicStretch


def test_cell_move(bulk_small, rng):
    move = CellMove(IsotropicStretch(0.1))

    context = StrainContext(bulk_small, rng=rng)
    move.context = context

    old_cell = context.atoms.cell.copy()
    old_positions = context.atoms.positions.copy()

    assert move.attempt_move()

    assert move.context.atoms.cell.shape == (3, 3)

    assert np.all(np.diag(move.context.atoms.cell) != np.diag(old_cell))
    assert np.any(move.context.atoms.positions != old_positions)

    old_positions = move.context.atoms.positions.copy()

    move.scale_atoms = False

    assert move.attempt_move()

    assert_allclose(move.context.atoms.positions, old_positions)

    move.scale_atoms = True
    move.check_move = lambda: False

    old_cell = move.context.atoms.cell.copy()

    assert not move.attempt_move()

    assert_allclose(move.context.atoms.cell, old_cell)
    assert_allclose(move.context.atoms.positions, old_positions)
