from __future__ import annotations

import numpy as np
import pytest

from quansino.mc.contexts import DisplacementContext
from quansino.operations import Ball, BaseOperation, Box, Sphere, Translation


def test_composite_operations(single_atom, rng):
    context = DisplacementContext(single_atom, rng)

    sphere = Sphere(0.1)
    box = Box(0.1)
    ball = Ball(0.1)

    composite_operation = sphere + box

    assert len(composite_operation) == 2

    assert composite_operation[0] == sphere
    assert composite_operation[1] == box

    assert composite_operation.calculate(context).shape == (1, 3)
    assert (np.linalg.norm(composite_operation.calculate(context)) > 0.0).all()

    composite_operation_2 = composite_operation + ball
    composite_operation_3 = ball + composite_operation

    assert len(composite_operation_2) == 3
    assert len(composite_operation_3) == 3

    assert composite_operation_2[0] == sphere
    assert composite_operation_2[1] == box
    assert composite_operation_2[2] == ball

    assert composite_operation_3[0] == ball
    assert composite_operation_3[1] == sphere
    assert composite_operation_3[2] == box

    composite_operation_4 = composite_operation + composite_operation_2

    assert len(composite_operation_4) == 5
    assert composite_operation_4[0] == sphere
    assert composite_operation_4[1] == box
    assert composite_operation_4[2] == sphere
    assert composite_operation_4[3] == box
    assert composite_operation_4[4] == ball

    composite_operation_5 = sphere * 5

    assert len(composite_operation_5) == 5

    for move in composite_operation_5:
        assert move == sphere

    assert composite_operation_5.calculate(context).shape == (1, 3)
    assert (np.linalg.norm(composite_operation_5.calculate(context)) > 0.0).all()

    with pytest.raises(ValueError):
        sphere * -2  # type: ignore

    with pytest.raises(ValueError):
        sphere * 4.2  # type: ignore

    with pytest.raises(ValueError):
        composite_operation_2 * 0  # type: ignore

    with pytest.raises(ValueError):
        composite_operation_2 * -1  # type: ignore

    with pytest.raises(ValueError):
        composite_operation_2 * 4.2  # type: ignore

    composite_operation_6 = composite_operation_2 * 2

    assert len(composite_operation_6) == 6

    for move in composite_operation_6:
        assert isinstance(move, BaseOperation)


def test_composite_operations_to_dict():
    sphere = Sphere(0.1)
    box = Box(0.1)
    translation = Translation()

    composite_operation = sphere + box + translation

    data = composite_operation.to_dict()

    assert data == {
        "name": "CompositeOperation",
        "kwargs": {
            "operations": [
                {"name": "Sphere", "kwargs": {"step_size": 0.1}},
                {"name": "Box", "kwargs": {"step_size": 0.1}},
                {"name": "Translation"},
            ]
        },
    }

    new_composite_operation = composite_operation.from_dict(data)

    assert isinstance(new_composite_operation, type(composite_operation))
    assert len(new_composite_operation) == len(composite_operation)
    assert data == new_composite_operation.to_dict()
