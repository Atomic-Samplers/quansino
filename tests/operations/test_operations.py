from __future__ import annotations

from quansino.mc.contexts import Context
from quansino.operations import BaseOperation, CompositeOperation


def test_operation(bulk_small, rng):
    operation = BaseOperation()

    assert operation.calculate(Context(bulk_small, rng)) is None

    assert isinstance(operation * 2, CompositeOperation)
    assert isinstance(operation + operation, CompositeOperation)
    assert isinstance(operation + operation * 2, CompositeOperation)
    assert isinstance(2 * operation, CompositeOperation)

    data = operation.to_dict()

    assert data == {"name": "BaseOperation"}

    new_operation = BaseOperation.from_dict(data)
    assert isinstance(new_operation, BaseOperation)
