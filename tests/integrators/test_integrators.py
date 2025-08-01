from __future__ import annotations

from tests.conftest import DummyIntegrator

from quansino.integrators.core import BaseIntegrator


def test_base_integrator():
    """Test the `BaseIntegrator` class."""
    integrator = DummyIntegrator()

    integrator.custom_attr = "test_value"  # type: ignore

    assert isinstance(integrator, BaseIntegrator)
    assert integrator.to_dict() == {"name": "DummyIntegrator"}

    data = {
        "name": "DummyIntegrator",
        "kwargs": {},
        "attributes": {"custom_attr": "test_value"},
    }
    new_integrator = DummyIntegrator.from_dict(data)
    assert isinstance(new_integrator, DummyIntegrator)
    assert new_integrator.to_dict() == {"name": "DummyIntegrator"}
