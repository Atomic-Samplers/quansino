from __future__ import annotations

import pytest

from quansino.registry import (
    get_class,
    get_class_name,
    get_typed_class,
    register,
    register_class,
)


class TestClass:
    """
    A test class for demonstrating the registry functionality.
    """


class NonExistingClass:
    """
    A non-existing class for testing error handling in the registry.
    """


@register()
class TestTypedClass:
    """
    A test class for demonstrating the registry functionality with type checking.
    """


def test_registry():
    """Test the registry functionality by registering a class and retrieving it."""
    registered_class = register_class(TestClass)

    assert get_class_name(registered_class) == "TestClass"
    assert get_class("TestClass") is registered_class

    with pytest.raises(TypeError):
        get_typed_class("TestClass", int)

    get_typed_class("TestTypedClass", TestTypedClass)

    with pytest.raises(KeyError):
        get_class("NonExistingClass")

    with pytest.raises(KeyError):
        get_class_name(NonExistingClass)
