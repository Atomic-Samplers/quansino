"""Module for Class registry serialization/deserialization of simulation objects."""

from __future__ import annotations

_class_registry = {}


def register_class(cls: type, class_name: str | None = None) -> type:
    """
    Register a class with a name directly (non-decorator version).

    Parameters
    ----------
    cls : type
        The class to register.
    class_name : str, optional
        The name to register the class with. If None, the class name is used.

    Returns
    -------
    type
        The registered class (for method chaining).
    """
    if class_name is None:
        class_name = cls.__name__

    _class_registry[class_name] = cls

    return cls


def register(class_name=None):
    """
    Decorator to register a class with a name.

    Parameters
    ----------
    class_name : str, optional
        The name to register the class with. If None, the class name is used.
    """

    def decorator(cls):
        return register_class(cls, class_name)

    return decorator


def get_class(class_name: str) -> type:
    """
    Get a class by its registered name.

    Parameters
    ----------
    class_name : str
        The name of the class to get.

    Returns
    -------
    class
        The class registered with the given name.

    Raises
    ------
    KeyError
        If the class is not registered.
    """
    if class_name not in _class_registry:
        raise KeyError(
            f"Class `{class_name}` not registered in the global registry. Available classes: {list(_class_registry.keys())}"
        )

    return _class_registry[class_name]


def get_class_name(cls: type) -> str | None:
    """
    Get the registered name of a class.

    Parameters
    ----------
    cls : type
        The class to get the name of.

    Returns
    -------
    str | None
        The registered name of the class, or None if not found.
    """
    for name, registered_cls in _class_registry.items():
        if registered_cls is cls:
            return name

    return None


def get_typed_class(name: str, expected_base: type) -> type:
    """
    Get a class by its registered name with type checking.

    Parameters
    ----------
    name : str
        The name of the class to get.
    expected_base : type
        The base class that the returned class should inherit from.

    Returns
    -------
    type
        The class registered with the given name.

    Raises
    ------
    KeyError
        If the class is not registered.
    TypeError
        If the registered class is not a subclass of expected_base.
    """
    cls = get_class(name)

    if not issubclass(cls, expected_base):
        raise TypeError(
            f"Class `{name}` is not a {expected_base.__name__} subclass. Got {cls.__name__} instead."
        )

    return cls
