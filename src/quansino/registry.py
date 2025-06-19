"""
Module for Class registry serialization/deserialization of simulation objects.

This module essentially provides a way to register classes with a name, retrieve them by name. This is useful for serialization and deserialization of custom simulation objects, allowing them to be stored and retrieved without needing to know their exact class at runtime. Users can register their custom classes in their code, which will then be available globally for serialization and deserialization purposes via their `__class__.__name__` attribute.

Users should avoid using the `__class_registry` global variable directly, and instead use the provided functions to register and retrieve classes. This ensures that the registry is used correctly and avoids potential issues with bad scoping or name collisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


__class_registry = {}


def register_class(cls: type, class_name: str | None = None) -> type:
    """
    Register a class with a name directly (non-decorator version).

    Parameters
    ----------
    cls : type
        The class to register.
    class_name : str | None, optional
        The name to register the class with, by default None. If None, `cls.__name__` is used.

    Returns
    -------
    type
        The registered class.
    """
    if class_name is None:
        class_name = cls.__name__

    __class_registry[class_name] = cls

    return cls


def register(class_name: str | None = None) -> Callable[[type], type]:
    """
    Decorator to register a class with a name.

    Parameters
    ----------
    class_name : str | None, optional
        The name to register the class with, by default None. If None, `cls.__name__` is used.

    Returns
    -------
    Callable[[type], type]
        A decorator that registers the class with the given name.
    """

    def decorator(cls: type) -> type:
        """
        Decorator function to register a class with a name
        when the class is defined.

        Parameters
        ----------
        cls : type
            The class to register.

        Returns
        -------
        type
            The registered class.
        """
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
    type
        The class registered with the given name.

    Raises
    ------
    KeyError
        If the class is not registered.
    """
    if class_name not in __class_registry:
        raise KeyError(
            f"Class `{class_name}` not registered in the global registry. Available classes: {list(__class_registry.keys())}"
        )

    return __class_registry[class_name]


def get_class_name(cls: type) -> str:
    """
    Get the registered name of a class.

    Parameters
    ----------
    cls : type
        The class to get the name of.

    Returns
    -------
    str
        The registered name of the class, or None if not found.

    Raises
    ------
    KeyError
        If the class is not registered.
    """
    for name, registered_cls in __class_registry.items():
        if registered_cls is cls:
            return name

    raise KeyError(
        f"Class `{cls.__name__}` not found in the global registry. Available classes: {list(__class_registry.values())}"
    )


def get_typed_class(name: str, expected_base: type) -> type:
    """
    Get a class by its registered name with runtime type checking.

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
    TypeError
        If the registered class is not a subclass of expected_base.
    """
    cls = get_class(name)

    if not issubclass(cls, expected_base):
        raise TypeError(
            f"Class `{name}` is not a {expected_base.__name__} subclass. Got {cls.__name__} instead."
        )

    return cls
