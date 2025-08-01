from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from quansino.mc.contexts import Context


class BaseIntegrator:
    """
    Abstract base class for integrators in Monte Carlo simulations.

    This class defines the interface for all integrators that can be used to
    perform operations on a simulation context. Integrators can be used to perform dynamical
    operations such as integration of equations of motion or other time-dependent processes.
    """

    __slots__ = ()

    def integrate(self, context: Context, *args: Any, **kwargs: Any) -> None:
        """
        Integrate the equations of motion based on the given context.

        Parameters
        ----------
        context : Context
            The context to use when calculating the integrator.
        *args: Any
            Additional positional arguments for the integrator.
        **kwargs: Any
            Additional keyword arguments for the integrator.
        """

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `BaseIntegrator` object to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            The dictionary representation of the `BaseIntegrator` object.
        """
        return {"name": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create an `BaseIntegrator` object from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the `BaseIntegrator` object.

        Returns
        -------
        Self
            The `BaseIntegrator` object created from the dictionary.
        """
        kwargs = data.get("kwargs", {})
        instance = cls(**kwargs)

        for key, value in data.get("attributes", {}).items():
            setattr(instance, key, value)

        return instance
