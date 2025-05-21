from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    from quansino.mc.contexts import (
        Context,
        DisplacementContext,
        ExchangeContext,
        StrainContext,
    )


@runtime_checkable
class BaseProtocol[ContextType: Context](Protocol):
    """
    Base protocol for all moves.
    """

    def attach_simulation(self, context: ContextType) -> None:
        """
        Attach the simulation context to the move. This method must be called before the move is used, and should be used to set the context attribute. This should be done by the Monte Carlo classes.

        Parameters
        ----------
        context : Context
            The simulation context to attach to the move.
        """
        ...

    def __call__(self) -> bool:
        """
        Perform the move.

        Returns
        -------
        bool
            Whether the move was successful.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the object.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """
        Create an object from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the object.

        Returns
        -------
        SerializableProtocol
            The object created from the dictionary.
        """
        ...


class DisplacementProtocol[ContextType: DisplacementContext](BaseProtocol, Protocol):
    """
    Protocol for displacement moves, typically used for moves that displace atoms.
    """


class CellProtocol[ContextType: StrainContext](BaseProtocol, Protocol):
    """
    Protocol for non-updatable moves, typically used for moves that do not require updating the simulation context such as cell moves.
    """

    is_updatable: Final = False


class ExchangeProtocol[ContextType: ExchangeContext](BaseProtocol, Protocol):
    """
    Protocol for updatable moves, to be used in Grand Canonical Monte Carlo simulations where the number of particles is not fixed.
    """

    is_updatable: Final = True

    def update(self, *args, **kwargs) -> None:
        """
        Update the move parameters based on the provided arguments.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        ...
