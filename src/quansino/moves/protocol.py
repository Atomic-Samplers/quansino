from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from quansino.mc.contexts import Context


class BaseProtocol(Protocol):
    """
    Base protocol for all moves.
    """

    context: Context

    def attach_simulation(self, context: Context) -> None:
        """
        Attach the simulation context to the move. This method must be called before the move is used, and should be used to set the context attribute. This should be done by the Monte Carlo classes.

        Parameters
        ----------
        context : Context
            The simulation context to attach to the move.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the move to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the move.
        """
        ...


class DisplacementProtocol(BaseProtocol, Protocol):
    """
    Protocol for displacement moves, typically used for moves that displace atoms.
    """

    def __call__(self) -> bool:
        """
        Perform the move.

        Returns
        -------
        bool
            Whether the move was successful.
        """
        ...


class CellProtocol(BaseProtocol, Protocol):
    """
    Protocol for non-updatable moves, typically used for moves that do not require updating the simulation context such as cell moves.
    """

    is_updatable: Literal[False] = False

    def __call__(self) -> bool:
        """
        Perform the move.

        Returns
        -------
        bool
            Whether the move was successful.
        """
        ...


class ExchangeProtocol(BaseProtocol, Protocol):
    """
    Protocol for updatable moves, to be used in Grand Canonical Monte Carlo simulations where the number of particles is not fixed.
    """

    is_updatable: Literal[True] = True

    def __call__(self) -> bool:
        """
        Perform the move.

        Returns
        -------
        bool
            Whether the move was successful.
        """
        ...

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
