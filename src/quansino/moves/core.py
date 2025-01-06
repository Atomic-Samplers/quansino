"""Module for Base Move class"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quansino.mc.contexts import Context
    from quansino.moves.operations import Operation


class BaseMove:
    """
    Helper Class to build Monte Carlo moves

    Parameters
    ----------
    move_operator: Operation
        The operation to perform in the move. The object must have a `calculate` method that takes a context as input.
    apply_constraints: bool, optional
        Whether to apply constraints to the move, by default True.

    Attributes
    ----------
    move_operator: Operation
        The operation to perform in the move. The object must have a `calculate` method that takes a context as input.
    apply_constraints: bool
        Whether to apply constraints to the move.

    Notes
    -----
    This class is a base class for all Monte Carlo moves, and should not be used directly. The __call__ method should be implemented in the subclass, performing the actual move and returning a boolean indicating whether the move was accepted.
    """

    def __init__(self, move_operator: Operation, apply_constraints=True):
        """Initialize the BaseMove object."""
        self.move_operator = move_operator
        self.apply_constraints = apply_constraints

    def __call__(self, *args, **kwds) -> bool: ...

    def attach_simulation(self, context: Context) -> None:
        """
        Attach the simulation context to the move. This method must be called before the move is used, and should be used to set the context attribute. This must be done by the Monte Carlo classes.

        Parameters
        ----------
        context: Context
            The simulation context to attach to the move.

        Notes
        -----
        The context object aim to provide the necessary information for the move to perform its operation, without having to pass whole objects around. Classes inheriting from BaseMove should define a `required_context` attribute that specifies the context class that the move requires.
        """
        self.context = context

    def check_move(self) -> bool:
        """Check if the move is accepted. This method should be implemented in the subclass, and should return a boolean indicating whether the move was accepted."""
        return True
