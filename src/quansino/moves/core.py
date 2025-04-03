"""Module for Base Move class"""

from __future__ import annotations

from quansino.mc.contexts import Context
from quansino.operations.core import Operation


class BaseMove[OperationType: Operation, ContextType: Context]:
    """
    Helper Class to build Monte Carlo moves

    Parameters
    ----------
    operation: OperationType
        The operation to perform in the move. The object must have a `calculate` method that takes a context as input.
    apply_constraints: bool, optional
        Whether to apply constraints to the move, by default True.

    Attributes
    ----------
    operation: OperationType
        The operation to perform in the move. The object must have a `calculate` method that takes a context as input.
    apply_constraints: bool
        Whether to apply constraints to the move.

    Notes
    -----
    This class is a base class for all Monte Carlo moves, and should not be used directly. The __call__ method should be implemented in the subclass, performing the actual move and returning a boolean indicating whether the move was accepted.
    """

    AcceptableContext = Context
    max_attempts: int = 10000

    def __init__(
        self, operation: OperationType, apply_constraints: bool = True
    ) -> None:
        """Initialize the BaseMove object."""
        self.operation = operation
        self.apply_constraints: bool = apply_constraints

    def attach_simulation(self, context: ContextType) -> None:
        """
        Attach the simulation context to the move. This method must be called before the move is used, and should be used to set the context attribute. This must be done by the Monte Carlo classes.

        Parameters
        ----------
        context: Context
            The simulation context to attach to the move.

        Notes
        -----
        The context object aim to provide the necessary information for the move to perform its operation, without having to pass whole objects around. Classes inheriting from BaseMove should define a `AcceptableContext` attribute that specifies the context class that the move requires.
        """
        self.context: ContextType = context

    def check_move(self) -> bool:
        """Check if the move is accepted. This method should be implemented in the subclass, and should return a boolean indicating whether the move was accepted."""
        return True
