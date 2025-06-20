from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from ase.cell import Cell

    from quansino.mc.contexts import Context
    from quansino.type_hints import IntegerArray


@runtime_checkable
class Serializable(Protocol):
    """
    Protocol for objects that can be serialized to and from a dictionary.
    """

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
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create an object from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the object.

        Returns
        -------
        Self
            The object created from the dictionary.
        """
        ...


@runtime_checkable
class Criteria[ContextType: Context](Serializable, Protocol):
    """
    Base protocol for all criteria in [`MonteCarlo`][quansino.mc.core.MonteCarlo] simulations. Criteria are used to evaluate whether a move is acceptable based on the current state of the system. Criteria classes should implement the [`evaluate`][quansino.protocols.Criteria.evaluate] method to perform the evaluation and return a boolean indicating whether the criteria are met. Such method take a simulation [`Context`][quansino.mc.contexts.Context] as parameter, which provides the necessary information about the current state of the system, such as atom positions, cell parameters, as well as any additional information needed to perform the evaluation.
    """

    def evaluate(self, context: ContextType) -> bool:
        """
        Calculate the operation to perform based on the given context.

        Parameters
        ----------
        context : ContextType
            The context to use when calculating the operation.

        Returns
        -------
        bool
            Whether the criteria are met for the current context.
        """
        ...


@runtime_checkable
class Move[ContextType: Context](Serializable, Protocol):
    """
    Base protocol for all acceptable moves in [`MonteCarlo`][quansino.mc.core.MonteCarlo] simulations. Moves perform specific tasks such as displacing atoms, deforming cells, or exchanging atoms. Move classes should implement the [`__call__`][quansino.protocols.Move.__call__] method to perform the move and return whether it was successful. Such method take a simulation [`Context`][quansino.mc.contexts.Context] as parameter, which provides the necessary information about the current state of the system, such as atom positions, cell parameters, as well as any additional information needed to perform the move.
    """

    def __call__(self, context: ContextType) -> bool:
        """
        Perform the move, i.e. displace, deform cells, or exchange atoms, in place. This method should be implemented in user defined classes, and should return a boolean indicating whether the attempted operation was successful. Users are free to implement their own contraints and logic for what constitutes a successful move. Criteria for success can include whether the move results in a valid configuration, and should typically involve geometric (regions, distances between atoms, ...) constraints. Energy constraints should not be used here, as they are typically handled by [`Criteria`][quansino.protocols.Criteria] classes.

        Returns
        -------
        bool
            Whether the move was successful.
        """
        ...

    def on_atoms_changed(
        self, added_indices: IntegerArray, removed_indices: IntegerArray
    ) -> None:
        """
        Function to run when atoms are added or removed from the simulation. This method should be implemented in user defined move classes to update the move's internal state based on the changes in atoms.

        Parameters
        ----------
        added_indices : IntegerArray
            The indices of the atoms that were added.
        removed_indices : IntegerArray
            The indices of the atoms that were removed.
        """
        ...

    def on_cell_changed(self, new_cell: Cell) -> None:
        """
        Function to run when the cell changes. This method should be implemented in user defined move classes to update the move's internal state based on the new cell.

        Parameters
        ----------
        new_cell : Cell
            The new cell.
        """
        ...


@runtime_checkable
class Operation[ContextType: Context](Serializable, Protocol):
    """
    Base protocol for all operations in [`MonteCarlo`][quansino.mc.core.MonteCarlo] simulations. Operations are used by [`Move`][quansino.protocols.Move] to calculate various operations, such as calculating displacements or deformation tensors. Operation classes should implement the [`calculate`][quansino.protocols.Operation.calculate] method to perform the operation and return the result. Such method take a simulation [`Context`][quansino.mc.contexts.Context] as parameter, which provides the necessary information about the current state of the system, such as atom positions, cell parameters, as well as any additional information needed to perform the operation.
    """

    def calculate(self, context: ContextType) -> Any:
        """
        Calculate the operation to perform based on the given context.

        Parameters
        ----------
        context : Context
            The context to use when calculating the operation.

        Returns
        -------
        Any
            The result of the calculation, typically a displacement or strain tensor.
        """
        ...
