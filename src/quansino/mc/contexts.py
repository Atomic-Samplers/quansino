"""Module for Monte Carlo contexts"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np
from ase.atoms import Atoms

from quansino.utils.atoms import reinsert_atoms

if TYPE_CHECKING:
    from numpy.random import Generator

    from quansino.mc.core import MoveStorage
    from quansino.moves.displacements import DisplacementMove
    from quansino.moves.exchange import ExchangeMove
    from quansino.type_hints import IntegerArray, Positions


class Context(ABC):
    """
    Base abstract class for Monte Carlo contexts. Context defines the interface between the simulation object, the moves and their criteria. The context object aim to provide the necessary information for the move to perform its operation, without having to pass whole objects around. Classes inheriting from [`Context`][quansino.mc.contexts.Context] should define a `save_state()` and `revert_state()` method that saves and reverts the state of the simulation after a move, respectively. Specific context might be required for different types of moves, for example, [`DisplacementContext`][quansino.mc.contexts.DisplacementContext] for displacement moves and [`ExchangeContext`][quansino.mc.contexts.ExchangeContext] for exchange moves. This class is not meant to be instantiated, and represent the bare minimum that a context object should implement.

    Attributes
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.

    Methods
    -------
    save_state()
        Save the current state of the context.
    revert_state()
        Revert to the previously saved state.
    """

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
        self.atoms: Atoms = atoms
        self.rng: Generator = rng

    @abstractmethod
    def save_state(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def revert_state(self, *args, **kwargs) -> None: ...


class DisplacementContext(Context):
    """
    Context for displacement moves i.e. moves that displace atoms.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.

    Attributes
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on. Used in the simulation and moves.
    rng : Generator
        The random number generator to use. Used in the criteria and moves.
    last_positions : Positions
        The positions of the atoms in the last saved state. Used in the simulation.
    last_results : dict[str, Any]
        The results of the ASE calculator in the last saved state. Used in the simulation.
    temperature : float
        The temperature of the simulation in Kelvin. Used in the criteria.
    moving_indices : IntegerArray
        Integer indices of atoms that are being displaced. Used in moves.

    Methods
    -------
    save_state()
        Save the current state of the context, confirming the last positions and results.
    revert_state()
        Revert to the previously saved state, discarding the last positions and results.
    reset()
        Reset the context by setting `moving_indices` to an empty list.
    """

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
        """
        Initialize the DisplacementContext object.

        Parameters
        ----------
        atoms : Atoms
            The atoms object to perform the simulation on.
        rng : Generator
            The random number generator to use.
        """
        super().__init__(atoms, rng)

        self.last_positions: Positions = self.atoms.get_positions()
        self.last_results: dict[str, Any] = {}

        self.temperature: float = 0.0

        self.reset()

    def save_state(self) -> None:
        """Save the current state of the context and update the last positions and results."""
        self.last_positions = self.atoms.get_positions()
        try:
            self.last_results = self.atoms.calc.results  # type: ignore
        except AttributeError:
            warn(
                "Atoms object does not have calculator results attached.", stacklevel=2
            )
            self.last_results = {}

    def revert_state(self) -> None:
        """Revert to the previously saved state and undo the last move."""
        self.atoms.positions = self.last_positions
        try:
            self.atoms.calc.atoms = self.atoms  # type: ignore
            self.atoms.calc.results = self.last_results  # type: ignore
        except AttributeError:
            warn("Atoms object does not have calculator attached.", stacklevel=2)

    def reset(self) -> None:
        """Reset the context by setting `moving_indices` to an empty list."""
        self.moving_indices: IntegerArray = []


class ExchangeContext(DisplacementContext):
    """
    Context for exchange moves i.e. moves that exchange atoms.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.
    moves : dict[str, MoveStorage[DisplacementMove | ExchangeMove, ExchangeContext]]
        Dictionary of displacement moves to update labels when atoms are added or removed.

    Attributes
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on. Used in the simulation and moves.
    rng : Generator
        The random number generator to use. Used in the criteria and moves.
    moves : dict[str, MoveStorage]
        Dictionary of displacement moves to update when atoms are added or removed. Used only in the context.
    chemical_potential : float
        The chemical potential of the system. Used in the criteria.
    number_of_particles : int
        The number of particles in the system. Used in the criteria.
    added_indices : IntegerArray
        Integer indices of atoms that were added in the last move. Used in the moves.
    added_atoms : Atoms
        Atoms that were added in the last move. Used in the criteria and moves.
    deleted_indices : IntegerArray
        Integer indices of atoms that were deleted in the last move. Used in the moves.
    deleted_atoms : Atoms
        Atoms that were deleted in the last move. Used in the criteria and moves.
    accessible_volume : float [Criteria]
        The accessible volume of the system. Used in the criteria.
    particle_delta : int
        The change in the number of particles in the system. Used in the criteria and moves.

    Methods
    -------
    save_state()
        Save the current state of the context and update move labels.
    revert_state()
        Revert to the previously saved state and undo the last move.
    reset()
        Reset the context by setting all attributes to their initial values.
    """

    def __init__(
        self,
        atoms: Atoms,
        rng: Generator,
        moves: dict[str, MoveStorage[DisplacementMove | ExchangeMove, ExchangeContext]],
    ) -> None:
        """
        Initialize the ExchangeContext object.

        Parameters
        ----------
        atoms : Atoms
            The atoms object to perform the simulation on.
        rng : Generator
            The random number generator to use.
        moves : dict[str, MoveStorage]
            Dictionary of displacement moves to update labels when atoms are added or removed.
        """
        super().__init__(atoms, rng)

        self.moves: dict[
            str, MoveStorage[DisplacementMove | ExchangeMove, ExchangeContext]
        ] = moves

        self.chemical_potential: float = 0.0
        self.number_of_particles: int = 0

        self.accessible_volume: float = self.atoms.cell.volume

        self.reset()

    def save_state(self) -> None:
        """
        Save the current state of the context and update move labels.

        Raises
        ------
        ValueError
            If the labels are not updated correctly.
        """
        if len(self.added_indices):
            for move_storage in self.moves.values():
                move = move_storage.move
                label: int = move.default_label or (
                    max(move.unique_labels) + 1 if len(move.unique_labels) else 0
                )
                move.set_labels(
                    np.hstack((move.labels, np.full(len(self.added_indices), label)))
                )

                if len(move.labels) != len(self.atoms):
                    raise ValueError("Labels not updated correctly.")
        elif len(self.deleted_indices):
            for move_storage in self.moves.values():
                move = move_storage.move
                move.set_labels(np.delete(move.labels, self.deleted_indices))

                if len(move.labels) != len(self.atoms):
                    raise ValueError("Labels not updated correctly.")

        self.number_of_particles += self.particle_delta

        super().save_state()

    def revert_state(self) -> None:
        """
        Revert the last move made by the context.

        Raises
        ------
        ValueError
            If the last deleted atoms are not saved.
        """
        if len(self.added_indices) != 0:
            del self.atoms[self.added_indices]
        if len(self.deleted_indices) != 0:
            if len(self.deleted_atoms) == 0:
                raise ValueError("Last deleted atoms not saved.")

            reinsert_atoms(self.atoms, self.deleted_atoms, self.deleted_indices)

        super().revert_state()

    def reset(self) -> None:
        """Reset the context by setting all attributes to their default values"""
        self.added_indices: IntegerArray = []
        self.added_atoms: Atoms = Atoms()
        self.deleted_indices: IntegerArray = []
        self.deleted_atoms: Atoms = Atoms()

        self.particle_delta: int = 0
