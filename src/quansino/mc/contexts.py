"""Module for Monte Carlo contexts"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.atoms import Atoms

if TYPE_CHECKING:
    from numpy.random import Generator

    from quansino.type_hints import IntegerArray, Positions


class Context:
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

        self.temperature: float = 0.0

        self.last_positions: Positions = atoms.get_positions()
        self.last_energy: float = np.nan

        self.reset()

    def reset(self) -> None:
        """Reset the context by setting `moving_indices` to an empty list."""
        self.moving_indices: IntegerArray = []


class StrainContext(DisplacementContext):
    """
    Context for strain moves i.e. moves that change the cell of the simulation.

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
    number_of_particles : int
        The number of particles in the system. Used in the criteria. By default equal to the number of atoms in the simulation.
    """

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
        """
        Initialize the StrainContext object.

        Parameters
        ----------
        atoms : Atoms
            The atoms object to perform the simulation on.
        rng : Generator
            The random number generator to use.
        """
        super().__init__(atoms, rng)

        self.pressure: float = np.nan
        self.last_cell = atoms.get_cell()


class ExchangeContext(DisplacementContext):
    """
    Context for exchange moves i.e. moves that exchange atoms.

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
    accessible_volume : float
        The accessible volume of the system. Used in the criteria.
    particle_delta : int
        The change in the number of particles in the system. Used in the criteria and moves.

    Methods
    -------
    save_state()
        Save the current state of the context and update move labels.
    """

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
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

        self.chemical_potential: float = np.nan
        self.number_of_exchange_particles: int = 0

        self.accessible_volume: float = self.atoms.cell.volume

        self.reset()

    def reset(self) -> None:
        """Reset the context by setting all attributes to their default values"""
        self.added_indices: IntegerArray = []
        self.added_atoms: Atoms = Atoms()
        self.deleted_indices: IntegerArray = []
        self.deleted_atoms: Atoms = Atoms()

        self.particle_delta: int = 0
