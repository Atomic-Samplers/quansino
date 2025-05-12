"""Module for Monte Carlo contexts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase.atoms import Atoms

if TYPE_CHECKING:
    from ase.cell import Cell
    from numpy.random import Generator

    from quansino.type_hints import IntegerArray, Positions


class Context:
    """
    Base class for Monte Carlo contexts. Contexts define the interface between the simulation object, the moves and their criteria. They aim to provide the necessary information for the move to perform its operation, without having to pass whole simulation objects around. Typically, the context should contain all the information required to restart the information, not more, not less. Specific context might be required for different types of moves, for example, [`DisplacementContext`][quansino.mc.contexts.DisplacementContext] for displacement moves and [`ExchangeContext`][quansino.mc.contexts.ExchangeContext] for exchange moves.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.

    Attributes
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.
    """

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
        self.atoms: Atoms = atoms
        self.rng: Generator = rng

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the context to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the context.
        """
        return {}


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
        The atoms object to perform the simulation on. Used in the simulation, and moves.
    rng : Generator
        The random number generator to use. Used in the simulation, criteria, and moves.
    temperature : float
        The temperature of the simulation in Kelvin. Used in the criteria.
    last_positions : Positions
        The positions of the atoms in the last saved state. Used in the simulation.
    last_energy : float
        The energy value from the last saved state. Used in the simulation.
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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the context to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the context.
        """
        return {
            "temperature": self.temperature,
            "last_positions": self.last_positions,
            "last_energy": self.last_energy,
            "moving_indices": self.moving_indices,
        }


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
        The random number generator to use. Used in the simulation, criteria, and moves.
    pressure : float
        The pressure of the system. Used in the criteria.
    last_cell : Cell
        The cell of the atoms in the last saved state. Used in the simulation.
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

        self.pressure = np.nan
        self.last_cell: Cell = atoms.get_cell()

    def to_dict(self):
        return {
            **super().to_dict(),
            "pressure": self.pressure,
            "last_cell": self.last_cell,
        }


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
        The random number generator to use. Used in the simulation, criteria, and moves.
    chemical_potential : float
        The chemical potential of the system. Used in the criteria.
    number_of_exchange_particles : int
        The number of particles that can be exchanged. Used in the criteria.
    accessible_volume : float
        The accessible volume of the system. Used in the criteria.
    added_indices : IntegerArray
        Integer indices of atoms that were added in the last move. Used in the moves.
    added_atoms : Atoms
        Atoms that were added in the last move. Used in the criteria and moves.
    deleted_indices : IntegerArray
        Integer indices of atoms that were deleted in the last move. Used in the moves.
    deleted_atoms : Atoms
        Atoms that were deleted in the last move. Used in the criteria and moves.

    Methods
    -------
    reset()
        Reset the context by setting all exchange-related attributes to their default values.
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
        """
        super().__init__(atoms, rng)

        self.chemical_potential = np.nan

        self.number_of_exchange_particles = 0
        self.accessible_volume = self.atoms.cell.volume

        self.default_label: int | None = None

        self.reset()

    def reset(self) -> None:
        """Reset the context by setting all attributes to their default values"""
        self.added_indices: IntegerArray = []
        self.added_atoms: Atoms = Atoms()
        self.deleted_indices: IntegerArray = []
        self.deleted_atoms: Atoms = Atoms()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the context to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the context.
        """
        return {
            **super().to_dict(),
            "chemical_potential": self.chemical_potential,
            "number_of_exchange_particles": self.number_of_exchange_particles,
            "accessible_volume": self.accessible_volume,
            "added_indices": self.added_indices,
            "added_atoms": self.added_atoms,
            "deleted_indices": self.deleted_indices,
            "deleted_atoms": self.deleted_atoms,
        }
