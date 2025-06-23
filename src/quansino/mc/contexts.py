"""Module for Monte Carlo contexts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase.atoms import Atoms

from quansino.utils.atoms import reinsert_atoms

if TYPE_CHECKING:
    from ase.cell import Cell
    from numpy.random import Generator

    from quansino.type_hints import IntegerArray, Positions


class Context:
    """
    Abstract base class for Monte Carlo contexts. Contexts define the interface between the simulation object, the moves and their criteria. They aim to provide the necessary information for the move to perform its operation, without having to pass whole simulation objects around. Specific context might be required for different types of moves, for example, [`DisplacementContext`][quansino.mc.contexts.DisplacementContext] for displacement moves, [`DeformationContext`][quansino.mc.contexts.DeformationContext] for cell deformation move, and [`ExchangeContext`][quansino.mc.contexts.ExchangeContext] for exchange moves.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.

    Attributes
    ----------
    atoms : Atoms
        The atoms object which the context operates on.
    rng : Generator
        The random number generator in use.
    """

    __slots__ = ("atoms", "last_results", "rng")

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
        """Initialize the `Context` object."""
        self.atoms: Atoms = atoms
        self.rng: Generator = rng

    def save_state(self) -> None:
        """
        Save the current state of the context. This method can be overridden by subclasses to save specific attributes.
        """

    def revert_state(self) -> None:
        """
        Revert the context to the last saved state. This method can be overridden by subclasses to revert specific attributes.
        """

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Context object to a dictionary.

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
    temperature : float
        The temperature of the simulation in Kelvin.
    last_positions : Positions
        The positions of the atoms in the last saved state.
    last_energy : float
        The energy value from the last saved state.
    _moving_indices : IntegerArray
        Integer indices of atoms that are being displaced.
    """

    __slots__ = ("_moving_indices", "last_energy", "last_positions", "temperature")

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
        """Initialize the DisplacementContext object."""
        super().__init__(atoms, rng)

        self.temperature: float = 0.0

        self.last_positions: Positions = atoms.get_positions()
        self.last_energy: float = np.nan

        self.reset()

    def reset(self) -> None:
        """Reset the context by setting `moving_indices` to an empty list."""
        self._moving_indices: IntegerArray = []

    def save_state(self) -> None:
        """Save the current state of the context, including the last positions and energy."""
        self.last_positions = self.atoms.get_positions()
        self.last_energy = self.atoms.get_potential_energy()

    def revert_state(self) -> None:
        """Revert the context to the last saved state, restoring the last positions."""
        self.atoms.positions = self.last_positions.copy()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `DisplacementContext` object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `DisplacementContext` object.
        """
        return {
            "temperature": self.temperature,
            "last_positions": self.last_positions,
            "last_energy": self.last_energy,
        }


class DeformationContext(DisplacementContext):
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
    pressure : float
        The pressure of the system.
    last_cell : Cell
        The cell of the atoms in the last saved state.
    """

    __slots__ = ("last_cell", "pressure")

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
        """Initialize the `DeformationContext` object."""
        super().__init__(atoms, rng)

        self.pressure: float = 0.0
        self.last_cell: Cell = atoms.get_cell()

    def save_state(self) -> None:
        """
        Save the current state of the context, including the last cell.
        """
        super().save_state()
        self.last_cell = self.atoms.get_cell()

    def revert_state(self) -> None:
        """
        Revert the context to the last saved state, restoring the last cell.
        """
        super().revert_state()
        self.atoms.set_cell(self.last_cell, scale_atoms=False)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `DeformationContext` object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `DeformationContext` object.
        """
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
    _added_indices : IntegerArray
        Integer indices of atoms that were added in the last move.
    _added_atoms : Atoms
        Atoms that were added in the last move.
    _deleted_indices : IntegerArray
        Integer indices of atoms that were deleted in the last move.
    _deleted_atoms : Atoms
        Atoms that were deleted in the last move.
    accessible_volume : float
        The accessible volume of the system.
    chemical_potential : float
        The chemical potential of the system.
    exchange_atoms : Atoms
        Atoms that can be exchanged in the simulation.
    number_of_exchange_particles : int
        The number of particles that can be exchanged.
    particle_delta : int
        The change in the number of particles in the last move, positive for addition and negative for deletion.
    """

    __slots__ = (
        "_added_atoms",
        "_added_indices",
        "_deleted_atoms",
        "_deleted_indices",
        "accessible_volume",
        "chemical_potential",
        "exchange_atoms",
        "number_of_exchange_particles",
        "particle_delta",
    )

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
        """Initialize the `ExchangeContext` object."""
        super().__init__(atoms, rng)

        self.chemical_potential = np.nan

        self.exchange_atoms: Atoms = Atoms()
        self.number_of_exchange_particles = 0

        self.accessible_volume = self.atoms.cell.volume

        self.reset()

    def reset(self) -> None:
        """
        Reset the context by setting all attributes to their default values.
        """
        self._added_indices: IntegerArray = []
        self._added_atoms: Atoms = Atoms()
        self._deleted_indices: IntegerArray = []
        self._deleted_atoms: Atoms = Atoms()

        self.particle_delta = 0

        super().reset()

    def revert_state(self) -> None:
        """
        Revert the context to the last saved state.
        """
        if len(self._added_indices) != 0:
            del self.atoms[self._added_indices]
        if len(self._deleted_indices) != 0:
            if len(self._deleted_atoms) == 0:
                raise ValueError("Last deleted atoms was not saved.")

            reinsert_atoms(self.atoms, self._deleted_atoms, self._deleted_indices)

        super().revert_state()
        self.reset()

    def save_state(self) -> None:
        """
        Save the current state of the context, including the number of exchange particles.
        """
        super().save_state()
        self.number_of_exchange_particles += self.particle_delta
        self.reset()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `ExchangeContext` object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `ExchangeContext` object.
        """
        return {
            **super().to_dict(),
            "chemical_potential": self.chemical_potential,
            "number_of_exchange_particles": self.number_of_exchange_particles,
            "accessible_volume": self.accessible_volume,
            "exchange_atoms": self.exchange_atoms,
        }
