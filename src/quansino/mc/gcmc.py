"""Module to perform Grand Canonical (μVT) Monte Carlo simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from warnings import warn

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import ExchangeContext
from quansino.mc.criteria import CanonicalCriteria, GrandCanonicalCriteria
from quansino.moves.displacement import DisplacementMove
from quansino.moves.exchange import ExchangeMove
from quansino.moves.protocol import ExchangeProtocol
from quansino.utils.atoms import reinsert_atoms

if TYPE_CHECKING:
    from ase.atoms import Atoms

    from quansino.mc.core import MoveStorage


class GrandCanonical[MoveProtocol: ExchangeProtocol, ContextType: ExchangeContext](
    Canonical[MoveProtocol, ContextType]
):
    """
    Grand Canonical (μVT) Monte Carlo object.

    Parameters
    ----------
    atoms : Atoms
        The atomic configuration.
    temperature : float
        The temperature of the simulation in Kelvin.
    chemical_potential : float
        The chemical potential of the system in eV.
    number_of_exchange_particles : int
        The number of particles that can be exchanged in the simulation.
    max_cycles : int
        The number of Monte Carlo cycles.
    default_displacement_move : MoveProtocol, optional
        The default displacement move.
    default_exchange_move : MoveStorage[MoveProtocol] | MoveProtocol, optional
        The default exchange move.
    **mc_kwargs
        Additional keyword arguments for the Monte Carlo simulation.

    Attributes
    ----------
    chemical_potential : float
        The chemical potential of the simulation in eV.
    number_of_exchange_particles : int
        The number of particles that can be exchanged in the simulation.
    """

    default_criteria: ClassVar = {
        ExchangeMove: GrandCanonicalCriteria,
        DisplacementMove: CanonicalCriteria,
    }
    default_context: ClassVar = ExchangeContext

    def __init__(
        self,
        atoms: Atoms,
        temperature: float = 298.15,
        chemical_potential: float = 0.0,
        number_of_exchange_particles: int = 0,
        max_cycles: int | None = None,
        default_displacement_move: (
            MoveStorage[MoveProtocol] | MoveProtocol | None
        ) = None,
        default_exchange_move: MoveStorage[MoveProtocol] | MoveProtocol | None = None,
        **mc_kwargs,
    ) -> None:
        """Initialize the Grand Canonical Monte Carlo object."""
        super().__init__(
            atoms,
            temperature=temperature,
            max_cycles=max_cycles,
            default_displacement_move=default_displacement_move,
            **mc_kwargs,
        )

        self.chemical_potential = chemical_potential
        self.number_of_exchange_particles = number_of_exchange_particles

        if default_exchange_move:
            self.add_move(default_exchange_move, name="default_exchange_move")

    @property
    def chemical_potential(self) -> float:
        """
        The chemical potential of the simulation.

        Returns
        -------
        float
            The chemical potential in eV.
        """
        return self.context.chemical_potential

    @chemical_potential.setter
    def chemical_potential(self, chemical_potential: float) -> None:
        """
        Set the chemical potential of the simulation.

        Parameters
        ----------
        chemical_potential : float
            The chemical potential in eV.
        """
        self.context.chemical_potential = chemical_potential

    @property
    def number_of_exchange_particles(self) -> int:
        """
        The number of particles that can be exchanged in the simulation.

        Returns
        -------
        int
            The number of particles.
        """
        return self.context.number_of_exchange_particles

    @number_of_exchange_particles.setter
    def number_of_exchange_particles(self, number_of_exchange_particles: int) -> None:
        """
        Set the number of particles that can be exchanged in the simulation.

        Parameters
        ----------
        number_of_exchange_particles : int
            The number of particles.
        """
        self.context.number_of_exchange_particles = number_of_exchange_particles

    def save_state(self) -> None:
        """
        Save the current state of the context and update move labels.

        Raises
        ------
        ValueError
            If the labels are not updated correctly.
        """
        for move_storage in self.moves.values():
            move_storage.move.update(
                self.context.added_indices, self.context.deleted_indices
            )

        super().save_state()

    def revert_state(self) -> None:
        """
        Revert the last move made by the context.

        Raises
        ------
        ValueError
            If the last deleted atoms are not saved.
        """
        if len(self.context.added_indices) != 0:
            del self.atoms[self.context.added_indices]
        if len(self.context.deleted_indices) != 0:
            if len(self.context.deleted_atoms) == 0:
                raise ValueError("Last deleted atoms not saved.")

            reinsert_atoms(
                self.atoms, self.context.deleted_atoms, self.context.deleted_indices
            )

        self.atoms.positions = self.context.last_positions.copy()

        try:
            self.atoms.calc.atoms = self.atoms.copy()  # type: ignore
            self.atoms.calc.results = self.last_results.copy()  # type: ignore
        except AttributeError:
            warn("Atoms object does not have calculator attached.", stacklevel=2)
