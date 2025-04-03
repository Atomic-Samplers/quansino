"""Module to perform Grand Canonical (μVT) Monte Carlo simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import ExchangeContext
from quansino.mc.core import MoveStorage
from quansino.mc.criteria import CanonicalCriteria, GrandCanonicalCriteria
from quansino.moves.displacements import DisplacementMove
from quansino.moves.exchange import ExchangeMove
from quansino.moves.protocol import ExchangeProtocol
from quansino.utils.atoms import reinsert_atoms

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from numpy.random import Generator as RNG


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
        The chemical potential of the system.
    number_of_particles : int
        The number of particles in the simulation.
    num_cycles : int
        The number of Monte Carlo cycles.
    default_displacement_move : DisplacementMove, optional
        The default displacement move.
    default_exchange_move : MoveStorage[MoveProtocol, ContextType] | MoveProtocol, optional
        The default exchange move.
    **mc_kwargs
        Additional keyword arguments for the Monte Carlo simulation.

    Attributes
    ----------
    acceptable_moves : ClassVar[dict[MoveProtocol, AcceptanceCriteria]]
        A dictionary mapping move types to their acceptance criteria.
    context : ExchangeContext
        The context of the Monte Carlo simulation.
    number_of_particles : int
        The number of particles in the simulation.
    """

    default_criteria: ClassVar = {
        ExchangeMove: GrandCanonicalCriteria,
        DisplacementMove: CanonicalCriteria,
    }
    default_context: ClassVar = ExchangeContext

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        chemical_potential: float,
        number_of_exchange_particles: int,
        num_cycles: int,
        default_displacement_move: MoveProtocol | None = None,
        default_exchange_move: MoveStorage[MoveProtocol] | MoveProtocol | None = None,
        **mc_kwargs,
    ) -> None:
        """
        Initialize the Grand Canonical Monte Carlo object.

        Parameters
        ----------
        atoms : Atoms
            The atomic configuration.
        temperature : float
            The temperature of the simulation in Kelvin.
        chemical_potential : float
            The chemical potential of the system.
        number_of_particles : int
            The number of particles in the simulation.
        num_cycles : int
            The number of Monte Carlo cycles.
        default_displacement_move : DisplacementMove, optional
            The default displacement move.
        default_exchange_move : MoveStorage[MoveProtocol, ContextType] | MoveProtocol, optional
            The default exchange move.
        **mc_kwargs
            Additional keyword arguments for the Monte Carlo simulation.
        """
        super().__init__(
            atoms,
            temperature=temperature,
            num_cycles=num_cycles,
            default_move=default_displacement_move,
            **mc_kwargs,
        )

        self.context.chemical_potential = chemical_potential
        self.number_of_exchange_particles = number_of_exchange_particles

        if isinstance(default_exchange_move, DisplacementMove):
            self.add_move(default_exchange_move, name="default_exchange")
        elif isinstance(default_exchange_move, MoveStorage):
            self.add_move(
                default_exchange_move.move,
                default_exchange_move.criteria,
                "default_exchange",
                default_exchange_move.interval,
                default_exchange_move.probability,
                default_exchange_move.minimum_count,
            )

    def create_context(self, atoms: Atoms, rng: RNG) -> ContextType:
        """
        Create the context for the Monte Carlo simulation.

        Parameters
        ----------
        atoms : Atoms
            The atomic configuration.
        rng : RNG
            The random number generator.

        Returns
        -------
        ExchangeContext
            The context for the Monte Carlo simulation.
        """
        return cast(ContextType, ExchangeContext(atoms, rng))

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
        The number of particles in the simulation.

        Returns
        -------
        int
            The number of particles.
        """
        return self.context.number_of_exchange_particles

    @number_of_exchange_particles.setter
    def number_of_exchange_particles(self, number_of_exchange_particles: int) -> None:
        """
        Set the number of particles in the simulation.

        Parameters
        ----------
        number_of_particles : int
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

        self.number_of_particles += self.context.particle_delta

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

        super().revert_state()
