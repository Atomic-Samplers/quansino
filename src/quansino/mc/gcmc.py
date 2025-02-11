"""Module to perform Grand Canonical (μVT) Monte Carlo simulations."""

from __future__ import annotations

import math
from dataclasses import astuple
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from ase.units import _e, _hplanck, _Nav, kB

from quansino.mc.canonical import Canonical, MetropolisCriteria
from quansino.mc.contexts import ExchangeContext
from quansino.mc.core import AcceptanceCriteria, MoveStorage
from quansino.moves.displacements import DisplacementMove
from quansino.moves.exchange import ExchangeMove

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from numpy.random import Generator as RNG


class GrandCanonicalCriteria[ContextType: ExchangeContext](
    AcceptanceCriteria[ContextType]
):
    """Default criteria for accepting or rejecting a Monte Carlo move in the Grand Canonical ensemble."""

    def evaluate(self, context, energy_difference: float) -> bool:
        """
        Evaluate the acceptance criteria for a Monte Carlo move.

        Parameters
        ----------
        context : ContextType
            The context of the Monte Carlo simulation.
        energy_difference : float
            The energy difference between the current and proposed states.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        number_of_particles = context.number_of_particles
        particle_delta = context.particle_delta

        volume = context.accessible_volume**particle_delta

        if context.added_atoms:
            mass = context.added_atoms.get_masses().sum()
        elif context.deleted_atoms:
            mass = context.deleted_atoms.get_masses().sum()
        else:
            raise ValueError("No atoms were added or deleted.")

        factorial_term = 1
        if particle_delta > 0:
            for i in range(
                number_of_particles + 1, number_of_particles + particle_delta + 1
            ):
                factorial_term /= i
        elif particle_delta < 0:
            for i in range(
                number_of_particles + particle_delta + 1, number_of_particles + 1
            ):
                factorial_term *= i

        debroglie_wavelength = (
            math.sqrt(
                _hplanck**2
                / (2 * np.pi * mass * kB * context.temperature / _Nav * 1e-3 * _e)
            )
            * 1e10
        ) ** (-3 * particle_delta)

        prefactor = volume * factorial_term * debroglie_wavelength
        exponential = (
            particle_delta * context.chemical_potential - energy_difference
        ) / (context.temperature * kB)
        criteria = math.exp(exponential)

        return context.rng.random() < criteria * prefactor


class GrandCanonical[MoveType: DisplacementMove, ContextType: ExchangeContext](
    Canonical[DisplacementMove, ExchangeContext]
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
    default_exchange_move : MoveStorage[MoveType, ContextType] | MoveType, optional
        The default exchange move.
    **mc_kwargs
        Additional keyword arguments for the Monte Carlo simulation.

    Attributes
    ----------
    acceptable_moves : ClassVar[dict[MoveType, AcceptanceCriteria]]
        A dictionary mapping move types to their acceptance criteria.
    context : ExchangeContext
        The context of the Monte Carlo simulation.
    number_of_particles : int
        The number of particles in the simulation.
    """

    acceptable_moves: ClassVar = {
        ExchangeMove: GrandCanonicalCriteria,
        DisplacementMove: MetropolisCriteria,
    }

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        chemical_potential: float,
        number_of_particles: int,
        num_cycles: int,
        default_displacement_move: DisplacementMove | None = None,
        default_exchange_move: (
            MoveStorage[MoveType, ContextType] | MoveType | None
        ) = None,
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
        default_exchange_move : MoveStorage[MoveType, ContextType] | MoveType, optional
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
        self.number_of_particles = number_of_particles

        if isinstance(default_exchange_move, DisplacementMove):
            self.add_move(default_exchange_move, name="default_exchange")
        elif isinstance(default_exchange_move, MoveStorage):
            typed_tuple = cast(
                tuple[MoveType, float, int, int, AcceptanceCriteria],
                astuple(default_exchange_move),
            )
            default_move, probability, interval, minimum_count, criteria = typed_tuple
            self.add_move(
                default_move,
                criteria,
                "default_exchange",
                interval,
                probability,
                minimum_count,
            )

    def create_context(self, atoms: Atoms, rng: RNG) -> ExchangeContext:
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
        return ExchangeContext(atoms, rng, self.moves)

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
    def number_of_particles(self) -> int:
        """
        The number of particles in the simulation.

        Returns
        -------
        int
            The number of particles.
        """
        return self.context.number_of_particles

    @number_of_particles.setter
    def number_of_particles(self, number_of_particles: int) -> None:
        """
        Set the number of particles in the simulation.

        Parameters
        ----------
        number_of_particles : int
            The number of particles.
        """
        self.context.number_of_particles = number_of_particles
