from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.units import _e, _hplanck, _Nav, kB

if TYPE_CHECKING:
    from quansino.mc.contexts import (
        Context,
        DisplacementContext,
        ExchangeContext,
        StrainContext,
    )


class Criteria(ABC):
    """
    Base class for Monte Carlo acceptance criteria.

    This abstract class defines the interface for all acceptance criteria
    used in Monte Carlo simulations. Implementations must provide an
    `evaluate` method that determines whether a move is accepted or rejected.
    """

    @abstractmethod
    def evaluate(self, context: Context, *args, **kwargs) -> bool:
        """
        Evaluate whether a Monte Carlo move should be accepted.

        Parameters
        ----------
        context : Context
            The simulation context containing information about the current state.
        *args, **kwargs
            Additional arguments for the evaluation.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the criteria to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the criteria.
        """
        return {"name": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Criteria:
        """
        Create a criteria object from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the criteria.

        Returns
        -------
        Criteria
            The criteria object created from the dictionary.
        """
        kwargs = data.get("kwargs", {})
        instance = cls(**kwargs)

        for key, value in data.get("attributes", {}).items():
            setattr(instance, key, value)

        return instance


class CanonicalCriteria(Criteria):
    """
    Acceptance criteria for Monte Carlo moves in the canonical (NVT) ensemble.

    This criteria implements the Metropolis algorithm for the canonical ensemble,
    where the number of particles, volume, and temperature are constant.
    """

    @staticmethod
    def evaluate(context: DisplacementContext) -> bool:
        """
        Evaluate the acceptance criteria for a Monte Carlo move.

        Parameters
        ----------
        context : DisplacementContext
            The context of the Monte Carlo simulation.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        energy_difference = context.atoms.get_potential_energy() - context.last_energy

        return context.rng.random() < math.exp(
            -energy_difference / (context.temperature * kB)
        )


class IsobaricCriteria(Criteria):
    """
    Acceptance criteria for Monte Carlo moves in the isothermal-isobaric (NPT) ensemble.

    This criteria implements the Metropolis algorithm for the NPT ensemble,
    where the number of particles, pressure, and temperature are constant.
    The acceptance probability accounts for both energy changes and volume changes.
    """

    @staticmethod
    def evaluate(context: StrainContext) -> bool:
        """
        Evaluate the acceptance criteria for a Monte Carlo move.

        Parameters
        ----------
        context : StrainContext
            The context of the Monte Carlo simulation.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        atoms = context.atoms
        temperature = context.temperature * kB
        energy_difference = atoms.get_potential_energy() - context.last_energy

        current_volume = atoms.get_volume()
        old_volume = context.last_cell.volume

        return context.rng.random() < math.exp(
            -energy_difference / temperature
            + context.pressure * (current_volume - old_volume)
            - (len(atoms) + 1) * np.log(current_volume / old_volume) * temperature
        )


class GrandCanonicalCriteria(Criteria):
    """
    Acceptance criteria for Monte Carlo moves in the grand canonical (μVT) ensemble.

    This criteria implements the Metropolis algorithm for the grand canonical ensemble,
    where the chemical potential, volume, and temperature are constant. The number
    of particles is allowed to fluctuate. The acceptance probability accounts for
    energy changes and particle insertion/deletion.
    """

    @staticmethod
    def evaluate(context: ExchangeContext) -> bool:
        """
        Evaluate the acceptance criteria for a Monte Carlo move.

        Parameters
        ----------
        context : ContextType
            The context of the Monte Carlo simulation.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        energy_difference = context.atoms.get_potential_energy() - context.last_energy

        number_of_exchange_particles = context.number_of_exchange_particles

        if context.added_atoms:
            mass = context.added_atoms.get_masses().sum()
            particle_delta = 1
        elif context.deleted_atoms:
            mass = context.deleted_atoms.get_masses().sum()
            particle_delta = -1
        else:
            raise ValueError("No atoms were added or deleted.")

        volume = context.accessible_volume**particle_delta

        factorial_term = 1
        if particle_delta > 0:
            for i in range(
                number_of_exchange_particles + 1,
                number_of_exchange_particles + particle_delta + 1,
            ):
                factorial_term /= i
        elif particle_delta < 0:
            for i in range(
                number_of_exchange_particles + particle_delta + 1,
                number_of_exchange_particles + 1,
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
