from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from ase.units import _e, _hplanck, _Nav, kB

if TYPE_CHECKING:
    from quansino.mc.contexts import (
        Context,
        DeformationContext,
        DisplacementContext,
        ExchangeContext,
    )


class BaseCriteria(ABC):
    """
    Base class for acceptance criteria, it defines the interface for acceptance criteria used in simulations. Implementations must provide an `evaluate` method that determines whether a move is accepted or rejected.
    """

    @abstractmethod
    def evaluate(self, context: Context, *args, **kwargs) -> bool:
        """
        Evaluate whether a Monte Carlo move should be accepted. This method should be implemented in subclasses.

        Parameters
        ----------
        context : Context
            The simulation context containing information about the current state.
        *args : Any
            Positional arguments passed to the method.
        **kwargs : Any
            Keyword arguments passed to the method.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `BaseCriteria` object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `BaseCriteria` object.
        """
        return {"name": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a `BaseCriteria` object from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary representation of the `BaseCriteria` object.

        Returns
        -------
        Self
            The `BaseCriteria` object created from the dictionary.
        """
        kwargs = data.get("kwargs", {})
        instance = cls(**kwargs)

        for key, value in data.get("attributes", {}).items():
            setattr(instance, key, value)

        return instance


class CanonicalCriteria(BaseCriteria):
    """
    Acceptance criteria for Monte Carlo simulation in the canonical (NVT) ensemble.
    """

    @staticmethod
    def evaluate(context: DisplacementContext) -> bool:
        """
        Evaluate the acceptance criteria.

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


class IsobaricCriteria(BaseCriteria):
    """
    Acceptance criteria for moves in the isothermal-isobaric (NPT) ensemble.
    """

    @staticmethod
    def evaluate(context: DeformationContext) -> bool:
        """
        Evaluate the acceptance criteria for a Monte Carlo move.

        Parameters
        ----------
        context : DeformationContext
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


class GrandCanonicalCriteria(BaseCriteria):
    """
    Acceptance criteria for Monte Carlo moves in the grand canonical (μVT) ensemble.
    """

    def evaluate(self, context: ExchangeContext) -> bool:
        """
        Evaluate the acceptance criteria for a Monte Carlo move.

        Parameters
        ----------
        context : ExchangeContext
            The context of the Monte Carlo simulation containing exchange-specific information such as added/deleted atoms, chemical potential, and accessible volume.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        energy_difference = context.atoms.get_potential_energy() - context.last_energy

        number_of_exchange_particles = context.number_of_exchange_particles
        mass = context.exchange_atoms.get_masses().sum()
        particle_delta = context.particle_delta

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
