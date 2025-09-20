"""Module to perform canonical (NVT) Monte Carlo simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast
from warnings import warn

import numpy as np

from quansino.mc.contexts import DisplacementContext, HamiltonianDisplacementContext
from quansino.mc.core import MonteCarlo
from quansino.mc.criteria import CanonicalCriteria, HamiltonianCanonicalCriteria
from quansino.moves.displacement import DisplacementMove, HamiltonianDisplacementMove

if TYPE_CHECKING:

    from ase.atoms import Atoms

    from quansino.protocols import Criteria, Move

MoveType = TypeVar("MoveType", bound="Move")
CriteriaType = TypeVar("CriteriaType", bound="Criteria")


class Canonical(MonteCarlo[MoveType, CriteriaType], Generic[MoveType, CriteriaType]):
    """
    Canonical Monte Carlo simulation object for performing NVT simulations. This class is a subclass of the [`MonteCarlo`][quansino.mc.core.MonteCarlo] class and provides additional functionality specific to canonical simulations. By default, it uses the [`DisplacementContext`][quansino.mc.contexts.DisplacementContext] context.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on, will be acted upon in place.
    temperature : float, optional
        The temperature of the simulation in Kelvin, by default 298.15 K.
    max_cycles : int, optional
        The number of Monte Carlo cycles to perform, by default equal to the number of atoms.
    default_displacement_move : MoveType | None, optional
        The default displacement move to perform in each cycle, by default None.
    **mc_kwargs : Any
        Additional keyword arguments to pass to the [`MonteCarlo`][quansino.mc.core.MonteCarlo] and [`Driver`][quansino.mc.driver.Driver] class.

    Attributes
    ----------
    default_context : ClassVar[type[DisplacementContext]]
        The default context used for the simulation, set to [`DisplacementContext`][quansino.mc.contexts.DisplacementContext].
    default_criteria : ClassVar[dict[type[Move], type[Criteria]]]
        The default criteria used for the simulation, set to [`CanonicalCriteria`][quansino.mc.criteria.CanonicalCriteria].
    """

    default_criteria: ClassVar = {
        DisplacementMove: CanonicalCriteria,
        HamiltonianDisplacementMove: HamiltonianCanonicalCriteria,
    }
    default_context: ClassVar = DisplacementContext

    def __init__(
        self,
        atoms: Atoms,
        temperature: float = 298.15,
        max_cycles: int | None = None,
        default_displacement_move: MoveType | None = None,
        **mc_kwargs: Any,
    ) -> None:
        """Initialize the `Canonical` object."""
        if max_cycles is None:
            max_cycles = len(atoms)

        super().__init__(atoms, max_cycles=max_cycles, **mc_kwargs)

        self.temperature = temperature

        if default_displacement_move:
            self.add_move(default_displacement_move, name="default_displacement_move")

        if self.default_logger:
            self.default_logger.add_field("AcptRate", lambda: self.acceptance_rate)

        if isinstance(self.context, DisplacementContext):
            self.context = cast("DisplacementContext", self.context)
        else:
            warn(
                "The context is not a `DisplacementContext`. This may lead to unexpected behavior.",
                UserWarning,
                2,
            )

    @property
    def temperature(self) -> float:
        """
        The temperature of the simulation in Kelvin, retrieved from the context.

        Returns
        -------
        float
            The temperature in Kelvin.
        """
        return self.context.temperature

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        """
        Set the temperature of the simulation in Kelvin, updating the context.

        Parameters
        ----------
        temperature : float
            The temperature in Kelvin.
        """
        self.context.temperature = temperature

    def validate_simulation(self) -> None:
        """
        Validate the simulation by checking if the last positions and last energy are set.
        """
        self.context.last_positions = self.atoms.get_positions()

        if np.isnan(self.context.last_potential_energy):
            self.context.last_potential_energy = self.atoms.get_potential_energy()

        super().validate_simulation()

    def revert_state(self) -> None:
        """
        Revert to the previously saved state of the atoms, attempting to restore the positions from the last saved state.
        """
        super().revert_state()

        try:
            self.atoms.calc.atoms.positions = self.atoms.positions.copy()  # type: ignore[try-attr]
        except AttributeError:
            warn(
                "The calculator does not support restoring positions. Please check that your calculator is fully compatible with quansino.",
                UserWarning,
                2,
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the `Canonical` object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the `Canonical` object.
        """
        dictionary = super().to_dict()
        dictionary["kwargs"]["temperature"] = self.temperature

        return dictionary


class HamiltonianCanonical(
    Canonical[MoveType, CriteriaType], Generic[MoveType, CriteriaType]
):
    """
    Hamiltonian Canonical Monte Carlo simulation object for performing NVT simulations with Hamiltonian moves. This class is a subclass of the [`Canonical`][quansino.mc.canonical.Canonical] class and provides additional functionality specific to Hamiltonian canonical simulations. By default, it uses the [`HamiltonianDisplacementContext`][quansino.mc.contexts.HamiltonianDisplacementContext] context.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on, will be acted upon in place.
    **canonical_kwargs : Any
        Additional keyword arguments to pass to the [`Canonical`][quansino.mc.canonical.Canonical] class.

    Attributes
    ----------
    default_context : ClassVar[type[HamiltonianDisplacementContext]]
        The default context used for the simulation, set to [`HamiltonianDisplacementContext`][quansino.mc.contexts.HamiltonianDisplacementContext].
    """

    default_context: ClassVar = HamiltonianDisplacementContext

    default_criteria: ClassVar = {
        **Canonical.default_criteria,
        HamiltonianDisplacementMove: HamiltonianCanonicalCriteria,
    }

    def __init__(self, atoms: Atoms, **canonical_kwargs: Any):
        """Initialize the `HamiltonianCanonical` object."""
        super().__init__(atoms, **canonical_kwargs)

        if self.default_logger is not None:
            self.default_logger.add_field("Ekin[eV]", self.atoms.get_kinetic_energy)

        if isinstance(self.context, HamiltonianDisplacementContext):
            self.context = cast("HamiltonianDisplacementContext", self.context)
        else:
            warn(
                "The context is not a `HamiltonianDisplacementContext`. This may lead to unexpected behavior.",
                UserWarning,
                2,
            )
