"""Module to perform Grand Canonical (μVT) Monte Carlo simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast
from warnings import warn

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import ExchangeContext
from quansino.mc.criteria import CanonicalCriteria, GrandCanonicalCriteria
from quansino.moves.displacement import DisplacementMove
from quansino.moves.exchange import ExchangeMove
from quansino.protocols import Criteria

if TYPE_CHECKING:

    from ase.atoms import Atoms

    from quansino.protocols import Move


class GrandCanonical[MoveType: Move, CriteriaType: Criteria](
    Canonical[MoveType, CriteriaType]
):
    """
    Grand Canonical (μVT) Monte Carlo object for performing simulations in the grand canonical ensemble. This class is a subclass of the [`Canonical`][quansino.mc.canonical.Canonical] class and provides additional functionality specific to grand canonical simulations. It uses the [`ExchangeContext`][quansino.mc.contexts.ExchangeContext] context by default.

    Parameters
    ----------
    atoms : Atoms
        The atomic configuration.
    exchange_atoms : Atoms | None, optional
        The atoms that can be exchanged in the simulation, by default None.
    temperature : float, optional
        The temperature of the simulation in Kelvin, by default 298.15 K.
    chemical_potential : float, optional
        The chemical potential of the system in eV, by default 0.0 eV.
    number_of_exchange_particles : int
        The number of particles that can be exchanged already in the `Atoms` object, by default 0.
    default_displacement_move : MoveType | None, optional
        The default displacement move to perform in each cycle, by default None.
    default_exchange_move : MoveType | None, optional
        The default exchange move to perform in each cycle, by default None.
    **mc_kwargs : Any
        Additional keyword arguments for the Monte Carlo simulation.

    Attributes
    ----------
    accessible_volume : float
        The accessible volume for exchange particles in the simulation in Ångstroms cubed.
    exchange_atoms : Atoms
        The atoms that can be exchanged in the simulation.
    chemical_potential : float
        The chemical potential of the simulation in eV.
    number_of_exchange_particles : int
        The number of particles that can be exchanged in the simulation.
    """

    default_criteria: ClassVar = {
        ExchangeMove: GrandCanonicalCriteria,
        DisplacementMove: CanonicalCriteria,
    }
    default_context = ExchangeContext

    def __init__(
        self,
        atoms: Atoms,
        exchange_atoms: Atoms | None = None,
        temperature: float = 298.15,
        chemical_potential: float = 0.0,
        number_of_exchange_particles: int = 0,
        default_displacement_move: MoveType | None = None,
        default_exchange_move: MoveType | None = None,
        **mc_kwargs,
    ) -> None:
        """Initialize the `GrandCanonical` object."""
        super().__init__(
            atoms,
            temperature=temperature,
            default_displacement_move=default_displacement_move,
            **mc_kwargs,
        )

        if exchange_atoms is not None:
            self.exchange_atoms = exchange_atoms

        self.chemical_potential = chemical_potential
        self.number_of_exchange_particles = number_of_exchange_particles

        if default_exchange_move:
            self.add_move(default_exchange_move, name="default_exchange_move")

        if self.default_logger:
            self.default_logger.add_field("Natoms", self.atoms.__len__, "{:>10d}")

        if isinstance(self.context, ExchangeContext):
            self.context = cast("ExchangeContext", self.context)
        else:
            warn(
                "The context is not a `ExchangeContext`. This may lead to unexpected behavior.",
                UserWarning,
                2,
            )

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
    def number_of_exchange_particles(self, value: int) -> None:
        """
        Set the number of particles that can be exchanged in the simulation.

        Parameters
        ----------
        value : int
            The number of particles.
        """
        self.context.number_of_exchange_particles = value

    @property
    def accessible_volume(self) -> float:
        """
        The accessible volume for exchange particles in the simulation.

        Returns
        -------
        float
            The accessible volume in Angstroms cubed.
        """
        return self.context.accessible_volume

    @accessible_volume.setter
    def accessible_volume(self, value: float) -> None:
        """
        Set the accessible volume for exchange particles in the simulation.

        Parameters
        ----------
        value : float
            The accessible volume in Ångstroms cubed.
        """
        self.context.accessible_volume = value

    @property
    def exchange_atoms(self) -> Atoms:
        """
        The atoms that can be exchanged in the simulation.

        Returns
        -------
        Atoms
            The exchange atoms.
        """
        return self.context.exchange_atoms

    @exchange_atoms.setter
    def exchange_atoms(self, value: Atoms) -> None:
        """
        Set the atoms that can be exchanged in the simulation.

        Parameters
        ----------
        value : Atoms
            The exchange atoms.
        """
        self.context.exchange_atoms = value

    def save_state(self) -> None:
        """
        Save the current state of the context and update move labels.
        """
        for move_storage in self.moves.values():
            move_storage.move.on_atoms_changed(
                self.context._added_indices, self.context._deleted_indices
            )

        super().save_state()

    def revert_state(self) -> None:
        """
        Revert the last move made to the context.
        """
        self.context.revert_state()

        try:
            self.atoms.calc.atoms = self.atoms.copy()  # type: ignore[try-attr]
            self.atoms.calc.results = self.last_results.copy()  # type: ignore[try-attr]
        except AttributeError:
            warn("`Atoms` object does not have calculator attached.", stacklevel=2)
