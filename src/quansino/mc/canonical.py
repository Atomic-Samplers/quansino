"""Module to perform canonical (NVT) Monte Carlo simulations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

from ase.units import kB

from quansino.mc.contexts import DisplacementContext
from quansino.mc.core import AcceptanceCriteria, MonteCarlo, MoveStorage
from quansino.moves.displacements import CompositeDisplacementMove, DisplacementMove

if TYPE_CHECKING:
    from collections.abc import Generator

    from ase.atoms import Atoms
    from numpy.random import Generator as RNG


class MetropolisCriteria(AcceptanceCriteria[DisplacementContext]):
    """Default criteria for accepting or rejecting a Monte Carlo move in the canonical ensemble."""

    def evaluate(self, context, energy_difference: float) -> bool:
        """
        Evaluate the acceptance criteria for a Monte Carlo move.

        Parameters
        ----------
        context : DisplacementContext
            The context of the Monte Carlo simulation.
        energy_difference : float
            The energy difference between the current and proposed states.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        return energy_difference < 0 or context.rng.random() < math.exp(
            -energy_difference / (context.temperature * kB)
        )


class Canonical[
    MoveType: DisplacementMove | CompositeDisplacementMove,
    ContextType: DisplacementContext,
](MonteCarlo[MoveType, ContextType]):
    """
    Canonical Monte Carlo simulation object.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on, will be acted upon in place.
    temperature : float
        The temperature of the simulation in Kelvin.
    num_cycles : int, optional
        The number of Monte Carlo cycles to perform, by default equal to the number of atoms.
    default_move : MoveStorage[MoveType, ContextType] | MoveType, optional
        The default move to perform in each cycle. If a `MoveStorage` object is provided, it will be used to initialize the move with its criteria and other parameters. If a `MoveType` object is provided, it will be added using the default criteria and parameters, by default None.
    **mc_kwargs
        Additional keyword arguments to pass to the [`MonteCarlo`][quansino.mc.core.MonteCarlo] parent class.

    Attributes
    ----------
    acceptable_moves : ClassVar[dict[DisplacementMove, MetropolisCriteria]]
        The acceptable moves for the simulation.
    accepted_moves : list[str]
        The names of the moves that were accepted in the last step.
    acceptance_rate : float
        The acceptance rate of the moves in the last step.
    """

    acceptable_moves: ClassVar = {DisplacementMove: MetropolisCriteria}

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        num_cycles: int | None = None,
        default_move: MoveStorage[MoveType, ContextType] | MoveType | None = None,
        **mc_kwargs,
    ) -> None:
        """Initialize the Canonical Monte Carlo object."""
        if num_cycles is None:
            num_cycles = len(atoms)

        super().__init__(atoms, num_cycles=num_cycles, **mc_kwargs)

        self.context.temperature = temperature

        if isinstance(default_move, DisplacementMove):
            self.add_move(default_move, name="default_move")
        elif isinstance(default_move, MoveStorage):
            self.add_move(
                default_move.move,
                default_move.criteria,
                "default_move",
                default_move.interval,
                default_move.probability,
                default_move.minimum_count,
            )

        if self.default_logger:
            self.default_logger.add_field("AcptRate", lambda: self.acceptance_rate)

    @property
    def temperature(self) -> float:
        """The temperature of the simulation in Kelvin."""
        return self.context.temperature

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        """Set the temperature of the simulation in Kelvin."""
        self.context.temperature = temperature

    def calculate_energy_difference(self) -> float:
        """
        Calculate the energy difference between the current and last state.

        Returns
        -------
        float
            The energy difference.
        """
        return self.atoms.get_potential_energy() - self.context.last_results["energy"]

    def yield_moves(self) -> Generator[str, None, None]:
        """
        Yield the names of accepted moves after evaluating their acceptance criteria.

        Yields
        ------
        Generator[str, None, None]
            The names of the accepted moves.
        """
        for move_name in super().yield_moves():
            move_storage = self.moves[move_name]
            move = move_storage.move

            if move():
                energy_difference = self.calculate_energy_difference()

                is_accepted = move_storage.criteria.evaluate(
                    self.context, energy_difference
                )

                if is_accepted:
                    self.context.save_state()
                    yield move_name
                else:
                    self.context.revert_state()

    def create_context(self, atoms: Atoms, rng: RNG) -> DisplacementContext:
        """
        Create a displacement context for the simulation.

        Parameters
        ----------
        atoms : Atoms
            The atomic configuration.
        rng : RNG
            The random number generator.

        Returns
        -------
        DisplacementContext
            The context for the Monte Carlo simulation.
        """
        return DisplacementContext(atoms, rng)

    def pre_step(self) -> None:
        """Perform operations before the Monte Carlo step."""
        if not self.context.last_results.get("energy", None):
            self.atoms.get_potential_energy()
            self.context.last_results = self.atoms.calc.results  # type: ignore

        self.context.last_positions = self.atoms.get_positions()

    def step(self) -> Any:
        """Perform a single Monte Carlo step, iterating over all selected moves."""
        self.pre_step()

        self.accepted_moves = list(self.yield_moves())

        self.acceptance_rate = len(self.accepted_moves) / self.num_cycles
