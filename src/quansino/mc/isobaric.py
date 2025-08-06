"""Module to perform isobaric (NPT) Monte Carlo simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast
from warnings import warn

from quansino.mc.canonical import Canonical
from quansino.mc.contexts import DeformationContext
from quansino.mc.criteria import CanonicalCriteria, IsobaricCriteria
from quansino.moves.cell import CellMove
from quansino.moves.displacement import DisplacementMove
from quansino.protocols import Criteria, Move

if TYPE_CHECKING:

    from ase.atoms import Atoms


class Isobaric[MoveType: Move, CriteriaType: Criteria](
    Canonical[MoveType, CriteriaType]
):
    """
    Isobaric (NPT) Monte Carlo simulation object for performing NPT simulations. This class is a subclass of the [`Canonical`][quansino.mc.canonical.Canonical] class and provides additional functionality specific to isobaric simulations. It uses the [`DeformationContext`][quansino.mc.contexts.DeformationContext] context by default.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on, will be acted upon in place.
    temperature : float, optional
        The temperature of the simulation in Kelvin, by default 298.15 K.
    pressure : float, optional
        The pressure of the simulation in eV/Å^3, by default 0.0.
    default_displacement_move : MoveType | None, optional
        The default displacement move to perform in each cycle, by default None.
    default_cell_move : MoveType | None, optional
        The default cell move to perform in each cycle, by default None.
    **mc_kwargs : Any
        Additional keyword arguments to pass to the parent classes.

    Attributes
    ----------
    pressure : float
        The pressure of the simulation in eV/Å^3.
    default_criteria : ClassVar
        The default criteria used for the simulation, set to [`CanonicalCriteria`][quansino.mc.criteria.CanonicalCriteria] for [`DisplacementMove`][quansino.moves.displacement.DisplacementMove] and [`IsobaricCriteria`][quansino.mc.criteria.IsobaricCriteria] for [`CellMove`][quansino.moves.cell.CellMove].
    default_context : ClassVar
        The default context used for the simulation, set to [`DeformationContext`][quansino.mc.contexts.DeformationContext].
    """

    default_criteria: ClassVar = {
        DisplacementMove: CanonicalCriteria,
        CellMove: IsobaricCriteria,
    }
    default_context: ClassVar = DeformationContext

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        pressure: float = 0.0,
        max_cycles: int | None = None,
        default_displacement_move: MoveType | None = None,
        default_cell_move: MoveType | None = None,
        **mc_kwargs,
    ) -> None:
        """Initialize the `Isobaric` object."""
        super().__init__(
            atoms, temperature, max_cycles, default_displacement_move, **mc_kwargs
        )

        self.pressure = pressure

        if default_cell_move:
            self.add_move(default_cell_move, name="default_cell_move")

        self.set_default_probability()

        if isinstance(self.context, DeformationContext):
            self.context = cast("DeformationContext", self.context)
        else:
            warn(
                "The context is not a `DeformationContext`. This may lead to unexpected behavior.",
                UserWarning,
                2,
            )

    @property
    def pressure(self) -> float:
        """
        The pressure of the simulation.

        Returns
        -------
        float
            The pressure in eV/Å^3.
        """
        return self.context.pressure

    @pressure.setter
    def pressure(self, pressure: float) -> None:
        """
        Set the pressure of the simulation.

        Parameters
        ----------
        pressure : float
            The pressure in eV/Å^3.
        """
        self.context.pressure = pressure

    def set_default_probability(self) -> None:
        """
        Set the default probability for the cell and displacement moves.

        The probability for cell moves is set to 1/(N+1) and the probability for displacement moves is set to 1/(1+1/N), where N is the number of atoms.
        """
        if cell_move := self.moves.get("default_cell_move"):
            cell_move.probability = 1 / (len(self.atoms) + 1)
        if displacement_move := self.moves.get("default_displacement_move"):
            displacement_move.probability = 1 / (1 + 1 / len(self.atoms))

    def validate_simulation(self) -> None:
        """
        This method also ensures that the cell is saved in the context.
        """
        self.context.last_cell = self.atoms.get_cell()

        super().validate_simulation()

    def revert_state(self) -> None:
        """
        Revert to the previously saved state and undo the last move. This method restores the cell state in addition to the state restored by the parent class.
        """
        super().revert_state()

        try:
            self.atoms.calc.atoms.cell = self.atoms.cell.copy()  # type: ignore[try-attr]
        except AttributeError:
            warn("Atoms object does not have calculator attached.", stacklevel=2)
