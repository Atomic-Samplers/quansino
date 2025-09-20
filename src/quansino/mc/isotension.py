"""Module to perform isobaric (NPT) Monte Carlo simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast
from warnings import warn

import numpy as np

from quansino.mc.contexts import DeformationContext
from quansino.mc.criteria import CanonicalCriteria, IsotensionCriteria
from quansino.mc.isobaric import Isobaric
from quansino.moves.cell import CellMove
from quansino.moves.displacement import DisplacementMove

if TYPE_CHECKING:

    from ase.atoms import Atoms

    from quansino.protocols import Criteria, Move
    from quansino.type_hints import Stress

MoveType = TypeVar("MoveType", bound="Move")
CriteriaType = TypeVar("CriteriaType", bound="Criteria")


class Isotension(Isobaric[MoveType, CriteriaType], Generic[MoveType, CriteriaType]):
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
        CellMove: IsotensionCriteria,
    }
    default_context: ClassVar = DeformationContext

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        pressure: float = 0.0,
        external_stress: Stress | None = None,
        max_cycles: int | None = None,
        default_displacement_move: MoveType | None = None,
        default_cell_move: MoveType | None = None,
        **mc_kwargs,
    ) -> None:
        """Initialize the `Isotension` object."""
        super().__init__(
            atoms,
            temperature,
            pressure,
            max_cycles,
            default_displacement_move,
            **mc_kwargs,
        )

        self.pressure = pressure
        self.external_stress = (
            np.zeros((3, 3)) if external_stress is None else external_stress
        )

        if default_cell_move:
            self.add_move(default_cell_move, name="default_cell_move")

            # if self.default_logger:
            # self.default_logger.add_field()

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
    def external_stress(self) -> Stress:
        """
        The external stress tensor applied to the system.

        Returns
        -------
        Stress
            The external stress tensor in eV/Å^3.
        """
        return self.context.external_stress

    @external_stress.setter
    def external_stress(self, stress: Stress) -> None:
        """
        Set the external stress tensor applied to the system.

        Parameters
        ----------
        stress : Stress
            The external stress tensor in eV/Å^3.
        """
        self.context.external_stress = stress
