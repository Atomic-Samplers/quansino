from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.units import kB

if TYPE_CHECKING:

    from quansino.mc.contexts import DisplacementContext


def maxwell_boltzmann_distribution(
    context: DisplacementContext, forced: bool = False
) -> None:
    """
    Set the momenta of the atoms in the context according to the Maxwell-Boltzmann distribution at the specified temperature.

    Parameters
    ----------
    context : DisplacementContext
        The simulation context containing the atoms.
    forced : bool, optional
        If True, the momenta will be scaled to exactly match the specified temperature, by default False.
    """
    temperature = context.temperature * kB
    atoms = context.atoms

    atoms.set_momenta(
        context.rng.standard_normal((len(context.atoms), 3))
        * np.sqrt(atoms.get_masses() * temperature)[:, None]
    )

    if forced:
        real_temperature = (
            2 * atoms.get_kinetic_energy() / atoms.get_number_of_degrees_of_freedom()
        ) + 1.0e-15
        scale = np.sqrt(temperature / real_temperature)
    else:
        scale = 1.0

    atoms.set_momenta(atoms.get_momenta() * scale)
