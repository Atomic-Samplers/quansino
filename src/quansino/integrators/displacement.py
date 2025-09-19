from __future__ import annotations

from typing import TYPE_CHECKING

from ase.units import fs

from quansino.integrators.core import BaseIntegrator

if TYPE_CHECKING:
    from quansino.mc.contexts import DisplacementContext


class Verlet(BaseIntegrator):
    """
    Class for a Verlet operation that displaces atoms based on their forces.

    This operation uses the Verlet algorithm to displace atoms based on their
    forces, scaled by a delta factor and adjusted by the masses of the atoms.

    Parameters
    ----------
    delta : float, optional
        The scaling factor for the displacement (default is 1.0).
    masses_scaling_power : float, optional
        The power to which the masses are raised for scaling (default is 0.5).

    Returns
    -------
    Displacement
        A displacement vector for the selected atoms based on their forces.
    """

    def __init__(
        self, dt: float = 1.0, max_steps: int = 100, apply_constraints: bool = True
    ) -> None:
        """
        Initialize the `Verlet` integrator.

        Parameters
        ----------
        dt : float, optional
            The time step for the integration in femtoseconds, by default 1.0 fs.
        max_steps : int, optional
            The maximum number of steps to perform in the integration, by default 100.
        apply_constraints : bool, optional
            Whether to apply constraints during the integration, by default True.
        """
        self.dt = dt * fs
        self.max_steps = max_steps
        self.apply_constraints = apply_constraints

    def integrate(self, context: DisplacementContext) -> None:
        """
        Perform a single step of the Velocity Verlet integration.

        Parameters
        ----------
        context : DisplacementContext
            The simulation context containing the atoms.
        """
        atoms = context.atoms
        masses = atoms.get_masses()[:, None]
        forces = atoms.get_forces()

        for _ in range(self.max_steps):
            new_momenta = atoms.get_momenta() + 0.5 * forces * self.dt

            positions = atoms.get_positions()
            atoms.set_positions(
                positions + new_momenta / masses * self.dt,
                apply_constraint=self.apply_constraints,
            )

            if self.apply_constraints:
                new_momenta = (atoms.positions - positions) * masses / self.dt

            forces = atoms.get_forces()
            atoms.set_momenta(
                new_momenta + 0.5 * forces * self.dt,
                apply_constraint=self.apply_constraints,
            )
