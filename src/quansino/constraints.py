"""Module containing additional constraints classes"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.constraints import FixConstraint

if TYPE_CHECKING:
    from ase.atoms import Atoms

    from quansino.type_hints import Momenta


class FixRot(FixConstraint):
    """
    Constraint class to remove the rotation of the system by
    subtracting the angular momentum from the momenta. Only to use
    with free boundary conditions. This constraint is not compatible with periodic boundary conditions.
    """

    def todict(self) -> dict:  # type: ignore
        """
        Convert the constraint to a dictionary.

        Returns
        -------
        dict
            The dictionary representation of the constraint.
        """
        return {"name": "FixRot", "kwargs": {}}

    def get_removed_dof(self, _) -> Literal[3]:  # type: ignore
        """Get the number of degrees of freedom removed by the constraint.

        Returns
        -------
        int
            The number of degrees of freedom removed by the constraint.
        """

        return 3

    def adjust_momenta(self, atoms: Atoms, momenta: Momenta) -> None:
        """Adjust the momenta of the atoms to remove the rotation.

        Parameters
        ----------
        atoms
            The atoms object to adjust the momenta of.
        momenta
            The momenta of the atoms to adjust.
        """

        positions_to_com = atoms.positions - atoms.get_center_of_mass()

        eig, vecs = atoms.get_moments_of_inertia(vectors=True)

        inv_inertia = np.linalg.inv(np.linalg.inv(vecs) @ np.diag(eig) @ vecs)
        angular_momentum = np.sum(np.cross(positions_to_com, momenta), axis=0)

        omega = inv_inertia @ angular_momentum

        correction = np.cross(omega, positions_to_com)

        masses = atoms.get_masses()
        momenta = momenta - correction * masses[:, None]
