"""Module containing additional constraints classes"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ase.constraints import FixConstraint

if TYPE_CHECKING:
    from ase.atoms import Atoms

    from quansino.type_hints import Momenta


class FixRot(FixConstraint):
    """
    Constraint class to remove the rotation of the system by subtracting the angular momentum from the momenta. Only to use with free boundary conditions, this constraint is not compatible with periodic boundary conditions.
    """

    def adjust_momenta(self, atoms: Atoms, momenta: Momenta) -> None:
        """
        Adjust the momenta of the atoms to remove angular momentum.

        Parameters
        ----------
        atoms : Atoms
            The atoms object to adjust the angular momentum for.
        momenta : Momenta
            The momenta of the atoms to be adjusted.
        """
        positions_to_com = atoms.positions - atoms.get_center_of_mass()

        eig, vecs = atoms.get_moments_of_inertia(vectors=True)

        inv_inertia = np.linalg.inv(np.linalg.inv(vecs) @ np.diag(eig) @ vecs)
        angular_momentum = np.sum(np.cross(positions_to_com, momenta), axis=0)

        omega = inv_inertia @ angular_momentum

        correction = np.cross(omega, positions_to_com)

        masses = atoms.get_masses()

        momenta[:] = momenta - correction * masses[:, None]

    def get_removed_dof(self, *_args: Any, **_kwargs: Any) -> Literal[3]:
        """
        Get the number of degrees of freedom removed by the [`FixRot`][quansino.constraints.FixRot] constraint.

        Returns
        -------
        Literal[3]
            The number of degrees of freedom removed by the [`FixRot`][quansino.constraints.FixRot] constraint, which is always 3.
        """
        return 3

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the [`FixRot`][quansino.constraints.FixRot] object to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the [`FixRot`][quansino.constraints.FixRot] object.
        """
        return {"name": "FixRot", "kwargs": {}}

    todict = to_dict  # type: ignore[ase]
