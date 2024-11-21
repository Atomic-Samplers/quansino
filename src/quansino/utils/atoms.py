"""Utility functions for working with atoms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from ase.constraints import FixConstraint


def has_constraint(atoms: Atoms, constraint_type: type[FixConstraint] | str) -> bool:
    """Check if the Atoms object has constraints.

    Parameters
    ----------
    atoms
        The Atoms object to check.

    Returns
    -------
    bool
        True if the Atoms object has constraints, False otherwise.
    """
    if not isinstance(constraint_type, str):
        constraint_type = constraint_type.__name__

    return any(c for c in atoms.constraints if c.__class__.__name__ == constraint_type)
