"""Utility functions for working with atoms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from ase.neighborlist import neighbor_list

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from ase.constraints import FixConstraint

    from quansino.typing import IntegerArray


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


def search_molecules(
    atoms: Atoms, cutoff: float | list[float] | tuple[float] | dict[tuple[str], float]
):
    """Search for molecules in the Atoms object.

    Parameters
    ----------
    atoms
        The Atoms object to search.
    cutoff
        The cutoff distance to use for the search. Can be a single float, a list or tuple of floats
        with the same length as the number of atom types, or a dictionary with the atom types as keys
        and the cutoff distances as values.

    Returns
    -------
    list[list[int]]
        A list of lists of atom indices, where each list contains the indices of the atoms in a
        molecule.
    """
    indices, neighbors = neighbor_list(
        "ij", atoms, cutoff=cutoff, self_interaction=False
    )

    connectivity = np.full((len(atoms), len(atoms)), 0)
    connectivity[indices, neighbors] = 1

    return {
        n: list(mol)
        for n, mol in enumerate(
            nx.connected_components(nx.from_numpy_array(connectivity))
        )
        if len(mol) > 1
    }


def pop_atoms(atoms: Atoms, indices: int | IntegerArray) -> tuple[Atoms, Atoms]:
    """Pop atoms from an Atoms object.

    Parameters
    ----------
    atoms
        The Atoms object to pop atoms from.
    indices
        The indices of the atoms to pop.

    Returns
    -------
    Atoms
        The Atoms object with the popped atoms.
    """
    mask = np.isin(np.arange(len(atoms)), indices)

    return atoms[mask], atoms[~mask]  # type: ignore
