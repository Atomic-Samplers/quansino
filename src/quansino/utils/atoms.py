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
    atoms: Atoms,
    cutoff: float | list[float] | tuple[float] | dict[tuple[str, str], float],
    required_size: int | tuple | None = None,
    default_array: IntegerArray | None = None,
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

    molecules = np.asarray(default_array) or np.full(len(atoms), -1)

    if required_size is None:
        required_size = (0, len(atoms))
    elif isinstance(required_size, int):
        required_size = (required_size, required_size)

    for n, mol in enumerate(nx.connected_components(nx.from_numpy_array(connectivity))):
        molecule_array = np.fromiter(mol, dtype=int)
        if required_size[0] <= molecule_array.size <= required_size[1]:
            molecules[molecule_array] = n

    return molecules


def insert_atoms(atoms: Atoms, new_atoms: Atoms, index: int) -> None:
    """Insert atoms into an Atoms object, in place.

    Parameters
    ----------
    atoms
        The Atoms object to insert atoms into.
    new_atoms
        The Atoms object with the atoms to insert.
    index
        The indices where to insert the atoms.

    Returns
    -------
    Atoms
        The Atoms object with the inserted atoms.
    """
    for name in atoms.arrays:
        new_array = (
            new_atoms.get_masses()
            if name == "masses"
            else new_atoms.arrays.get(name, 0)
        )

        atoms.arrays[name] = np.insert(atoms.arrays[name], index, new_array, axis=0)

    for name, array in new_atoms.arrays.items():
        if name not in atoms.arrays:
            new_array = np.zeros((len(atoms), *array.shape[1:]), dtype=array.dtype)
            new_array[index : index + len(new_atoms)] = array

            atoms.set_array(name, new_array)
