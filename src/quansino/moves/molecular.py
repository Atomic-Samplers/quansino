from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quansino.moves.atomic import AtomicMove

if TYPE_CHECKING:
    from quansino.typing import Displacement, IntegerArray


class MolecularMove(AtomicMove):
    def __init__(
        self,
        molecule_ids: IntegerArray | dict[int, IntegerArray | int],
        delta: float = 1.0,
        molecule_moving_per_step: int = 1,
        move_type: str = "rotation",
        apply_constraints: bool = True,
    ) -> None:
        self.molecule_ids = molecule_ids
        self.molecule_moving_per_step = molecule_moving_per_step

        super().__init__(
            delta=delta,
            moving_indices=np.array(list(self._molecule_ids.values())).flatten(),
            move_type=move_type,
            moving_per_step=1,
            apply_constraints=apply_constraints,
        )

        self.move_functions.update(
            {
                "rotation": self.rotation,
                "translation_rotation": self.translation_rotation,
            }
        )

    def __call__(self) -> bool | list[bool]:
        successes = []

        molecule_ids_list = list(self.molecule_ids.keys())
        for _ in range(self.molecule_moving_per_step):
            self.state.to_move = int(self.context.rng.choice(molecule_ids_list))
            self.state.to_move = np.array(self.molecule_ids[self.state.to_move])
            successes.append(super().__call__())

        return successes

    @property
    def molecule_ids(self) -> dict[int, IntegerArray | int]:
        return self._molecule_ids

    @molecule_ids.setter
    def molecule_ids(
        self, molecule_ids: IntegerArray | dict[int, IntegerArray | int]
    ) -> None:
        if isinstance(molecule_ids, (list, tuple, np.ndarray)):
            molecule_ids = {
                int(i): np.where(i == molecule_ids)[0]
                for i in np.unique(molecule_ids)
                if i >= 0
            }
        self._molecule_ids = molecule_ids

    def rotation(self, center="COM") -> Displacement:
        """Move atoms in a random direction"""
        if isinstance(self.state.to_move, int):
            self.state.to_move = [self.state.to_move]

        molecule = self._context.atoms[self.state.to_move]
        phi, theta, psi = self._context.rng.uniform(0, 2 * np.pi, 3)
        molecule.euler_rotate(phi, theta, psi, center=center)  # type: ignore

        return molecule.positions - self._context.atoms.positions[self.state.to_move]  # type: ignore

    def translation_rotation(self, center="COM") -> Displacement:
        """Move atoms in a random direction"""
        if isinstance(self.state.to_move, int):
            self.state.to_move = [self.state.to_move]

        molecule = self._context.atoms[self.state.to_move]
        phi, theta, psi = self._context.rng.uniform(0, 2 * np.pi, 3)
        molecule.euler_rotate(phi, theta, psi, center=center)  # type: ignore

        return (
            molecule.positions  # type: ignore
            + self._context.rng.uniform(0, 1, 3) @ self._context.atoms.cell
            - molecule.positions.mean(axis=0)  # type: ignore
        )

    def update_indices(
        self,
        new_indices: int | IntegerArray | None = None,
        old_indices: int | IntegerArray | None = None,
    ):
        is_addition = new_indices is not None
        is_removal = old_indices is not None
        assert is_addition ^ is_removal

        if is_addition:
            self.molecule_ids[max(self.molecule_ids) + 1] = new_indices
        elif is_removal:
            for key, values in self.molecule_ids.items():
                if np.array_equal(values, old_indices):
                    del self.molecule_ids[key]
                    return
