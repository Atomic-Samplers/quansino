from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quansino.moves.atomic import AtomicMove
from quansino.moves.base import BaseState

if TYPE_CHECKING:
    from quansino.typing import Displacement, IntegerArray


@dataclass
class MolecularState(BaseState):
    molecules_to_move: IntegerArray | None = None


class MolecularMove(AtomicMove):
    def __init__(
        self,
        molecule_ids: IntegerArray | dict[int, IntegerArray],
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

        self.state = MolecularState()

    def __call__(self) -> bool | list[bool]:
        if len(self.molecule_ids) == 0:
            return False

        molecule_ids_list = list(self.molecule_ids.keys())

        molecules_moving = (
            len(molecule_ids_list)
            if len(molecule_ids_list) < self.molecule_moving_per_step
            else self.molecule_moving_per_step
        )

        if self.state.molecules_to_move is None:
            self.state.molecules_to_move = self.context.rng.choice(
                molecule_ids_list, size=molecules_moving, replace=False
            )

        successes = []
        for molecule in self.state.molecules_to_move:
            self.state.to_move = self.molecule_ids[molecule]
            successes.append(super().__call__())

        self.state.molecules_to_move = None
        return successes

    @property
    def molecule_ids(self) -> dict[int, IntegerArray]:
        return self._molecule_ids

    @molecule_ids.setter
    def molecule_ids(
        self, molecule_ids: IntegerArray | dict[int, IntegerArray]
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

        return self._context.rng.uniform(
            0, 1, 3
        ) @ self._context.atoms.cell - molecule.positions.mean(  # type: ignore
            axis=0
        )

    def update_indices(
        self, to_add: IntegerArray | None = None, to_remove: IntegerArray | None = None
    ):
        is_addition = to_add is not None
        is_removal = to_remove is not None

        if not (is_addition ^ is_removal):
            raise ValueError("Either new_indices or old_indices should be provided")

        if is_addition:
            self.molecule_ids[max(self.molecule_ids) + 1] = to_add
        elif is_removal:
            for key, values in self.molecule_ids.items():
                if np.array_equal(values, to_remove):
                    del self.molecule_ids[key]
                    return
