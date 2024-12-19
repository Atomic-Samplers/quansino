"""Module for Monte Carlo moves"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Callable, ClassVar

    from quansino.mc.core import MonteCarloContext
    from quansino.typing import IntegerArray


@dataclass
class BaseState:
    to_move: int | IntegerArray | None = None
    moved: int | IntegerArray | None = None


class BaseMove:
    REQUIRED_ATTRIBUTES: ClassVar[set[str]] = {"atoms", "rng"}
    MAX_ATTEMPTS: ClassVar[int] = 10000

    def __init__(
        self,
        moving_indices: IntegerArray | None,
        moving_per_step: int,
        move_type: str,
        move_functions: dict[str, Callable],
        apply_constraints=True,
    ):
        self.moving_indices = [] if moving_indices is None else moving_indices

        self.moving_per_step = moving_per_step

        self.move_type = move_type
        self.move_functions = move_functions

        self.apply_constraints = apply_constraints

        self.state = BaseState()

    def __call__(self) -> bool | list[bool]:
        atoms = self._context.atoms
        old_positions = atoms.get_positions()

        if self.state.to_move is None:
            self.state.to_move = self._context.rng.choice(
                self.moving_indices,
                size=self.moving_per_step,
                replace=False,  # type: ignore
            )

        for _ in range(self.MAX_ATTEMPTS):
            translation = np.full((len(atoms), 3), 0.0)
            translation[self.state.to_move] = self.move_functions[self.move_type]()

            atoms.set_positions(
                atoms.positions + translation, apply_constraint=self.apply_constraints
            )

            if self.check_move():
                self.state.moved = self.state.to_move
                self.state.to_move = None
                return True

            atoms.positions = old_positions

        self.state.moved = None
        self.state.to_move = None
        return False

    @property
    def context(self) -> MonteCarloContext:
        return self._context

    @context.setter
    def context(self, context: MonteCarloContext) -> None:
        for attr in self.REQUIRED_ATTRIBUTES:
            if not hasattr(context, attr):
                raise AttributeError(f"Missing attribute {attr} in context.")

        self.moving_indices = (
            np.arange(len(context.atoms))
            if self.moving_indices is None
            else self.moving_indices
        )

        self._context = context

    def check_move(self) -> bool:
        return True

    def update_indices(
        self,
        new_indices: int | IntegerArray | None = None,
        old_indices: int | IntegerArray | None = None,
    ):
        is_addition = new_indices is not None
        is_removal = old_indices is not None
        assert is_addition ^ is_removal

        if is_addition:
            self.moving_indices = np.append(self.moving_indices, new_indices)
        elif is_removal:
            self.moving_indices = np.delete(self.moving_indices, old_indices)
