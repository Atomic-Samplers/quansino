"""Module for Monte Carlo moves"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from quansino.moves.base import BaseMove, BaseState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from quansino.typing import Displacement, IntegerArray


class AtomicMove(BaseMove):
    """Base class for Monte Carlo moves

    Parameters
    ----------
    moving_indices
        The indices of the atoms to move.
    delta
        The maximum distance to move the atoms.
    moving_per_step
        The number of atoms to move per step.
    move_type
        The type of move to perform. Can be "box" or "ball".

    Attributes
    ----------
    REQUIRED_ATTRIBUTES : ClassVar[set[str]]
        The required attributes for the Monte Carlo context.
    MAX_ATTEMPTS : ClassVar[int]
        The maximum number of attempts to make a move.
    delta: float
        The maximum distance to move the atoms.
    moving_indices : IntegerArray
        The indices of the atoms to move.
    moving_per_step : int
        The number of atoms to move per step.

    This class is the core of Monte Carlo moves in `quansino`. All Monte Carlo moves should inherit
    from this class and implement the `__call__` method. The `__call__` method should perform the
    move and update the positions of the atoms in the context.

    These classes rely on their attributes being set by the Monte Carlo context to interact and update
    attributes of the Monte Carlo simulation and `atoms` object. The required attributes are defined in
    the `REQUIRED_ATTRIBUTES` class variable. Base class will only rely on the `atoms` and `rng` attributes, and should work with all Monte Carlo classes. If a custom move requires additional attributes, they should be added to the `REQUIRED_ATTRIBUTES` set and be linked to a Monte Carlo class that possesses them.

    Basic customization can be performed by manually adding move functions to the `move_functions` dictionary. The `move_functions` dictionary should contain the move type as the key and the move function as the value. The move function should return a `Displacement` for the atoms to move.

    Other attributes can be modified during execution to obtain highly customized moves. This can easily done between steps by using `irun`:

    ```python
    for step in simulation.irun(step=1000):
        if simulation.nsteps > 1000:
            simulation.moves["default"].delta = 0.1
        if simulation.nsteps % 500 == 0:
            simulation.moves["default"].move_type = "ball"
        else:
            simulation.moves["default"].move_type = "box"
    ```

    Moves should be added to a simulation by using the `add_move` method of the `MonteCarlo` class. Note that a move can only be added to one simulation at a time. Nothing prevents to use multiple moves in the same simulation. If moves were to be shared between simulations, a new instance of the move should be created.

    If multiple move share the same interval,

    The `check_move` method can be overridden to add additional checks to the move. This method should return a boolean indicating if the move is valid.
    """

    def __init__(
        self,
        delta: float = 1.0,
        moving_indices: IntegerArray | None = None,
        moving_per_step: int = 1,
        move_type: str = "box",
        apply_constraints: bool = True,
    ) -> None:
        self.delta = delta

        move_functions = {
            "box": self.box,
            "ball": self.ball,
            "sphere": partial(self.ball, r=self.delta),
            "translation": self.translation,
        }

        super().__init__(
            moving_indices,
            moving_per_step,
            move_type,
            move_functions,
            apply_constraints=apply_constraints,
        )

        self.state = BaseState()

    def box(self) -> Displacement:
        return self.context.rng.uniform(
            -self.delta, self.delta, size=(self.moving_per_step, 3)
        )

    def ball(self, r: float | NDArray[np.floating] | None = None) -> Displacement:
        """Move atoms in a ball"""
        if r is None:
            r = self.context.rng.uniform(0, self.delta, size=self.moving_per_step)

        phi = self.context.rng.uniform(0, 2 * np.pi, size=self.moving_per_step)
        cos_theta = self.context.rng.uniform(-1, 1, size=self.moving_per_step)
        sin_theta = np.sqrt(1 - cos_theta**2)

        return np.column_stack(
            (r * sin_theta * np.cos(phi), r * sin_theta * np.sin(phi), r * cos_theta)
        )

    def translation(self) -> Displacement:
        return (
            self._context.rng.uniform(0, 1, (len(self.moving_indices), 3))
            @ self._context.atoms.cell.array[None, :]
            - self._context.atoms.positions[self.state.to_move]
        )
