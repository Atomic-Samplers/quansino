"""Module for Monte Carlo moves"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ase.build import molecule

from quansino.mc.core import MonteCarloContext, MoveStore
from quansino.moves.atomic import AtomicMove
from quansino.moves.base import BaseMove
from quansino.moves.molecular import MolecularMove
from quansino.utils.atoms import insert_atoms

if TYPE_CHECKING:
    from ase.atom import Atom
    from ase.atoms import Atoms

    from quansino.typing import IntegerArray


@dataclass
class ExchangeContext(MonteCarloContext):
    moves: dict[str, MoveStore]


@dataclass
class ExchangeState:
    last_added: IntegerArray | None = None
    last_deleted: IntegerArray | None = None
    last_deleted_atoms: Atoms | None = None


class AtomicExchangeMove(AtomicMove):
    def __init__(
        self,
        exchange_atom: Atom,
        bias_towards_insert: float = 0.5,
        adding_type: str = "translation",
        move_update_to_skip: str | list[str] | None = None,
        exchange_atoms_indices: IntegerArray | None = None,
    ) -> None:
        self.exchange_atom = exchange_atom

        exchange_atoms_indices = exchange_atoms_indices or []

        super().__init__(moving_indices=exchange_atoms_indices, move_type=adding_type)

        self.bias_towards_insert = bias_towards_insert

        if move_update_to_skip is None:
            self.move_update_to_skip = []

    def __call__(self) -> bool | list[bool]:
        self.exchange_state = ExchangeState()

        if self.context.rng.random() < self.bias_towards_insert:
            self.context.atoms.append(self.exchange_atom)

            self.state.to_move = [-1]

            if not BaseMove.__call__(self):
                del self.context.atoms[-1]
                return False

            self.exchange_state.last_added = np.arange(len(self._context.atoms))[
                self.state.moved
            ]
        else:
            if len(self.moving_indices) == 0:
                return False

            if self.state.to_move is None:
                self.state.to_move = self.context.rng.choice(self.moving_indices)

            self.state.to_move = np.reshape(self.state.to_move, -1)

            self.exchange_state.last_deleted = np.arange(len(self._context.atoms))[
                self.state.to_move
            ]
            self.exchange_state.last_deleted_atoms = self._context.atoms[
                self.state.to_move
            ]  # type: ignore

            del self.context.atoms[self.state.to_move]

            self.state.to_move = None

        return True

    @property
    def move_update_to_skip(self) -> list[str]:
        return self._move_update_to_skip

    @move_update_to_skip.setter
    def move_update_to_skip(self, move_update_to_skip: str | list[str]) -> None:
        if isinstance(move_update_to_skip, str):
            self._move_update_to_skip = [move_update_to_skip]
        else:
            self._move_update_to_skip = move_update_to_skip

    @property
    def context(self) -> ExchangeContext:
        return self._context

    @context.setter
    def context(self, context: MonteCarloContext) -> None:
        if not isinstance(context, ExchangeContext):
            raise AttributeError("ExchangeMoves require a ExchangeContext")

        self._context = context

    def update_moves(self) -> None:
        for name, move_store in self._context.moves.items():
            if name not in self.move_update_to_skip:
                move_store.move.update_indices(
                    self.exchange_state.last_added, self.exchange_state.last_deleted
                )

    def revert_move(self) -> None:
        if self.exchange_state.last_added is not None:
            del self.context.atoms[self.exchange_state.last_added]
        elif self.exchange_state.last_deleted is not None:
            if self.exchange_state.last_deleted_atoms is None:
                raise ValueError("No atoms to put back, this should not happen")

            insert_atoms(
                self._context.atoms,
                self.exchange_state.last_deleted_atoms,
                self.exchange_state.last_deleted,
            )


class MolecularExchangeMove(MolecularMove, AtomicExchangeMove):
    def __init__(
        self,
        exchange_atoms: Atoms | str,
        bias_towards_insert: float = 0.5,
        adding_type: str = "translation_rotation",
        move_update_to_skip: str | list[str] | None = None,
        exchange_molecule_indices: IntegerArray | dict[int, IntegerArray] | None = None,
    ) -> None:
        if isinstance(exchange_atoms, str):
            exchange_atoms = molecule(exchange_atoms)

        self.exchange_atoms = exchange_atoms

        exchange_molecule_indices = exchange_molecule_indices or {}

        super().__init__(molecule_ids=exchange_molecule_indices, move_type=adding_type)

        self.bias_towards_insert = bias_towards_insert

        if move_update_to_skip is None:
            self.move_update_to_skip = []

    def __call__(self) -> bool | list[bool]:
        self.exchange_state = ExchangeState()

        if self.context.rng.random() < self.bias_towards_insert:
            self.context.atoms.extend(self.exchange_atoms)

            self.state.to_move = np.arange(
                len(self.context.atoms) - len(self.exchange_atoms),
                len(self.context.atoms),
            )

            if not BaseMove.__call__(self):
                del self.context.atoms[-len(self.exchange_atoms) :]
                return False

            self.exchange_state.last_added = self.state.to_move
        else:
            ids_list = list(self.molecule_ids.keys())

            if len(ids_list) == 0:
                return False

            if self.state.to_move is None:
                self.state.to_move = self.molecule_ids[
                    self.context.rng.choice(ids_list)
                ]

            mask = np.isin(np.arange(len(self._context.atoms)), self.state.to_move)
            self.exchange_state.last_deleted = np.where(mask)[0]
            self.exchange_state.last_deleted_atoms = self.context.atoms[mask]  # type: ignore

            del self.context.atoms[mask]

        return True
