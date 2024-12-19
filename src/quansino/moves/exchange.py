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

if TYPE_CHECKING:
    from ase.atom import Atom
    from ase.atoms import Atoms

    from quansino.typing import IntegerArray


@dataclass
class ExchangeContext(MonteCarloContext):
    moves: dict[str, MoveStore]


@dataclass
class AtomicExchangeState:
    last_added: int | None = None
    last_deleted: int | IntegerArray | None = None
    last_deleted_atoms: Atoms | None = None


@dataclass
class MolecularExchangeState:
    last_added: IntegerArray | None = None
    last_deleted: IntegerArray | None = None
    last_deleted_atoms: Atoms | None = None


class MolecularExchangeMove(MolecularMove):
    def __init__(
        self,
        exchange_atoms: Atoms | str,
        bias_towards_insert: float = 0.5,
        adding_type: str = "translation_rotation",
        moves_to_update: bool | str | list[str] = True,
        exchange_molecule_indices: (
            IntegerArray | dict[int, int | IntegerArray] | None
        ) = None,
    ) -> None:
        if isinstance(exchange_atoms, str):
            exchange_atoms = molecule(exchange_atoms)

        self.exchange_atoms = exchange_atoms

        self.number_of_exchange_atoms = len(self.exchange_atoms)

        exchange_molecule_indices = exchange_molecule_indices or {}

        super().__init__(
            molecule_ids=exchange_molecule_indices,
            delta=0.0,
            molecule_moving_per_step=1,
            move_type=adding_type,
            apply_constraints=False,
        )

        self.bias_towards_insert = bias_towards_insert

        self.moves_to_update = moves_to_update

    def __call__(self) -> bool | list[bool]:
        self.exchange_state = MolecularExchangeState()

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

    @property
    def moves_to_update(self) -> list[str]:
        return self._move_to_update

    @moves_to_update.setter
    def moves_to_update(self, moves_to_update: bool | str | list[str]) -> None:
        if isinstance(moves_to_update, str):
            self._move_to_update = [moves_to_update]
        elif isinstance(moves_to_update, bool):
            self._move_to_update = list(self._context.moves) if moves_to_update else []
        else:
            self._move_to_update = moves_to_update

    @property
    def context(self) -> ExchangeContext:
        return self._context

    @context.setter
    def context(self, context: MonteCarloContext) -> None:
        if not isinstance(context, ExchangeContext):
            raise AttributeError("ExchangeMoves require a ExchangeContext")

        self._context = context

    def update_moves(self) -> None:
        (
            self._context.moves[i].move.update_indices(
                self.exchange_state.last_added, self.exchange_state.last_deleted
            )
            for i in self.moves_to_update
        )

    def revert_move(self) -> None:
        is_addition = self.exchange_state.last_added is not None
        is_removal = self.exchange_state.last_deleted is not None
        assert is_addition ^ is_removal

        if is_addition:
            del self.context.atoms[self.exchange_state.last_added]
        elif is_removal:
            self.context.atoms.extend(self.exchange_state.last_deleted_atoms)


class AtomicExchangeMove(AtomicMove):
    def __init__(
        self,
        exchange_atom: Atom,
        bias_towards_insert: float = 0.5,
        adding_type: str = "translation_rotation",
        moves_to_update: bool | str | list[str] = True,
        exchange_atoms_indices: IntegerArray | None = None,
    ) -> None:
        self.exchange_atom = exchange_atom

        self.number_of_exchange_atoms = 1

        exchange_atoms_indices = exchange_atoms_indices or []

        super().__init__(
            moving_indices=exchange_atoms_indices,
            delta=0.0,
            moving_per_step=1,
            move_type=adding_type,
            apply_constraints=False,
        )

        self.bias_towards_insert = bias_towards_insert

        self.moves_to_update = moves_to_update

    def __call__(self) -> bool | list[bool]:
        self.exchange_state = AtomicExchangeState()

        if self.context.rng.random() < self.bias_towards_insert:
            self.context.atoms.append(self.exchange_atom)

            self.state.to_move = -1

            if not BaseMove.__call__(self):
                del self.context.atoms[-1]
                return False

            self.exchange_state.last_added = self.state.to_move
        else:
            if len(self.moving_indices) == 0:
                return False

            if self.state.to_move is None:
                self.state.to_move = self.context.rng.choice(self.moving_indices)

            self.exchange_state.last_deleted = self.state.to_move
            self.exchange_state.last_deleted_atoms = self._context.atoms[
                [self.state.to_move]
            ]  # type: ignore

            del self.context.atoms[self.state.to_move]

        return True

    @property
    def moves_to_update(self) -> list[str]:
        return self._move_to_update

    @moves_to_update.setter
    def moves_to_update(self, moves_to_update: bool | str | list[str]) -> None:
        if isinstance(moves_to_update, str):
            self._move_to_update = [moves_to_update]
        elif isinstance(moves_to_update, bool):
            self._move_to_update = list(self._context.moves) if moves_to_update else []
        else:
            self._move_to_update = moves_to_update

    @property
    def context(self) -> ExchangeContext:
        return self._context

    @context.setter
    def context(self, context: MonteCarloContext) -> None:
        if not isinstance(context, ExchangeContext):
            raise AttributeError("ExchangeMoves require a ExchangeContext")

        self._context = context

    def update_moves(self) -> None:
        (
            self._context.moves[i].move.update_indices(
                self.exchange_state.last_added, self.exchange_state.last_deleted
            )
            for i in self.moves_to_update
        )

    def revert_move(self) -> None:
        is_addition = self.exchange_state.last_added is not None
        is_removal = self.exchange_state.last_deleted is not None
        assert is_addition ^ is_removal

        if is_addition:
            del self.context.atoms[self.exchange_state.last_added]
        elif is_removal:
            self.context.atoms.extend(self.exchange_state.last_deleted_atoms)
