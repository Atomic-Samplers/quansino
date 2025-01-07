"""Module for Monte Carlo contexts"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from quansino.utils.atoms import reinsert_atoms

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from numpy.random import Generator

    from quansino.moves.displacements import DisplacementMove
    from quansino.typing import IntegerArray


class Context(ABC):
    """
    Base class for Monte Carlo contexts.

    Attributes
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.

    Methods
    -------
    register_success()
        Register a successful move, saving the current state.
    register_failure()
        Register a failed move, reverting any changes made.
    """

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
        self.atoms = atoms
        self.rng = rng

    @abstractmethod
    def register_success(self) -> None: ...

    @abstractmethod
    def register_failure(self) -> None: ...


class DisplacementContext(Context):
    """
    Context for displacement moves i.e. moves that displace atoms.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.

    Attributes
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.
    selected_candidates : IntegerArray | None
        Integer labels of atoms potentially selected for displacement. Can be set to preselect specific atoms before executing a move. If None, the move selects atoms itself. Reset to None after move.
    moving_indices : IntegerArray | None
        Integer indices of atoms that are being displaced. This attribute should be set and used only in `quansino.moves.displacements.DisplacementMove.attempt_move` to inform `quansino.moves.operations.Operation` about which atoms to displace. Reset to None after move.
    moved_candidates : IntegerArray | None
        Integer labels of atoms that were displaced in the last move. Reset to None in case of a failed move. Used to revert the move.

    Methods
    -------
    register_success()
        Register a successful move, saving the current state. In this context, it saves the atoms that were displaced to `moved_candidates`. It also resets `selected_candidates` and `moving_indices`.
    register_failure()
        Register a failed move, reverting any changes made. In this context, it resets `selected_candidates`, `moving_indices`, and `moved_candidates`.

    Notes
    -----
    This class is a subclass of `Context` and should be used for displacement moves. It provides additional attributes to keep track of atoms to be displaced and atoms that were displaced.
    """

    def __init__(self, atoms: Atoms, rng: Generator) -> None:
        """Initialize the DisplacementContext object."""
        super().__init__(atoms, rng)

        self.selected_candidates: IntegerArray | None = None
        self.moving_indices: IntegerArray | None = None
        self.moved_candidates: IntegerArray | None = None

    def register_success(self) -> None:
        """Register a successful move, saving the current state."""
        self.moved_candidates = self.selected_candidates
        self.selected_candidates = None
        self.moving_indices = None

    def register_failure(self) -> None:
        """Register a failed move, reverting any changes made."""
        self.selected_candidates = None
        self.moving_indices = None
        self.moved_candidates = None


class ExchangeContext(DisplacementContext):
    """
    Context for exchange moves i.e. moves that add or remove atoms.

    Parameters
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.
    moves : dict[str, DisplacementMove]
        Dictonary of displacement moves to update when atoms are added or removed. By default, ExchangeMonteCarlo classes should add all their displacement moves here.

    Attributes
    ----------
    atoms : Atoms
        The atoms object to perform the simulation on.
    rng : Generator
        The random number generator to use.
    moves : dict[str, DisplacementMove]
        Dictonary of displacement moves to update when atoms are added or removed. By default, ExchangeMonteCarlo classes should add all their displacement moves here.
    addition_candidates : Atoms | None
        Atoms to be added in the next move. If None, the move selects atoms itself. Reset to None after move.
    deletion_candidates : IntegerArray | None
        Integer labels of atoms to be deleted in the next move. If None, the move selects atoms itself. Reset to None after move.
    last_added_indices : IntegerArray | None
        Integer indices of atoms that were added in the last move. Reset to None in case of a failed move. Used to revert the move.
    last_deleted_indices : IntegerArray | None
        Integer indices of atoms that were deleted in the last move. Reset to None in case of a failed move. Used to revert the move.
    last_deleted_atoms : Atoms | None
        Atoms that were deleted in the last move. Reset to None in case of a failed move. Used to revert the move.

    Methods
    -------
    reset()
        Reset the context by setting all attributes to None.
    register_exchange_success()
        Register a successful exchange move, saving the current state. It resets `addition_candidates` and `deletion_candidates`.
    register_exchange_failure()
        Register a failed exchange move, reverting any changes made. It reverts the move and resets all attributes.
    revert_move()
        Revert the last move made by the context.

    Notes
    -----
    This class is a subclass of `DisplacementContext` and should be used for exchange moves. It provides additional attributes to keep track of atoms to be added and removed, and atoms that were added and removed.

    `reset()` should be called before each move to ensure that the context is in a clean state. This is because the context is reused for each move, and attributes cannot be overwritten due to the fact that there are two types of moves that can be performed (addition, deletion).
    """

    def __init__(
        self, atoms: Atoms, rng: Generator, moves: dict[str, DisplacementMove]
    ) -> None:
        """Initialize the ExchangeContext object."""
        super().__init__(atoms, rng)

        self.moves: dict[str, DisplacementMove] = moves

        self.addition_candidates: Atoms | None = None
        self.deletion_candidates: IntegerArray | None = None
        self.last_added_indices: IntegerArray | None = None
        self.last_deleted_indices: IntegerArray | None = None
        self.last_deleted_atoms: Atoms | None = None

    def reset(self) -> None:
        """Reset the context by setting last move attributes to None."""
        self.last_added_indices = None
        self.last_deleted_indices = None
        self.last_deleted_atoms = None

    def register_exchange_success(self) -> None:
        """Register a successful exchange move, only resetting the candidates. The last move attributes must be saved in the move object, depending on the exchange type done."""
        self.addition_candidates = None
        self.deletion_candidates = None

    def register_exchange_failure(self) -> None:
        """Register a failed exchange move, reverting any changes made. It reverts the move and resets all attributes."""
        self.revert_move()
        self.addition_candidates = None
        self.deletion_candidates = None
        self.last_added_indices = None
        self.last_deleted_indices = None
        self.last_deleted_atoms = None

    def revert_move(self) -> None:
        """Revert the last move made by the context."""
        if self.last_added_indices is not None:
            del self.atoms[self.last_added_indices]
        elif self.last_deleted_indices is not None:
            if self.last_deleted_atoms is None:
                raise ValueError

            reinsert_atoms(
                self.atoms, self.last_deleted_atoms, self.last_deleted_indices
            )
