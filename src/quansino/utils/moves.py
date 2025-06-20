"""Module for storing and managing moves with their acceptance criteria."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from quansino.protocols import Criteria, Move
from quansino.registry import get_typed_class

if TYPE_CHECKING:
    from typing import Any


@dataclass(slots=True)
class MoveStorage[MoveType: Move, CriteriaType: Criteria]:
    """
    Dataclass to store moves and their acceptance criteria.

    This generic dataclass pairs moves with their corresponding acceptance criteria and execution parameters for Monte Carlo simulations.

    Attributes
    ----------
    move : MoveType
        The move object implementing the Move protocol.
    criteria : CriteriaType
        The acceptance criteria object implementing the Criteria protocol.
    interval : int
        The interval at which the move is selected.
    probability : float
        The probability of the move being selected.
    minimum_count : int
        The minimum number of times the move must be performed in a cycle.
    """

    move: MoveType
    criteria: CriteriaType
    interval: int
    probability: float
    minimum_count: int

    def to_dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the MoveStorage object.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the MoveStorage object.
        """
        return {
            "name": self.__class__.__name__,
            "kwargs": {
                "move": self.move.to_dict(),
                "criteria": self.criteria.to_dict(),
                "interval": self.interval,
                "probability": self.probability,
                "minimum_count": self.minimum_count,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Load the MoveStorage object from a dictionary.
        This method is used to restore the state of the MoveStorage object from a saved state.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary representation of the MoveStorage object.
        """
        data = deepcopy(data)

        kwargs = data["kwargs"]

        move_data = kwargs["move"]
        move_class: type[MoveType] = get_typed_class(move_data["name"], Move)
        kwargs["move"] = move_class.from_dict(move_data)

        criteria_data = kwargs["criteria"]
        criteria_class: type[CriteriaType] = get_typed_class(
            criteria_data["name"], Criteria
        )
        kwargs["criteria"] = criteria_class.from_dict(criteria_data)

        return cls(**kwargs)

    def __repr__(self) -> str:
        """
        Return a string representation of the `MoveStorage` object.

        Returns
        -------
        str
            A string representation of the `MoveStorage` object.
        """
        return f"{self.__class__.__name__}(move={self.move}, criteria={self.criteria}, interval={self.interval}, probability={self.probability}, minimum_count={self.minimum_count})"
