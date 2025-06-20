"General purpose utils module."

from __future__ import annotations

from quansino.registry import register_class
from quansino.utils.atoms import has_constraint, reinsert_atoms, search_molecules
from quansino.utils.moves import MoveStorage
from quansino.utils.strings import get_auto_header_format

__all__ = [
    "MoveStorage",
    "get_auto_header_format",
    "has_constraint",
    "reinsert_atoms",
    "search_molecules",
]

utils_registry = {"MoveStorage": MoveStorage}

register_class(MoveStorage, "MoveStorage")
