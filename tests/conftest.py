from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.eam import EAM
from ase.calculators.emt import EMT
from numpy.random import Generator as RNG
from numpy.random import default_rng

from quansino.integrators.core import BaseIntegrator
from quansino.mc.contexts import Context
from quansino.mc.criteria import BaseCriteria
from quansino.mc.fbmc import AdaptiveForceBias
from quansino.moves.core import BaseMove
from quansino.operations.core import BaseOperation


@pytest.fixture(name="eam_calc")
def eam_calc_fixture() -> EAM:
    path = Path(__file__).parent
    return EAM(
        potential=str(
            Path(path, ".potentials/cu_eam_mishin2001.alloy").resolve().expanduser()
        )
    )


@pytest.fixture(name="bulk_small")
def bulk_small_fixture(eam_calc) -> Atoms:
    atoms = bulk("Cu", cubic=True)
    atoms.calc = eam_calc
    return atoms


@pytest.fixture(name="bulk_medium")
def bulk_medium_fixture(eam_calc) -> Atoms:
    atoms = bulk("Cu", cubic=True) * (3, 3, 3)
    atoms.calc = eam_calc
    return atoms


@pytest.fixture(name="bulk_large")
def bulk_large_fixture(eam_calc) -> Atoms:
    atoms = bulk("Cu", cubic=True) * (6, 6, 6)
    atoms.calc = eam_calc
    return atoms


@pytest.fixture(name="single_atom")
def single_atom_fixture() -> Atoms:
    atom = Atoms("H", positions=[[0, 0, 0]])
    atom.calc = EMT()
    return atom


@pytest.fixture(name="empty_atoms")
def empty_atoms_fixture() -> Atoms:
    empty_atoms = Atoms()
    empty_atoms.calc = EMT()
    return empty_atoms


@pytest.fixture(name="rng")
def rng_fixture() -> RNG:
    return np.random.default_rng()


class DummyOperation(BaseOperation):
    """A dummy operation that sets all atom positions to zero."""

    def __init__(self, name: str = "DummyOperation") -> None:
        super().__init__()
        self.name = name
        self.move_count = 0

    def calculate(self, context: Context):
        context.atoms.set_positions(np.zeros((len(context.atoms), 3)))
        self.move_count += 1


class DummyIntegrator(BaseIntegrator):
    """A dummy integrator that does nothing."""

    def __init__(self, name: str = "DummyIntegrator") -> None:
        super().__init__()
        self.name = name

    def integrate(self, context: Context):
        """Perform a dummy integration step."""


class DummyMove(BaseMove[BaseOperation, Context]):
    """A dummy move that applies a dummy operation."""

    def __init__(
        self, operation: BaseOperation, apply_constraints: bool = True
    ) -> None:
        super().__init__(operation, apply_constraints)

    def check_move(self, context) -> bool:
        return True

    @property
    def default_operation(self) -> DummyOperation:
        return DummyOperation()

    def __call__(self, context):
        """
        Apply the move to the context.

        Parameters
        ----------
        context : Context
            The context in which the move is applied.
        """
        self.operation.calculate(context)
        return True


class DummyContext(Context):
    """A dummy context that mimics the Context class."""

    def __init__(self) -> None:
        super().__init__(Atoms(), default_rng())


class DummyCalculator:
    """A dummy calculator that returns a fixed potential energy."""

    def __init__(self, dummy_value: float = -1.0) -> None:
        self.dummy_value = dummy_value
        self.results = {}

    def get_potential_energy(self, *args, **kwargs) -> float:
        self.results["energy"] = self.dummy_value
        return self.dummy_value


class EMTUncertaintyReadyCalculator(EMT):
    """A dummy calculator that returns a fixed potential energy with uncertainties."""

    def __init__(self, len_atoms: int, **kwargs) -> None:
        self.len_atoms = len_atoms
        super().__init__(**kwargs)

    def get_property(self, *args, **kwargs):
        result = super().get_property(*args, **kwargs)

        self.results[AdaptiveForceBias.energies_variance_keyword] = (
            default_rng().random((5,)) * 0.1
        )
        self.results[AdaptiveForceBias.forces_variance_keyword] = (
            default_rng().random((5, self.len_atoms, 3)) * 0.1
        )

        return result


class DummyCriteria(BaseCriteria):
    """A dummy criteria that always returns True."""

    @staticmethod
    def evaluate(context) -> bool:
        return True


class DummySimulation:
    """A dummy simulation class to mimic a Monte Carlo simulation."""

    def __init__(self, atoms, context, moves) -> None:
        self.atoms = atoms
        self.context = context
        self.moves = moves


class DummyStream:
    """A dummy stream class that does nothing on read or write."""

    def __init__(self, *args, **kwargs):
        self.closed = False

    def read(self, *args, **kwargs): ...

    def write(self, *args, **kwargs): ...
