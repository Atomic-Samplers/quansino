from __future__ import annotations

from sys import stdout
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.atoms import Atoms
from ase.build import molecule
from ase.cluster import Octahedron
from ase.constraints import FixAtoms
from ase.optimize import BFGSLineSearch
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from quansino.mc.criteria import GrandCanonicalCriteria
from quansino.mc.gcmc import GrandCanonical
from quansino.moves.displacement import DisplacementMove
from quansino.moves.exchange import ExchangeMove

if TYPE_CHECKING:
    from quansino.mc.contexts import DisplacementContext

atoms = Octahedron("Pt", 4)
atoms.center(10)

com = atoms.get_center_of_mass()

o2_molecule = molecule("O2", vacuum=10.0)

device = "cpu"
orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
calc = ORBCalculator(orbff, device=device)
atoms.calc = calc
o2_molecule.calc = calc

o2_molecule_bfgs = BFGSLineSearch(o2_molecule)
o2_molecule_bfgs.run(fmax=0.01, steps=100)

energy = o2_molecule.get_potential_energy()

chemical_potential = 0.5 * (0.633926342106 + energy - 0.1)  # From NIST, ZPE adjusted

gas = Atoms("O")

mc = GrandCanonical(
    atoms,
    exchange_atoms=gas,
    chemical_potential=chemical_potential,
    temperature=298.15,
    number_of_exchange_particles=0,
    max_cycles=len(atoms),
    default_displacement_move=DisplacementMove(labels=np.arange(len(atoms))),
    default_exchange_move=ExchangeMove(labels=np.full(len(atoms), -1)),
    seed=42,
    logging_interval=1,
    logfile=stdout,
    trajectory="orb-v3_grand_canonical_pt_np.xyz",
    logging_mode="w",
)

mc.moves["default_exchange_move"].probability = 0.1
mc.moves["default_displacement_move"].probability = 0.9

mc.run(100)


def check_move(context: DisplacementContext) -> bool:
    """
    Check if the move can be performed based on the center of mass distance.

    Returns
    -------
    bool
        True if the move can be performed, False otherwise.
    """
    atoms_about_to_be_placed = atoms[context._moving_indices]
    com_distance = (
        atoms_about_to_be_placed.positions  # type: ignore[shape]
        - atoms[atoms.symbols == "Pt"].get_center_of_mass()  # type: ignore[shape]
    )

    return not bool(np.any(np.linalg.norm(com_distance, axis=1) > 8.0))


mc.moves["default_exchange_move"].move.check_move = check_move

mc.accessible_volume = np.pi * 3 / 4 * (8.0**3)

mc.run(100)


class GrandCanonicalMinimizationCriteria(GrandCanonicalCriteria):
    """
    Acceptance criteria for Monte Carlo moves in the grand canonical (μVT) ensemble with minimization.

    This criteria extends the GrandCanonicalCriteria to include minimization of the system's energy.
    """

    def __init__(self, optimizer: BFGSLineSearch, run_kwargs: dict[str, Any]) -> None:
        """
        Initialize the GrandCanonicalMinimizationCriteria.
        """
        self.optimizer = optimizer
        self.run_kwargs = run_kwargs

    def evaluate(self, *args, **kwargs) -> bool:
        """
        Evaluate the acceptance criteria for a Monte Carlo move with minimization.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        self.optimizer.reset()

        self.optimizer.run(**self.run_kwargs)
        return super().evaluate(*args, **kwargs)


mc.moves["default_exchange_move"].criteria = GrandCanonicalMinimizationCriteria(
    optimizer=BFGSLineSearch(atoms, logfile=None),  # type: ignore[ase]
    run_kwargs={"fmax": 0.05, "steps": 100},
)

mc.run(100)


class FixedGrandCanonicalMinimizationCriteria(GrandCanonicalCriteria):
    """
    Acceptance criteria for Monte Carlo moves in the grand canonical (μVT) ensemble with minimization.

    This criteria extends the GrandCanonicalCriteria to include minimization of the system's energy with fixed atoms.
    """

    def __init__(self, optimizer: BFGSLineSearch, run_kwargs: dict[str, Any]) -> None:
        """
        Initialize the GrandCanonicalMinimizationCriteria.
        """
        self.optimizer = optimizer
        self.run_kwargs = run_kwargs

    def evaluate(self, context) -> bool:
        """
        Evaluate the acceptance criteria for a Monte Carlo move with fixed minimization.

        Parameters
        ----------
        context : ExchangeContext
            The context for the Monte Carlo move.

        Returns
        -------
        bool
            True if the move is accepted, False otherwise.
        """
        self.optimizer.reset()

        mask = np.full(len(context.atoms), True, dtype=bool)
        mask[context._moving_indices] = False

        context.atoms.set_constraint(FixAtoms(mask=mask))
        self.optimizer.run(**self.run_kwargs)
        context.atoms.set_constraint(None)

        return super().evaluate(context)


mc.moves["default_exchange_move"].criteria = FixedGrandCanonicalMinimizationCriteria(
    optimizer=BFGSLineSearch(atoms, logfile="-"),
    run_kwargs={"fmax": 0.05, "steps": 100},
)

mc.run(100)
