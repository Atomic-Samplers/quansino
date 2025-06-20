from __future__ import annotations

from sys import stdout

import numpy as np
from ase.atoms import Atoms
from ase.build import molecule
from ase.cluster import Octahedron
from ase.optimize import BFGSLineSearch
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from quansino.mc.gcmc import GrandCanonical
from quansino.moves.displacement import DisplacementMove
from quansino.moves.exchange import ExchangeMove

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
