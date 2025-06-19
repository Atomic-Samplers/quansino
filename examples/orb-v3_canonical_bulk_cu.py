from __future__ import annotations

from sys import stdout

import numpy as np
from ase.build import bulk
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from quansino.mc.canonical import Canonical
from quansino.moves.displacement import DisplacementMove

device = "cpu"
orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
calc = ORBCalculator(orbff, device=device)
atoms = bulk("Cu", "fcc", cubic=True)

atoms.calc = calc
atoms.get_potential_energy()

mc = Canonical(
    atoms,
    temperature=500,
    max_cycles=len(atoms),
    default_displacement_move=DisplacementMove(labels=np.arange(len(atoms))),
    seed=42,
    logging_interval=1,
    logfile=stdout,
    trajectory="orb-v3_canonical_bulk_cu.xyz",
    logging_mode="w",
)

mc.run(100)
