from __future__ import annotations

from sys import stdout

import numpy as np
from ase.build import bulk
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from quansino.mc.criteria import IsobaricCriteria
from quansino.mc.isobaric import Isobaric
from quansino.moves.cell import CellMove
from quansino.moves.displacement import DisplacementMove
from quansino.protocols import Criteria, Move

device = "cpu"
orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
calc = ORBCalculator(orbff, device=device)
atoms = bulk("Cu", "fcc", cubic=True)

atoms.calc = calc

mc = Isobaric[Move, Criteria](
    atoms,
    temperature=500,
    max_cycles=len(atoms),
    default_displacement_move=DisplacementMove(labels=np.arange(len(atoms))),
    default_cell_move=CellMove(),
    seed=42,
    logging_interval=1,
    logfile=stdout,
    trajectory="orb-v3_isobaric_bulk_cu.xyz",
    logging_mode="w",
)

for steps in mc.irun(100):
    for _ in steps:
        pass


del mc.moves["default_displacement_move"]
del mc.moves["default_cell_move"]


hybrid_move = DisplacementMove(labels=np.arange(len(atoms))) * len(atoms) + CellMove()
mc.max_cycles = 1

mc.add_move(hybrid_move, name="hybrid_move", criteria=IsobaricCriteria())

for _ in mc.srun(100):
    pass
