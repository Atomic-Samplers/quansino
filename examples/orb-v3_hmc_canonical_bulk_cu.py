from __future__ import annotations

from sys import stdout

from ase.cluster import Octahedron
from ase.constraints import FixCom
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from quansino.integrators.displacement import Verlet
from quansino.mc.canonical import HamiltonianCanonical
from quansino.moves.displacement import HamiltonianDisplacementMove

atoms = Octahedron("Pt", 4)

device = "cpu"
orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
calc = ORBCalculator(orbff, device=device)
# atoms = bulk("Cu", "fcc", cubic=True)

atoms.calc = calc
atoms.get_potential_energy()

atoms.constraints = [FixCom()]

mc = HamiltonianCanonical(
    atoms,
    temperature=500,
    max_cycles=1,
    default_displacement_move=HamiltonianDisplacementMove(
        operation=Verlet(dt=5.0, max_steps=5)
    ),
    seed=42,
    logging_interval=1,
    logfile=stdout,
    trajectory="orb-v3_hmc_canonical_bulk_cu.xyz",
    logging_mode="w",
)

mc.run(1000)
