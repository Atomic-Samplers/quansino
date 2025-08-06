This page contains examples of how to use [`quansino`](https://github.com/Atomic-Samplers/quansino) to run various Monte Carlo simulations.

## orb-v3 examples

### Cu-bulk (fcc) canonical ensemble

Below is an example of how to run a canonical ensemble simulation for a bulk copper (Cu) system using the `orb-v3` MLIP with the [`quansino`](https://github.com/Atomic-Samplers/quansino) package. The simulation is carried out at 500 K, performing displacement moves on all the atoms in the system. The trajectory is saved to an XYZ file, logging is directly sent to standard output, and the simulation is seeded for reproducibility.

```python
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

mc.run(1000)
```

### Cu-bulk (fcc) NPT ensemble

Below is an example of how to run an NPT ensemble simulation for a bulk copper (Cu) system using the `orb-v3` MLIP with the `quansino` package. The simulation is performed at 500 K and 0 atm, with various moves applied to the system.

```python
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
```

### Pt nanocluster (fcc) GCMC ensemble, oxygen adsorption

Below is an example of how to run a Grand Canonical Monte Carlo (GCMC) simulation for a platinum (Pt) nanocluster system using the `orb-v3` MLIP with the `quansino` package. The simulation is performed at 298.15 K, with oxygen exchange moves applied to the system.

```python
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
```

Running the simulation will result in the following logger output:

```
Step Epot[eV] AcptRate Natoms
0 -218.0509 0.000 44
1 -218.1546 0.614 44
2 -218.5050 0.636 44
3 -218.5735 0.614 44
4 -218.3855 0.659 44
5 -218.5388 0.636 44
6 -218.5623 0.636 44
7 -218.7438 0.795 44
8 -218.7147 0.500 44
9 -218.7750 0.614 44
10 -218.9371 0.568 44
11 -218.9552 0.591 44
12 -219.1030 0.614 44
13 -219.1480 0.614 44
14 -219.3729 0.614 44
15 -219.2637 0.500 44
16 -219.3318 0.545 44
```

The logger shows that the acceptance rate is around 0.6, which is reasonable, but the number of atoms remains constant. This means that most accepted moves are displacement moves, and the exchange move is not being accepted. This is because we attempt to place oxygen atoms everywhere in the unit cell, which is not desirable.

Instead, we want to attempt moves only close to the nanoparticle where the atoms can bind and have enough excess chemical potential to be accepted. To do this, we can customize the `check_move` attribute of the `ExchangeMove` to only accept moves that are close to the nanoparticle. Below is an example of how to do this:

```python
...


def check_move(context: ExchangeContext) -> bool:
    """
    Check if the move can be performed based on the center of mass distance.

    Returns
    -------
    bool
        True if the move can be performed, False otherwise.
    """
    atoms_about_to_be_placed = atoms[mc.context._moving_indices]
    com_distance = (
        atoms_about_to_be_placed.positions  # type: ignore[shape]
        - atoms[atoms.symbols == "Pt"].get_center_of_mass()  # type: ignore[shape]
    )

    return not bool(np.any(np.linalg.norm(com_distance, axis=1) > 8.0))


mc.moves["default_exchange_move"].move.check_move = check_move
```

In this way, the exchange move will attempt to place oxygen atoms only within a certain distance. If the distance is too large, it will retry a placement. This is done `move.max_attempts` times, which is set to 10,000 by default. If the move can't find a suitable position after all attempts, it will be rejected.

We also change the `accessible_volume` attribute of the `GrandCanonical` class to reflect the volume where the oxygen atoms can be placed. This is important for the acceptance rate calculation.

```python
mc.accessible_volume = np.pi * 3 / 4 * (8.0**3)
```

After these changes, we can run the simulation again and get the following output:

```
Step Epot[eV] AcptRate Natoms
0 -218.0509 0.000 44
1 -218.0484 0.636 44
2 -218.1835 0.636 44
3 -218.3500 0.636 44
4 -218.3445 0.545 44
5 -218.3713 0.636 44
6 -218.5544 0.432 44
7 -223.5220 0.636 45
8 -223.6393 0.545 45
9 -223.6890 0.591 45
10 -223.7481 0.545 45
11 -224.1059 0.614 45
12 -224.1784 0.636 45
13 -224.2668 0.614 45
14 -224.2314 0.545 45
15 -224.3243 0.568 45
16 -228.7678 0.568 46
```

We can see the number of atoms is increasing, indicating that the exchange move is being accepted and oxygen atoms are being placed in the system. Still, we might want to increase the acceptance rate further by using `quansino`'s flexibility and creating a custom criteria for the exchange move that optimizes the system before running the criteria:

```python
from quansino.mc.criteria import GrandCanonicalCriteria


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
```

This criteria will run a minimization of the system before evaluating the acceptance criteria, which greatly helps in finding a suitable position for the oxygen atoms. This can be seen in the logger output:

```
Step Epot[eV] AcptRate Natoms
0 -218.0509 0.000 44
1 -227.8073 0.682 45
2 -233.6502 0.568 46
3 -245.7113 0.682 48
4 -258.2237 0.659 50
5 -276.5334 0.614 53
6 -292.4062 0.614 56
7 -310.4333 0.568 59
8 -306.1936 0.545 58
9 -324.0663 0.455 61
10 -330.0961 0.591 62
11 -341.8466 0.705 64
12 -341.7592 0.591 64
13 -356.9901 0.591 66
14 -367.7151 0.614 68
15 -379.8796 0.591 70
16 -397.0771 0.523 73
```

This approach resembles a basin-hopping algorithm and unfortunately does not respect detailed balance. However, it can be useful in some cases where the acceptance rate is low and the system is not converging. A middle ground can be found by fixing every other atom and only optimizing the position of the oxygen atoms being placed, which is left as an exercise for the reader (or can be found in the `examples` directory of the `quansino` repository).
