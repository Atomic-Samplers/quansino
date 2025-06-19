The [`quansino`](https://github.com/Atomic-Samplers/quansino) package allows you to easily create flexible and modular Monte Carlo simulations in Python. Below is an overview of the main concepts and components of the package.


### The [`MonteCarlo`][quansino.mc.core.MonteCarlo] Class

The [`MonteCarlo`][quansino.mc.core.MonteCarlo] class is the core of the [`quansino`](https://github.com/Atomic-Samplers/quansino) package. The role of this class is to manage the simulation process, including:

- Initializing the simulation with a set of parameters.
- Running the simulation for a specified number of iterations or convergence criteria.
- Managing the observers and their potential files.
- Yielding the move that are going to be run.

In [`quansino`](https://github.com/Atomic-Samplers/quansino), a simulation can be summarized via the following pseudo-code:

```
iterate over steps:
    iterate over moves:
        perform move
        evaluate criteria
        save or revert state

    call observers

    if "converged":
        stop
    else:
        continue to next step
```

This skeleton is what is run when you call the [`run`][quansino.mc.core.MonteCarlo] method of the [`MonteCarlo`][quansino.mc.core.MonteCarlo] class. The method call the [`step`][quansino.mc.core.MonteCarlo.step] method, `max_steps` times, which in turn calls the [`yield_moves`][quansino.mc.core.MonteCarlo.yield_moves] method to select the moves to be performed. At the end of each step, the observers are called to perform their actions, such as saving the state or logging information.

Convenience methods such as [`irun`][quansino.mc.driver.Driver.irun] or [`srun`][quansino.mc.core.MonteCarlo.srun] are provided to run the simulation with more or less flexibility:

```python
for steps in mc.srun(max_steps=1000):
    pass
    # Any custom code here will be run after each step, before observers are called

for steps in mc.irun(max_steps=1000):
    for move_name in steps:
        pass
        # Any custom code here will be run before each move is performed

        # Any custom code here will be run after each step, before observers are called
```

Users seeking even more flexibility are free to subclass the [`MonteCarlo`][quansino.mc.core.MonteCarlo] class and override the [`step`][quansino.mc.core.MonteCarlo.step] and [`yield_moves`][quansino.mc.core.MonteCarlo.yield_moves] methods to implement custom logic.

To perform the simulation, the [`MonteCarlo`][quansino.mc.core.MonteCarlo] class calls multiple objects each aiming to perform a specific task. These objects will be described in the following sections.

### [`Move`][quansino.protocols.Move] Classes

[`Move`][quansino.protocols.Move] classes are responsible for defining actions that can be performed during the simulation. Each move when called perform geometric, exchange, or other types of operations on the system. The [`Move`][quansino.protocols.Move] class is defined as a [`Protocol`][typing.Protocol], allowing users to create their custom classes, and pass them in the Monte Carlo simulation, which take the [`Move`][quansino.protocols.Move] protocol as a Generic Type.

For a more focused approach, the package provide the [`BaseMove`][quansino.moves.core.BaseMove] class and their subclasses, such as [`DisplacementMove`][quansino.moves.displacement.DisplacementMove], [`ExchangeMove`][quansino.moves.exchange.ExchangeMove], and [`CellMove`][quansino.moves.cell.CellMove].

Each of these move classes implements the [`__call__`][quansino.protocols.Move.__call__] method, which is invoked during the simulation to perform the move. In the case of [`DisplacementMove`][quansino.moves.displacement.DisplacementMove], it will attempt to displace a particle selected randomly from the system. These behaviors are heavily tunable and can be customized by passing parameters to the move classes.

```python
from quansino.moves import DisplacementMove

move = DisplacementMove(
    labels=[0, 1, 1, 2, 2, 2, -1, -1],
    operation=Ball(0.1),
    apply_constraints=True,  # ASE constraints will be applied when performing the move.
)  # This move will attempt to displace one particle each time it is called.
```

!!! warning
    The word "particle" is used as a general term to refer to any entity in the simulation that is considered a single unit, such as a single atom, molecules, or other entities. In the above example, the labels represent which atoms are grouped together as a "particle". Atoms with the same label are considered part of the same particle, and the move will move the entire particle as a single unit, allowing molecular displacements.

[`Operations`][quansino.protocols.Operation] will be detailed in the next section, but it is important to note that it defines the action that will be performed on the particle. In this case, `Ball(0.1)` will displace the particle by a random distance of maximum 0.1 Å in a ball around the particle's current position.

If users want to move multiple particles at once (per criteria), or perform more complex actions, they can simply add moves together:

```python
from quansino.moves import DisplacementMove, ExchangeMove

# Create a move that will attempt to displace one particle and add/remove another particle
weird_move = DisplacementMove(...) + ExchangeMove(...)

# Multiplication is also supported, allowing to repeat the same move multiple times
multi_displacement_move = (
    DisplacementMove(...) * 10
)  # Will attempt to displace 10 particles per criteria
```

### [`Operation`][quansino.operations.core.Operation] Classes

[`Operations`][quansino.protocols.Operation] based classes define the specific actions that can be performed on the system during the simulation. The [`Operation`][quansino.protocols.Operation] class is also defined as a [`Protocol`][typing.Protocol], allowing users to easily create custom operations. Sensible defaults are provided in the package, such as [`Ball`][quansino.operations.displacement.Ball], [`Box`][quansino.operations.displacement.Box], [`Sphere`][quansino.operations.displacement.Sphere]. Most of the time these classes only define a single [`calculate`][quansino.protocols] method, which takes a [`Context`][quansino.mc.contexts.Context] as an argument and returns the displacement/strain vector to be applied to the particle/box.

```python
from quansino.operations import Ball, Translation

operation = Ball(
    0.1
)  # Displaces particles by a random distance of maximum 0.1 Angstroms in a ball around the particle's current position

displacement_vector = operation.calculate(
    ...
)  # Returns a random displacement vector used in the move.

operation = Translation()  # Translates particles randomly in the unit cell.
```

Most of the time, operations are attempted in the [`__call__`][quansino.protocols.Move.__call__] method of the [`Move`][quansino.protocols.Move] class, and is checked against a user-defined criteria `check_move` [`Callable`][typing.Callable] attribute. This can be used to define constraints, such as ensuring that particles do not overlap or go outside of a defined box. By default, operations are attempted a limited amount times, and the first successful one is used. This can be customized by modifying the `max_attempts` attribute of the [`BaseMove`][quansino.moves.core.BaseMove] class.

### [`Criteria`][quansino.protocols.Criteria] Classes

[`Criteria`][quansino.protocols.Criteria] based classes are used to evaluate whether a move is successful or not. They define the conditions that must be met for a move to be accepted. The [`Criteria`][quansino.protocols.Criteria] class is also defined as a [`Protocol`][typing.Protocol], allowing users to create custom criteria. The package provides several built-in criteria, such as [`CanonicalCriteria`][quansino.mc.criteria.CanonicalCriteria], [`GrandCanonicalCriteria`][quansino.mc.criteria.GrandCanonicalCriteria], and [`IsobaricCriteria`][quansino.mc.criteria.IsobaricCriteria]. When adding a move to a simulation, the criteria can be passed as well. Convenience classes, such as [`BaseCriteria`][quansino.mc.criteria.BaseCriteria], [`CanonicalCriteria`][quansino.mc.criteria.CanonicalCriteria], and [`GrandCanonicalCriteria`][quansino.mc.criteria.GrandCanonicalCriteria] are provided. These classes implement the [`evaluate`][quansino.protocols.Criteria.evaluate] method, which takes the context as argument.

```python
from quansino.mc.criteria import CanonicalCriteria

criteria = CanonicalCriteria()

criteria.evaluate(context)  # Returns True if the move is accepted, False otherwise
```

### [`Context`][quansino.mc.contexts.Context] Class

The [`Context`][quansino.mc.contexts.Context] class provides the necessary information about the current state of the simulation. It encapsulates all relevant data that may be needed by moves, operations, and criteria to make decisions. This includes information about the system's configuration, (Atoms object), temperature, pressure, and other simulation parameters. The [`Context`][quansino.mc.contexts.Context] class lives both in the [`MonteCarlo`][quansino.mc.core.MonteCarlo] class and in the [`Move`][quansino.protocols.Move] class, allowing to access the context from both places. The context is passed to [`Operations`][quansino.protocols.Operation] and [`Criteria`][quansino.protocols.Criteria] classes when they are called, allowing them to access the necessary information to perform their actions.

Each kind of simulation (e.g., canonical, grand canonical) have its own context (see [`DisplacementContext`][quansino.mc.contexts.DisplacementContext], [`DeformationContext`][quansino.mc.contexts.DeformationContext], etc.) which is initialized with the relevant parameters.

!!! danger
    [`Context`][quansino.mc.contexts.Context] based classes are not meant to be modified by users. It is used internally by the package to manage the state of the simulation. Unless you are implementing a custom move, operation, or criteria, that requires additional information to be passed, you should not need to interact with the [`Context`][quansino.mc.contexts.Context] class directly. Instead, you can rely on the methods provided by the [`MonteCarlo`][quansino.mc.core.MonteCarlo] class to access the necessary information.

Find below a diagram summarizing the relationships between the main classes in the [`quansino`](https://github.com/Atomic-Samplers/quansino) package:

``` mermaid
classDiagram
  Criteria <-- MonteCarlo
  Move <-- MonteCarlo
  Context <-- MonteCarlo
  class MonteCarlo{
    context
    moves[MoveType, CriteriaType]
    add_move(move, criteria, name, interval, probability)
    run(max_steps)
    step()
    yield_moves()
  }
  class Context{
    atoms
    rng
  }
  class Move{
    __call__(context)
  }
  class Criteria{
    evaluate(context)
  }
  class Operation{
    calculate(context)
  }
  Operation <-- Move
```

### Summary

Gathering the information from the previous sections, a typical simulation setup in [`quansino`](https://github.com/Atomic-Samplers/quansino) would look like this:

```python
import numpy as np

from quansino.mc import Canonical
from quansino.moves import DisplacementMove, ExchangeMove
from quansino.operations import Ball
from quansino.mc.criteria import CanonicalCriteria

from ase.build import bulk

atoms = bulk("Cu", "fcc", cubic=True)

atoms.calc = ...

mc = Canonical(atoms, temperature=300, max_cycles=len(atoms), seed=42)

move = DisplacementMove(
    labels=np.arange(len(atoms)),
    operation=Ball(0.1),
)

mc.add_move(
    move,
    criteria=CanonicalCriteria(),
    name="displacement_move",
    interval=1,
    probability=1.0,
)

for steps in mc.irun(1000):
    for move_name in steps:  # <-- len(atoms) cycles.
        print(f"Performed move: {move_name}")

# Alternatively:
mc.run(1000)
```
