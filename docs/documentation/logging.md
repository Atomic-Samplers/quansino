Quansino differs from ASE for logging in that it does not make use of its own logging module:

- The [`Logger`][quansino.io.logger.Logger] class is used to create custom loggers that are flexible and can be configured to log any kind of information.
- The [`TrajectoryObserver`][quansino.io.trajectory.TrajectoryObserver] class only supports `extxyz` (`xyz`) format to log atomic trajectories.
- There is an additional [`RestartObserver`][quansino.io.restart.RestartObserver] class that can be used to log the entire simulation state at regular intervals for future restarts.

All these classes are located in the `quansino.io` module and will be described in detail in the following sections.

### The [`Logger`][quansino.io.logger.Logger] class

The [`Logger`][quansino.io.logger.Logger] class is designed to create custom loggers that can be configured to log various types of information. It provides a flexible interface for logging messages, warnings, and errors. The logger makes use of a dictionary called `fields` to store functions to be called when writing to the log. This allows for easy customization of the logging behavior. For example, to log the potential energy of a system, one can define the following field using the [`add_field`][quansino.io.logger.Logger.add_field] method.

```python
from quansino.io import Logger

logger = Logger()
logger.add_field("potential_energy", atoms.get_potential_energy)
```

[`add_field`][quansino.io.logger.Logger.add_field]  takes a name and a function as arguments; the function should not take any arguments and should return the value to be logged.

It is also possible to specify the format of the logged value using the `format` argument, which must be a format string compatible with Python's `str.format()` method. For example, to log the potential energy with a precision of 3 decimal places, one can do:

```python
logger.add_field("potential_energy", atoms.get_potential_energy, format="{:.3f}")
```

The name provided to [`add_field`][quansino.io.logger.Logger.add_field] will be used in the header of the logfile. The format to be used will be automatically determined based on the `format` argument, but can also be specified manually.

Functions that return 1D arrays of values can also be logged, in which case the `format` argument must be explicitly specified to indicate how the values should be formatted. For example, to log the symbols of the atoms in the system, one can do:

```python
atoms = ...

logger.add_field(
    "Symbols",
    atoms.get_chemical_symbols,
    str_format="{:>4s}" * len(atoms),
    header_format=f"{{:>{4 * len(atoms)}s}}",
    is_array=True,
)

logger.add_field(
    tuple(f"Symbol[{i}]" for i in range(len(atoms))),
    atoms.get_chemical_symbols,
    str_format="{:>12s}" * len(atoms),
    is_array=True,
)
# header_format can be set to None and will be automatically generated from str_format
```

A [`Logger`][quansino.io.logger.Logger] sent as an argument to [`MonteCarlo`][quansino.mc.core.MonteCarlo] classes will be automatically configured to log various properties of interest, such as potential energy, forces, and stresses, depending on the simulation type. However, custom loggers manually created by the user will have to be configured manually. To this end, the [`Logger`][quansino.io.logger.Logger] class provides convenience methods such as [`add_md_fields`][quansino.io.logger.Logger.add_md_fields], [`add_mc_fields`][quansino.io.logger.Logger.add_mc_fields], and [`add_opt_fields`][quansino.io.logger.Logger.add_opt_fields] to add fields that are commonly used in molecular dynamics, Monte Carlo, and optimization simulations, respectively.

### The [`TrajectoryObserver`][quansino.io.trajectory.TrajectoryObserver] class

The [`TrajectoryObserver`][quansino.io.trajectory.TrajectoryObserver] class is used to log atomic trajectories in the `extxyz` (`xyz`) format. It provides a simple interface for writing atomic positions, velocities, and forces to a file. The trajectory can be written to a file at regular intervals during the simulation, allowing for easy analysis of the atomic motion.

As of now, the ASE `Trajectory` writer and reader are not compatible with `quansino`. Users are encouraged to use the [`TrajectoryObserver`][quansino.io.trajectory.TrajectoryObserver] class provided by `quansino` for logging atomic trajectories.

### The [`RestartObserver`][quansino.io.restart.RestartObserver] class

The [`RestartObserver`][quansino.io.restart.RestartObserver] class is designed to log entire simulation states at regular intervals, allowing for future restarts of the simulation. It is particularly useful for long-running simulations where one might want to save the state of the system periodically. In `quansino`, objects possess methods such as `to_dict` and `from_dict` that allow for easy serialization and deserialization of the simulation state. The [`RestartObserver`][quansino.io.restart.RestartObserver] class uses these methods to save the state of the simulation in JSON format, which can be easily read and written to files.

```python
from quansino.mc.canonical import Canonical

from ase.io.jsonio import read_json

with open("restart.json", "r") as f:
    data = read_json(f)

mc = Canonical.from_dict(data)
...
```

Observers are fairly flexible and files can be changed at any time during the simulation. For example, to have restart files written every 1000 steps, one can do:

```python
mc = ...

mc.add_observer(
    "my_restart",
    RestartObserver(file="restart.json", interval=1000, mode="w"),
)

for step in mc.srun(10000):
    if step % 1000 == 0:
        mc.observers["my_restart"].file = f"restart_{mc.step_count}.json"
```

With this setup, the simulation state will be logged at the beginning to `restart.json`, and then every 1000 steps, the file will be updated to a new file named `restart_1000.json`, `restart_2000.json`, etc. This allows for easy restarts of the simulation from the last saved state.

### File management

In `quansino`, file management is handled by a custom [`ObserverManager`][quansino.io.file.ObserverManager] class that is responsible for managing the files used in the simulation. The main purpose of this class is to ensure that files are properly closed after use and to provide a consistent interface for file operations. Each [`MonteCarlo`][quansino.mc.core.MonteCarlo] class has its own [`ObserverManager`][quansino.io.file.ObserverManager] instance, which is used to manage the files associated with that simulation.

In practice, when the [`attach_observer`][quansino.mc.driver.Driver.attach_observer] method is called on a [`MonteCarlo`][quansino.mc.core.MonteCarlo] class, the [`ObserverManager`][quansino.io.file.ObserverManager] instance is used to open the specified file and create an observer that will write the simulation data to that file. The [`ObserverManager`][quansino.io.file.ObserverManager] ensures that the file is properly closed when the simulation is finished, preventing any potential data loss or corruption.

Users are free to create their own [`ObserverManager`][quansino.io.file.ObserverManager] instances if they need to manage files in a custom way, but in most cases, the default behavior provided by the [`MonteCarlo`][quansino.mc.core.MonteCarlo] classes will suffice. The [`ObserverManager`][quansino.io.file.ObserverManager] class is designed to be flexible and can be extended or modified as needed to suit specific requirements.

Users should call the convenience method [`close`][quansino.mc.driver.Driver.close] to close the files managed by the [`ObserverManager`][quansino.io.file.ObserverManager] instance. This will ensure that all data is properly written to the files and that the files are closed correctly. Note that the [`MonteCarlo`][quansino.mc.core.MonteCarlo] classes will not automatically close the files when the simulation is finished.
