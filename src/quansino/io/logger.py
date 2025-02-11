"""General purpose logging module for atomistic simulations."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from ase.parallel import world
from ase.utils import IOContext

from quansino.utils.strings import get_auto_header_format

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import IO, Any

    from ase import Atoms
    from ase.md.md import MolecularDynamics
    from ase.optimize.optimize import Optimizer

    from quansino.mc.core import MonteCarlo


class Logger(IOContext):
    """
    A general purpose logger for atomistic simulations, if created manually, the
    [`add_field`][quansino.io.logger.Logger.add_field] method must be called to
    configure the fields to log.

    Callable required for [`add_field`][quansino.io.logger.Logger.add_field] can be
    easily created by calling functions to obtain the desired value. For example,
    to log the current energy of an ASE atoms object, use:

    ``` python
    logger.add_field("Epot[eV]", atoms.get_potential_energy)
    ```

    The logger can also be configured using convenience methods, such as
    [`add_mc_fields`][quansino.io.logger.Logger.add_mc_fields] and
    [`add_opt_fields`][quansino.io.logger.Logger.add_opt_fields]. This will be done
    automatically by Monte Carlo classes if the `logfile` parameter is set.

    Parameters
    ----------
    logfile: IO | str | Path
        File path or open file object for logging, use "-" for standard output.
    mode: str
        File opening mode if logfile is a filename or path.
    comm: Any
        MPI communicator for parallel simulations.

    Attributes
    ----------
    logfile
        The opened file object.
    fields
        Dictionary of fields to log, fields can be added with the
        [`add_field`][quansino.io.logger.Logger.add_field] method, or using
        convenience methods such as
        [`add_mc_fields`][quansino.io.logger.Logger.add_mc_fields] and
        [`add_opt_fields`][quansino.io.logger.Logger.add_opt_fields].
    """

    def __init__(
        self, logfile: IO | str | Path, mode: str = "a", comm: Any = world
    ) -> None:
        """Initialize the molecular dynamics logger."""
        self.fields: dict[str | tuple[str, ...], Any] = {}
        self.logfile = self.openfile(logfile, mode=mode, comm=comm)

    def __call__(self) -> None:
        """
        Log the current state of the simulation. Writes a new line to the log file containing the current values of all configured fields.
        """
        parts = []

        for key in self.fields:
            value = self.fields[key]["function"]()

            if self.fields[key]["is_list"]:
                parts.append(self.fields[key]["str_format"].format(*value))
            else:
                parts.append(self.fields[key]["str_format"].format(value))

        self.logfile.write(" ".join(parts) + "\n")
        self.logfile.flush()

    def __del__(self) -> None:
        """Clean up by closing the log file."""
        self.close()

    def create_header(self) -> str:
        """
        Create the header format string based on configured options.

        Returns
        -------
        str
            Formatted header string.
        """
        to_write = []

        for name in self.fields:
            header_format = self.fields[name]["header_format"]
            if self.fields[name]["is_list"]:
                to_write.append(header_format.format(*name))
            else:
                to_write.append(header_format.format(name))

        return " ".join(to_write)

    def add_field(
        self,
        name: str | list[str] | tuple[str, ...],
        function: Callable[[], Any],
        str_format: str = "{:10.3f}",
        header_format: str | None = None,
        is_list: bool = False,
    ) -> None:
        """
        Add one field to the logger, which track a value that
        changes during the simulation.

        Parameters
        ----------
        name
            Name of the field to add.
        function
            Callable object returning the value of the field.
        str_format
            Format string for field value.
        header_format
            Format string for field name in the header line.
        is_list
            Whether the field's function returns a list of values.

        Examples
        --------
        ``` python
        logger.add_field("Epot[eV]", atoms.get_potential_energy)
        logger.add_field(
            "Class",
            lambda: simulation.__class__.__name__,
            "{:>12s}",
        )
        ```

        Notes
        -----
        The callable can return a list of values to log arrays or vectors. In this case, `name` should be a list or tuple of strings, and `str_format` should be a format string with the same number of placeholders as the length of the list. The `is_list` parameter should be set to `True`.
        """
        if isinstance(name, list):
            name = tuple(name)

        if header_format is None:
            header_format = get_auto_header_format(str_format)

        self.fields[name] = {
            "function": function,
            "str_format": str_format,
            "header_format": header_format,
            "is_list": is_list,
        }

    def add_mc_fields(self, simulation: MonteCarlo) -> None:
        """
        Convenience function to add commonly used fields for [`MonteCarlo`][quansino.mc.core.MonteCarlo] simulation, add the following fields to the logger:

        - Class: The name of the simulation class.
        - Step: The current simulation step.
        - Epot[eV]: The current potential energy.

        Parameters
        ----------
        simulation
            The `MonteCarlo` simulation object.
        """
        names = ["Step", "Epot[eV]"]
        functions = [lambda: simulation.nsteps, simulation.atoms.get_potential_energy]
        str_formats = ["{:<12d}", "{:>12.4f}"]

        for name, function, str_format in zip(
            names, functions, str_formats, strict=False
        ):
            self.add_field(name, function, str_format)

    def add_md_fields(self, dyn: MolecularDynamics) -> None:
        """
        Convenience function to add commonly used fields for Molecular Dynamics simulations, add the following fields to the logger:

        - Time[ps]: The current simulation time in picoseconds.
        - Epot[eV]: The current potential energy.
        - Ekin[eV]: The current kinetic energy.
        - T[K]: The current temperature.

        Parameters
        ----------
        dyn
            The ASE `MolecularDynamics` object.
        """
        names = ["Time[ps]", "Epot[eV]", "Ekin[eV]", "T[K]"]
        functions = [
            lambda: dyn.get_time() / (1000 * units.fs),
            dyn.atoms.get_potential_energy,
            dyn.atoms.get_kinetic_energy,
            dyn.atoms.get_temperature,
        ]
        str_formats = ["{:<12.4f}"] + ["{:>12.4f}"] * 3 + ["{:>10.2f}"]

        for name, function, str_format in zip(
            names, functions, str_formats, strict=False
        ):
            self.add_field(name, function, str_format)

    def add_opt_fields(self, optimizer: Optimizer) -> None:
        """
        Convenience function to add commonly used fields for ASE optimizers, add the
        following fields to the logger:

        - Optimizer: The name of the optimizer class.
        - Step: The current optimization step.
        - Time: The current time in HH:MM:SS format.
        - Epot[eV]: The current potential energy.
        - Fmax[eV/A]: The maximum force component.

        Parameters
        ----------
        optimizer
            The ASE `Optimizer` object.
        """
        names = ["Optimizer", "Step", "Time", "Epot[eV]", "Fmax[eV/A]"]
        functions = [
            lambda: optimizer.__class__.__name__,
            lambda: optimizer.nsteps,
            lambda: "{:02d}:{:02d}:{:02d}".format(*time.localtime()[3:6]),
            optimizer.optimizable.get_potential_energy,
            lambda: np.linalg.norm(optimizer.optimizable.get_forces(), axis=1).max(),
        ]
        str_formats = ["{:<24s}"] + ["{:>4d}"] + ["{:>12s}"] + ["{:>12.4f}"] * 2

        for name, function, str_format in zip(
            names, functions, str_formats, strict=False
        ):
            self.add_field(name, function, str_format)

    def add_stress_fields(
        self,
        atoms: Atoms,
        include_ideal_gas: bool = True,
        mask: list[bool] | None = None,
    ) -> None:
        """
        Add stress fields to the logger, add the following fields to the logger:

        - Stress[xx][GPa]: The xx component of the stress tensor.
        - Stress[yy][GPa]: The yy component of the stress tensor.
        - Stress[zz][GPa]: The zz component of the stress tensor.
        - Stress[yz][GPa]: The yz component of the stress tensor.
        - Stress[xz][GPa]: The xz component of the stress tensor.
        - Stress[xy][GPa]: The xy component of the stress tensor.

        These can be masked using the `mask` parameter.

        Parameters
        ----------
        atoms : Atoms
            The ASE atoms object.
        include_ideal_gas : bool, optional
            Whether to include the ideal gas contribution to the stress.
        mask : list[bool], optional
            A list of booleans to mask the stress components to log.
            The default is to log all components.
        """
        if mask is None:
            mask = [True] * 6

        def log_stress():
            stress = atoms.get_stress(include_ideal_gas=include_ideal_gas)
            stress = tuple(stress / units.GPa)
            return np.array([s for n, s in enumerate(stress) if mask[n]])

        components = ["xx", "yy", "zz", "yz", "xz", "xy"]

        names = [
            f"Stress[{component}][GPa]"
            for n, component in enumerate(components)
            if mask[n]
        ]

        formats = "{:>18.3f}" * sum(mask)

        self.add_field(names, log_stress, formats, is_list=True)

    def remove_fields(self, pattern: str) -> None:
        """
        Remove fields whose names contain the given pattern.

        Parameters
        ----------
        pattern : str
            Pattern to match in field names. For compound fields
            (tuple of names), matches if any component contains the pattern.
        """
        for field_name in list(self.fields.keys()):
            if isinstance(field_name, tuple):
                if any(pattern in name for name in field_name):
                    self.fields.pop(field_name)
            else:
                if pattern in field_name:
                    self.fields.pop(field_name)

    def write_header(self) -> None:
        """Write the header line to the log file."""
        self.logfile.write(f"{self.create_header()}\n")
