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
    easily created with lambda functions that return the desired value. For example,
    to log the current energy of an ASE atoms object, use:

    ``` python
    logger.add_field("Epot[eV]", lambda: atoms.get_potential_energy())
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
        File opening mode if logfile is a filename.
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
    fields_string_format
        Dictionary of string formats for fields.
    """

    def __init__(
        self, logfile: IO | str | Path, mode: str = "a", comm: Any = world
    ) -> None:
        """Initialize the logger."""
        self.fields = {}
        self.fields_string_format = {}
        self.logfile = self.openfile(logfile, mode=mode, comm=comm)

        self._field_cache = {}
        self._is_field_list = {}

    def add_field(
        self,
        name: str | list[str],
        function: Callable,
        str_format: str | list[str] = "12.4f",
    ) -> None:
        """
        Add a field to the logger, tracking the value returned by a callable object.

        Parameters
        ----------
        name
            Name of the field to add.
        function
            Callable object returning the value of the field.
        str_format
            Format string for field value.

        Examples
        --------
        ``` python
        logger.add_field("Epot[eV]", lambda: atoms.get_potential_energy())
        logger.add_field(
            ["Class", "Step"],
            [lambda: simulation.__class__.__name__, lambda: simulation.nsteps],
            [">12s", ">12d"],
        )
        logger.add_field(
            ["MovingAtoms[N]", "Econs[eV]"],
            [
                lambda: len(simulation.indices),
                lambda: simulation.transfered_energy - simulation.atoms.get_potential_energy(),
            ],
            [">12d", ">12.4f"],
        )
        ```
        """
        if isinstance(name, list | tuple | np.ndarray):
            assert isinstance(str_format, list | tuple | np.ndarray)
            assert len(name) == len(str_format)
            self._is_field_list[function] = True
        else:
            self._is_field_list[function] = False

        self.fields[function] = name
        self.fields_string_format[function] = str_format
        self._build_format_cache(function)

    def add_mc_fields(self, simulation: MonteCarlo) -> None:
        """
        Convenience function to add commonly used fields for Monte Carlo simulation,
        add the following fields to the logger:

        - Class: The name of the simulation class.
        - Step: The current simulation step.
        - Epot[eV]: The current potential energy.

        Parameters
        ----------
        simulation
            The Monte Carlo simulation object.
        """
        names = ["Step", "Epot[eV]"]
        functions = [lambda: simulation.nsteps, simulation.atoms.get_potential_energy]
        str_formats = ["<12d", ">12.4f"]

        for name, function, str_format in zip(
            names, functions, str_formats, strict=False
        ):
            self.add_field(name, function, str_format)

    def add_md_fields(self, dyn: MolecularDynamics) -> None:
        """
        Convenience function to add commonly used fields for MD simulations, add the
        following fields to the logger:

        - Time[ps]: The current simulation time in picoseconds.
        - Etot[eV]: The current total energy.
        - Epot[eV]: The current potential energy.
        - Ekin[eV]: The current kinetic energy.
        - T[K]: The current temperature.

        Parameters
        ----------
        dyn
            The :class:~ase.md.md.MolecularDynamics` object.
        """
        names = ["Time[ps]", "Etot[eV]", "Epot[eV]", "Ekin[eV]", "T[K]"]
        functions = [
            lambda: dyn.get_time() / (1000 * units.fs),
            dyn.atoms.get_total_energy,
            dyn.atoms.get_potential_energy,
            dyn.atoms.get_kinetic_energy,
            dyn.atoms.get_temperature,
        ]
        str_formats = ["<12.4f"] + [">12.4f"] * 3 + [">10.2f"]

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
            The ASE optimizer object.
        """
        names = ["Optimizer", "Step", "Time", "Epot[eV]", "Fmax[eV/A]"]
        functions = [
            lambda: optimizer.__class__.__name__,
            lambda: optimizer.nsteps,
            lambda: "{:02d}:{:02d}:{:02d}".format(*time.localtime()[3:6]),
            optimizer.optimizable.get_potential_energy,
            lambda: np.linalg.norm(optimizer.optimizable.get_forces(), axis=1).max(),
        ]
        str_formats = ["<24s"] + [">4d"] + [">12s"] + [">12.4f"] * 2

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
        Add the stress fields to the logger.

        Parameters
        ----------
        atoms
            The ASE atoms object.
        include_ideal_gas
            Whether to include the ideal gas contribution to the stress.
        """
        if mask is None:
            mask = [True] * 6

        def log_stress():
            stress = atoms.get_stress(include_ideal_gas=include_ideal_gas)
            stress = tuple(stress / units.GPa)
            return np.array([s for n, s in enumerate(stress) if mask[n]])

        components = ["xx", "yy", "zz", "yz", "xz", "xy"]

        names = [
            f"{component}Stress[GPa]"
            for n, component in enumerate(components)
            if mask[n]
        ]

        formats = [">14.3f"] * sum(mask)

        self.add_field(names, log_stress, formats)

    def remove_fields(self, name: str) -> None:
        """
        Remove one or multiple field(s) from the logger. Work by finding partial
        matches of the field name(s) in the current fields. List fields count as a
        single field, i.e., if a match is found in a list field, the whole list field
        is removed

        Parameters
        ----------
        name
            Name of the field to remove.
        """
        for func, field_name in list(self.fields.items()):
            if name in field_name:
                self.fields.pop(func, None)
                self.fields_string_format.pop(func, None)
                self._field_cache.pop(func, None)

    def write_header(self) -> None:
        """Write the header line to the log file."""
        self.logfile.write(f"{self._create_header_format()}\n")

    def _build_format_cache(self, key: Callable) -> None:
        """Build the format cache for a field.

        Parameters
        ----------
        key: Callable
            The field to build the cache for.
        """
        if self._is_field_list[key]:
            self._field_cache[key] = " ".join(
                f"{{:{f}}}" for f in self.fields_string_format[key]
            )
        else:
            self._field_cache[key] = f"{{:{self.fields_string_format[key]}}}"

    def _create_header_format(self) -> str:
        """
        Create the header string based on configured fields.

        Returns
        -------
        str
            Formatted header string.
        """
        to_write = []

        for key in self.fields:
            name = self.fields[key]
            str_format = self.fields_string_format[key]

            if self._is_field_list[key]:
                to_write.extend(
                    [
                        f"{n:{get_auto_header_format(fmt)}}"
                        for n, fmt in zip(name, str_format, strict=False)
                    ]
                )
            else:
                to_write.append(f"{name:{get_auto_header_format(str_format)}}")

        return " ".join(to_write)

    def __call__(self) -> None:
        """
        Writes a new line to the log file containing the current values of all
        configured fields.
        """
        to_write = [
            (
                self._field_cache[key].format(*key())
                if self._is_field_list[key]
                else self._field_cache[key].format(key())
            )
            for key in self.fields
        ]

        self.logfile.write(" ".join(to_write) + "\n")
        self.logfile.flush()

    def __del__(self) -> None:
        """Close the log file on deletion."""
        self.close()
