"""General purpose logging module for atomistic simulations."""

from __future__ import annotations

import time
from typing import IO, TYPE_CHECKING

import numpy as np
from ase import units

from quansino.io.core import TextObserver
from quansino.utils.strings import get_auto_header_format

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from ase import Atoms
    from ase.md.md import MolecularDynamics
    from ase.optimize.optimize import Optimizer

    from quansino.mc.driver import Driver
    from quansino.type_hints import Stress


class Logger(TextObserver):
    """
    A general purpose logger for atomistic simulations, if created manually, the [`add_field`][quansino.io.logger.Logger.add_field] method must be called to configure fields to log. The logger will write the current values of all configured fields to the log file at specified intervals. Callable required for [`add_field`][quansino.io.logger.Logger.add_field] can be easily created by calling functions to obtain the desired value. The logger can also be configured using convenience methods, such as [`add_mc_fields`][quansino.io.logger.Logger.add_mc_fields] and [`add_opt_fields`][quansino.io.logger.Logger.add_opt_fields]. This will be done automatically by Monte Carlo classes if the `logfile` parameter is set.

    Example
    -------
    ``` python
    logger.add_field("Epot[eV]", atoms.get_potential_energy)
    logger.add_field(
        "Class",
        lambda: simulation.__class__.__name__,
        "{:>12s}",
    )
    logger.add_mc_fields(my_mc_simulation)
    logger.add_opt_fields(my_optimization_simulation)
    logger.add_stress_fields(atoms, mask=[True, True, True, False, False, False])
    ```

    Parameters
    ----------
    logfile : IO | str | Path
        File path or open file object for logging.
    interval : int
        Interval at which to log the fields, e.g., every `interval` steps.
    mode: str, optional
        File opening mode if logfile is a filename or path, by default "a".
    **observer_kwargs : Any
        Additional keyword arguments to pass to the parent `TextObserver` class.

    Attributes
    ----------
    fields
        Dictionary of fields to log, fields can be added with the
        [`add_field`][quansino.io.logger.Logger.add_field] method, or using
        convenience methods such as
        [`add_mc_fields`][quansino.io.logger.Logger.add_mc_fields] and
        [`add_opt_fields`][quansino.io.logger.Logger.add_opt_fields].
    """

    __slots__ = ("fields",)

    def __init__(
        self,
        logfile: IO | str | Path,
        interval: int,
        mode: str = "a",
        **observer_kwargs: Any,
    ) -> None:
        """Initialize the simulation `Logger` object."""
        super().__init__(file=logfile, interval=interval, mode=mode, **observer_kwargs)

        self.fields: dict[str | tuple[str, ...], dict[str, Any]] = {}

    def __call__(self) -> None:
        """
        Log the current state of the simulation. Writes a new line to the log file containing the current values of all configured fields.
        """
        parts = []

        for key in self.fields:
            value = self.fields[key]["function"]()

            if self.fields[key]["is_array"]:
                parts.append(self.fields[key]["str_format"].format(*value))
            else:
                parts.append(self.fields[key]["str_format"].format(value))

        self._file.write(" ".join(parts) + "\n")
        self._file.flush()

    def create_header(self) -> str:
        """
        Create the header format string based on configured fields.

        Returns
        -------
        str
            Formatted header string.
        """
        to_write = []

        for name in self.fields:
            header_format = self.fields[name]["header_format"]
            if self.fields[name]["is_array"] and isinstance(name, tuple):
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
        is_array: bool = False,
    ) -> None:
        """
        Add one field to the logger, which track a value that changes during the simulation.

        Parameters
        ----------
        name : str | list[str] | tuple[str, ...]
            Name of the field to add, can be a single string or a list/tuple of strings for array fields.
        function : Callable[[], Any]
            Callable object returning the value of the field.
        str_format : str, optional
            Format string for field value, by default "{:10.3f}".
        header_format : str | None, optional
            Format string for field name in the header line, if `None`, it will be automatically generated based on the `str_format` parameter.
        is_array : bool, optional
            Whether the field's function returns a list of values, by default False. If True, `name` can either be a single string or a list/tuple of strings for each component of the array, and `str_format` should contain the same number of placeholders as the length of the list.

        Example
        -------
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
        The callable can return a list of values to log 1D-arrays or vectors. In this case, `str_format` should be a format string with the same number of placeholders as the length of the list. The `is_array` parameter should be set to `True`, see [`add_stress_fields`][quansino.io.logger.Logger.add_stress_fields] for an example.
        """
        if isinstance(name, list):
            name = tuple(name)

        if header_format is None:
            header_format = get_auto_header_format(str_format)

        self.fields[name] = {
            "function": function,
            "str_format": str_format,
            "header_format": header_format,
            "is_array": is_array,
        }

    def add_mc_fields(self, simulation: Driver) -> None:
        """
        Convenience function to add commonly used fields for [`MonteCarlo`][quansino.mc.core.MonteCarlo] simulations, add the following fields to the logger:

        - Class: The name of the simulation class.
        - Step: The current simulation step.
        - Epot[eV]: The current potential energy.

        Parameters
        ----------
        simulation : MonteCarlo
            The [`MonteCarlo`][quansino.mc.core.MonteCarlo] simulation object to track.
        """
        names = ["Class", "Step", "Epot[eV]"]
        functions = [
            lambda: simulation.__class__.__name__,
            lambda: simulation.step_count,
            simulation.atoms.get_potential_energy,
        ]
        str_formats = ["{:<24s}", "{:>12d}", "{:>12.4f}"]

        for name, function, str_format in zip(
            names, functions, str_formats, strict=False
        ):
            self.add_field(name, function, str_format)

    def add_md_fields(self, simulation: MolecularDynamics) -> None:
        """
        Convenience function to add commonly used fields for `MolecularDynamics` simulations, add the following fields to the logger:

        - Time[ps]: The current simulation time in picoseconds.
        - Epot[eV]: The current potential energy.
        - Ekin[eV]: The current kinetic energy.
        - T[K]: The current temperature.

        Parameters
        ----------
        simulation : MolecularDynamics
            The ASE `MolecularDynamics` object to track.
        """
        names = ["Time[ps]", "Epot[eV]", "Ekin[eV]", "T[K]"]
        functions = [
            lambda: simulation.get_time() / (1000 * units.fs),
            simulation.atoms.get_potential_energy,
            simulation.atoms.get_kinetic_energy,
            simulation.atoms.get_temperature,
        ]
        str_formats = ["{:<12.4f}"] + ["{:>12.4f}"] * 2 + ["{:>10.2f}"]

        for name, function, str_format in zip(
            names, functions, str_formats, strict=False
        ):
            self.add_field(name, function, str_format)

    def add_opt_fields(self, simulation: Optimizer) -> None:
        """
        Convenience function to add commonly used fields for `Optimizer` simulations, add the following fields to the logger:

        - Optimizer: The name of the optimizer class.
        - Step: The current optimization step.
        - Time: The current time in HH:MM:SS format.
        - Epot[eV]: The current potential energy.
        - Fmax[eV/A]: The maximum force component.

        Parameters
        ----------
        simulation : Optimizer
            The ASE `Optimizer` object to track.
        """
        names = ["Class", "Step", "Time", "Epot[eV]", "Fmax[eV/A]"]
        functions = [
            lambda: simulation.__class__.__name__,
            lambda: simulation.nsteps,
            lambda: "{:02d}:{:02d}:{:02d}".format(*time.localtime()[3:6]),
            simulation.optimizable.get_potential_energy,
            lambda: np.linalg.norm(simulation.optimizable.get_forces(), axis=1).max(),
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
        Add stress fields to the logger for all components of the stress tensor. These can be masked using the `mask` parameter.

        Parameters
        ----------
        atoms : Atoms
            The ASE atoms object to track.
        include_ideal_gas : bool, optional
            Whether to include the ideal gas contribution to the stress, by default True.
        mask : list[bool] | None, optional
            A list of booleans to mask the stress components to log, by default None. If None, all components will be logged. The order of components is: xx, yy, zz, yz, xz, xy.
        """
        if mask is None:
            mask = [True] * 6

        def log_stress() -> Stress:
            """
            Get the stress tensor from the atoms object and convert it to GPa.

            Returns
            -------
            Stress
                The stress tensor in GPa, masked according to the `mask` parameter.
            """
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

        self.add_field(names, log_stress, formats, is_array=True)

    def remove_fields(self, pattern: str) -> None:
        """
        Remove fields whose names contain the given pattern.

        Parameters
        ----------
        pattern : str
            Pattern to match in field names. For compound fields (tuple of names), matches if any component contains the pattern.
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
        self._file.write(f"{self.create_header()}\n")
