"""Module to perform force bias Monte Carlo simulations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from ase.units import kB

from quansino.mc.core import MonteCarlo
from quansino.utils.atoms import has_constraint

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms
    from numpy.typing import NDArray

    from quansino.type_hints import Displacement, Forces, Masses


class ForceBias(MonteCarlo):
    """
    Force Bias Monte Carlo class to perform simulations as described in
    https://doi.org/10.1063/1.4902136.

    Parameters
    ----------
    atoms: Atoms
        The atomic system being simulated.
    delta: float
        Delta parameter in Angstrom which influence how much the atoms are moved.
    temperature: float
        The temperature of the simulation in Kelvin. Default: 298.15.
    **mc_kwargs
        Additional keyword arguments to pass to the MonteCarlo superclass. See [`MonteCarlo`][quansino.mc.core.MonteCarlo] for more information.

    Attributes
    ----------
    gamma_max_value: float
        Maximum value for the gamma parameter, used to avoid overflow errors.
    delta: float
        Delta parameter in Angstrom which influence how much the atoms are moved.
    temperature: float
        The temperature of the simulation in Kelvin.
    """

    gamma_max_value = 709.782712

    def __init__(
        self, atoms: Atoms, delta: float, temperature: float = 298.15, **mc_kwargs
    ) -> None:
        """Initialize the Force Bias Monte Carlo object."""
        self.delta = delta
        self.temperature = temperature

        self.size = (len(atoms), 3)

        self.update_masses(atoms.get_masses())
        self.set_masses_scaling_power(np.full((len(atoms), 3), 0.25))

        super().__init__(atoms, **mc_kwargs)

        if not has_constraint(self.atoms, "FixCom"):
            warn(
                "No `FixCom` constraint found, `ForceBias` simulations lead to sustained drift of the center of mass.",
                stacklevel=2,
            )

        self.gamma = 0.0

        if self.default_logger:
            self.default_logger.add_field(
                "Gamma/GammaMax",
                lambda: np.max(np.abs(self.gamma / self.gamma_max_value)),
                str_format="{:>16.2f}",
            )

        self.current_size = self.size

    def calculate_gamma(self, forces: NDArray[np.floating]) -> None:
        """
        Calculate the gamma parameter for the Monte Carlo step, along with the denominator for the trial probability.

        Parameters
        ----------
        forces
            The forces acting on the atoms.
        """
        self.gamma = np.clip(
            (forces * self.delta) / (2 * self.temperature * kB),
            -self.gamma_max_value,
            self.gamma_max_value,
        )

        self.denominator = np.exp(self.gamma) - np.exp(-self.gamma)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the object."""
        dictionary = MonteCarlo.to_dict(self)
        dictionary.update(
            {
                "temperature": self.temperature,
                "delta": self.delta,
                "masses_scaling_power": self.masses_scaling_power,
            }
        )

        return dictionary

    def set_masses_scaling_power(
        self, value: dict[str, float] | NDArray | float
    ) -> None:
        if isinstance(value, dict):
            self.masses_scaling_power = np.full(self.size, 0.25)

            for el in np.unique(self.atoms.symbols):
                indices = self.atoms.symbols == el
                self.masses_scaling_power[indices, :] = value.get(el, 0.25)

        elif isinstance(value, float | np.floating):
            self.masses_scaling_power = np.full(self.size, value)
        elif isinstance(value, np.ndarray):
            if value.shape != self.size:
                raise ValueError(
                    f"Invalid shape for masses_scaling_power. Expected {self.size}, got {value.shape}."
                )

            self.masses_scaling_power = value
        else:
            raise ValueError("Invalid value type for masses_scaling_power.")

        self.mass_scaling = np.power(
            np.min(self.shaped_masses) / self.shaped_masses, self.masses_scaling_power
        )

    def update_masses(self, masses: Masses | None = None) -> None:
        if masses is None:
            masses = self.atoms.get_masses()

        if masses.ndim == 1:
            masses = np.broadcast_to(masses[:, np.newaxis], self.size)

        self.shaped_masses = masses

    def step(self) -> Forces:  # type: ignore
        """Perform one Force Bias Monte Carlo step."""
        forces = self.atoms.get_forces()
        positions = self.atoms.get_positions()

        self.current_size = (len(self.atoms), 3)

        self.calculate_gamma(forces)

        self.zeta = self.get_zeta()

        probability_random = self._rng.random(self.current_size)
        converged = self.calculate_trial_probability() > probability_random

        while not np.all(converged):
            self.current_size = probability_random[~converged].shape
            self.zeta[~converged] = self.get_zeta()

            probability_random[~converged] = self._rng.random(self.current_size)

            converged = self.calculate_trial_probability() > probability_random

        displacement = self.zeta * self.delta * self.mass_scaling

        self.atoms.set_momenta(self.shaped_masses * displacement)
        corrected_displacement = self.atoms.get_momenta() / self.shaped_masses

        self.atoms.set_positions(positions + corrected_displacement)

        return forces

    def get_zeta(self) -> Displacement:
        """
        Get the zeta parameter for the Monte Carlo step.

        Returns
        -------
        Displacement
            The zeta parameter.
        """
        return self._rng.uniform(-1, 1, self.current_size)

    def calculate_trial_probability(self) -> NDArray:
        """
        Calculate the trial probability for the Monte Carlo step.

        Returns
        -------
        NDArray[np.floating]
            The trial probability.
        """
        sign_zeta = np.sign(self.zeta)

        probability_trial = np.exp(sign_zeta * self.gamma) - np.exp(
            self.gamma * (2 * self.zeta - sign_zeta)
        )
        probability_trial *= sign_zeta

        return np.divide(
            probability_trial,
            self.denominator,
            out=np.ones_like(probability_trial),
            where=self.denominator != 0,
        )


class AdaptiveForceBias(ForceBias):
    """
    Adaptive Force Bias Monte Carlo class to perform simulations with adaptive delta parameter. The delta parameter is adjusted based on the variation coefficient of the forces or energies which is calculated based on the variance of the forces or energies. The current class assume that the forces or energies are calculated by a committee of MACECalculator.

    ```python
    model_paths = ["path/to/model_0", "path/to/model_1", "path/to/model_2"]
    mace_calcs = MACECalculator(model_paths=model_paths)
    ```

    The variation of the `delta` parameter allow the simulation to explore the phase space more efficiently based on the variance of the forces or energies. The `delta` parameter is adjusted based on the following formula:

    Parameters
    ----------
    atoms: Atoms
        The atomic system being simulated.
    min_delta: float
        Minimum delta parameter in Angstrom.
    max_delta: float
        Maximum delta parameter in Angstrom.
    temperature: float
        The temperature of the simulation in Kelvin. Default: 298.15.
    scheme: str
        Scheme to use for variation coefficient calculation. Default: "forces".
    reference_variance: float
        Reference variance for the variation coefficient. Default: 0.1.
    update_function: str
        Update function to use for delta parameter adjustment. Default: "tanh".
    **mc_kwargs
        Additional keyword arguments to pass to the ForceBias superclass.
    """

    def __init__(
        self,
        atoms: Atoms,
        min_delta: float,
        max_delta: float,
        temperature: float = 298.15,
        scheme: str = "forces",
        reference_variance: float = 0.1,
        update_function: str = "tanh",
        **mc_kwargs,
    ):
        self.reference_variance = reference_variance

        self.min_delta = min_delta
        self.max_delta = max_delta

        self.schemes = {
            "forces": self.get_forces_variation_coef,
            "energy": self.get_energy_variation_coef,
        }
        self.scheme = scheme

        self.update_functions = {"tanh": self.tanh_update, "exp": self.exp_update}
        self.update_function = update_function

        super().__init__(atoms, 0, temperature, **mc_kwargs)

        if self.default_logger:
            if self.scheme == "forces":
                self.default_logger.add_field(
                    "MeanDelta",
                    lambda: np.mean(self.cache_delta),
                    str_format="{:>16.2f}",
                )
                self.default_logger.add_field(
                    "MinDelta", lambda: np.min(self.cache_delta), str_format="{:>16.2f}"
                )
                self.default_logger.add_field(
                    "MaxDelta", lambda: np.max(self.cache_delta), str_format="{:>16.2f}"
                )
                self.default_logger.add_field(
                    "MeanForcesVar",
                    lambda: np.mean(self.variation_coef),
                    str_format="{:>16.6f}",
                )
            elif self.scheme == "energy":
                self.default_logger.add_field(
                    "Delta", lambda: self.cache_delta, str_format="{:>16.2f}"
                )
                self.default_logger.add_field(
                    "EnergyVar", lambda: self.variation_coef, str_format="{:>16.6f}"
                )

    @property
    def delta(self) -> float:
        """
        Calculate the adaptive delta parameter based on the variation coefficient.

        Returns
        -------
        float
            The adaptive delta parameter.
        """
        self.variation_coef = self.get_variation_coef(self.atoms)

        self.cache_delta = self.min_delta + (
            self.max_delta - self.min_delta
        ) * self.update_functions[self.update_function](self.variation_coef)
        return self.cache_delta

    @delta.setter
    def delta(self, value: float):
        """
        Setter for the delta property. This is a no-op as delta is calculated dynamically. Only here for compatbility with the superclass.

        Parameters
        ----------
        value: float
            The value to set (ignored).
        """

    def get_forces_variation_coef(self, atoms: Atoms) -> NDArray:
        """
        Calculate the variation coefficient based on committee forces.

        Parameters
        ----------
        atoms: Atoms
            The atomic system being simulated.

        Returns
        -------
        NDArray
            The variation coefficient for the forces.
        """
        try:
            forces_committee = atoms.calc.results["forces_comm"]  # type: ignore
            return np.std(forces_committee, axis=0) / np.mean(
                np.abs(forces_committee), axis=0
            )
        except (KeyError, AttributeError):
            warn(
                "No committee forces available, using default reference variance.",
                stacklevel=2,
            )
            return np.full((len(atoms), 3), self.reference_variance)

    def get_energy_variation_coef(self, atoms: Atoms) -> float:
        """
        Calculate the variation coefficient based on committee energies.

        Parameters
        ----------
        atoms: Atoms
            The atomic system being simulated.

        Returns
        -------
        float
            The variation coefficient for the energies.
        """
        try:
            energies_committee = atoms.calc.results["energies"]  # type: ignore
            return np.std(energies_committee, axis=0) / len(self.atoms)
        except (KeyError, AttributeError):
            warn(
                "No committee energies available, using default reference variance.",
                stacklevel=2,
            )
            return self.reference_variance

    def tanh_update(self, variation_coefficient: float | NDArray) -> float | NDArray:
        """
        Update function using the hyperbolic tangent (tanh) function.

        Parameters
        ----------
        variation_coefficient: float | NDArray
            The variation coefficient.

        Returns
        -------
        float | NDArray
            The updated value.
        """
        return 1 - np.tanh(
            variation_coefficient / self.reference_variance * math.atanh(0.5)
        )

    def exp_update(self, variation_coefficient: float) -> float:
        """
        Update function using the exponential function.

        Parameters
        ----------
        variation_coefficient: float
            The variation coefficient.

        Returns
        -------
        float
            The updated value.
        """
        return np.exp(-variation_coefficient / self.reference_variance * math.log(2))

    @property
    def scheme(self) -> str:
        """
        Get the current scheme for variation coefficient calculation.

        Returns
        -------
        str
            The current scheme.
        """
        return self._scheme

    @scheme.setter
    def scheme(self, value: str):
        """
        Set the scheme for variation coefficient calculation.

        Parameters
        ----------
        value: str
            The scheme to set.
        """
        self._scheme = value
        self.get_variation_coef = self.schemes[value]

    @property
    def update_function(self) -> str:
        """
        Get the current update function for delta parameter adjustment.

        Returns
        -------
        str
            The current update function.
        """
        return self._update_function

    @update_function.setter
    def update_function(self, value: str):
        """
        Set the update function for delta parameter adjustment.

        Parameters
        ----------
        value: str
            The update function to set.
        """
        if value not in self.update_functions:
            raise ValueError(
                f"Invalid update function {value}, available functions: {self.update_functions}"
            )

        self._update_function = value
