"""Module to perform force bias Monte Carlo simulations."""

from __future__ import annotations

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

    from quansino.type_hints import Displacement, Forces, Masses, ShapedMasses


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
    masses_scaling_power: NDArray
        Power to which the mass ratio is raised to scale the displacement.
    mass_scaling: NDArray
        Scaling factors for the atomic displacements based on masses.
    shaped_masses: NDArray
        Masses of the atoms shaped for vectorized calculations.
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
                "No `FixCom` constraint found, `ForceBias` simulations can lead to sustained drift of the center of mass.",
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

    def calculate_gamma(self, forces: Forces) -> None:
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
        self, value: dict[str, float] | ShapedMasses | float
    ) -> None:
        """
        Set the power to which the mass ratio is raised to scale the displacement.

        Parameters
        ----------
        value : dict[str, float] | NDArray | float
            The power value(s). If a dict, keys are element symbols and values are the powers.
            If an NDArray, it must have shape (n_atoms, 3). If a float, the same value is used for all atoms.

        Raises
        ------
        ValueError
            If the value has an invalid type or if an NDArray with incorrect shape is provided.
        """
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

    def update_masses(self, masses: ShapedMasses | Masses | None = None) -> None:
        """
        Update the masses used for displacement scaling.

        Parameters
        ----------
        masses : Masses | None, optional
            The masses to use. If None, uses the masses from the atoms object.
        """
        if masses is None:
            masses = self.atoms.get_masses()

        if masses.ndim == 1:
            masses = np.broadcast_to(masses[:, np.newaxis], self.size)

        self.shaped_masses = masses

    def step(self) -> Forces:
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
            The zeta parameter with values uniformly distributed between -1 and 1.
        """
        return self._rng.uniform(-1, 1, self.current_size)

    def calculate_trial_probability(self) -> NDArray:
        """
        Calculate the trial probability for the Monte Carlo step based on the force bias.

        Returns
        -------
        NDArray
            The trial probability for each atom and direction.
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
