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


class ForceBias(MonteCarlo):
    """Force Bias Monte Carlo class to perform simulations as described in
    https://doi.org/10.1063/1.4902136.

    Parameters
    ----------
    atoms: Atoms
        The atomic system being simulated.
    delta: float
        Delta parameter in Angstrom which influence how much the atoms are moved.
    temperature: float
        The temperature of the simulation in Kelvin. Default: 298.15.
    seed: int | None
        Seed for the random number generator, None for random seed. Default: None.

    Attributes
    ----------
    GAMMA_MAX_VALUE: float
        Maximum value for the gamma parameter, used to avoid overflow errors.
    delta: float
        Delta parameter in Angstrom which influence how much the atoms are moved.
    temperature: float
        The temperature of the simulation in Kelvin.
    """

    GAMMA_MAX_VALUE = 709.782712

    def __init__(
        self, atoms: Atoms, delta: float, temperature: float = 298.15, **mc_kwargs
    ) -> None:
        """Initialize the Force Bias Monte Carlo object."""
        self.delta = delta
        self.temperature = temperature * kB

        self._size = (atoms.get_global_number_of_atoms(), 3)
        self.update_masses(atoms.get_masses())

        self.masses_scaling_power = 0.25

        MonteCarlo.__init__(self, atoms, **mc_kwargs)

        if not has_constraint(self.atoms, "FixCom"):
            warn(
                "No `FixCom` constraint found, `ForceBias` simulations lead to sustained drift of the center of mass.",
                stacklevel=2,
            )

        self.gamma = 0.0

        if self.default_logger:
            self.default_logger.add_field(
                "Gamma/GammaMax",
                lambda: np.max(np.abs(self.gamma / self.GAMMA_MAX_VALUE)),
                str_format=">16.2f",
            )

    def _calculate_gamma(self, forces: NDArray[np.floating]) -> None:
        """Calculate the gamma parameter for the Monte Carlo step, along with the denominator for the trial probability.

        Parameters
        ----------
        forces
            The forces acting on the atoms.
        """
        self.gamma = np.clip(
            (forces * self.delta) / (2 * self.temperature),
            -self.GAMMA_MAX_VALUE,
            self.GAMMA_MAX_VALUE,
        )

        self.denominator = np.exp(self.gamma) - np.exp(-self.gamma)

    def todict(self) -> dict[str, Any]:
        """Return a dictionary representation of the object."""
        dictionary = MonteCarlo.todict(self)
        dictionary.update(
            {
                "temperature": self.temperature / kB,
                "delta": self.delta,
                "masses_scaling_power": self.masses_scaling_power,
            }
        )

        return dictionary

    @property
    def masses_scaling_power(self) -> NDArray[np.floating]:
        """Get the power of the masses scaling factor.

        The property can be set as a float, a dictionary with the element symbol as
        key and the power as value, or a numpy array with the power for each atom.
        """
        return self._masses_scaling_power

    @masses_scaling_power.setter
    def masses_scaling_power(
        self, value: dict[str, float] | NDArray[np.floating] | float
    ) -> None:
        if isinstance(value, dict):
            self._masses_scaling_power = np.full(self._size, 0.25)

            for el in np.unique(self.atoms.symbols):
                indices = self.atoms.symbols == el
                self._masses_scaling_power[indices, :] = value.get(el, 0.25)

        elif isinstance(value, float | np.floating):
            self._masses_scaling_power = np.full(self._size, value)
        elif isinstance(value, np.ndarray):
            assert value.shape == self._size
            self._masses_scaling_power = value
        else:
            raise ValueError("Invalid value type for masses_scaling_power.")

        self._mass_scaling = np.power(
            np.min(self._masses) / self._masses, self._masses_scaling_power
        )

    def update_masses(self, masses: NDArray | None = None) -> None:
        if masses is None:
            masses = self.atoms.get_masses()

        self._masses = np.broadcast_to(masses[:, np.newaxis], self._size)

    def step(self) -> NDArray[np.floating]:  # type: ignore
        """Perform one Force Bias Monte Carlo step."""
        forces = self.atoms.get_forces()
        positions = self.atoms.get_positions()
        self._size = (self.atoms.get_global_number_of_atoms(), 3)
        self._calculate_gamma(forces)

        self.zeta = self._get_zeta()

        probability_random = self._get_random()
        converged = self._calculate_trial_probability() > probability_random

        while not np.all(converged):
            self._size = probability_random[~converged].shape
            self.zeta[~converged] = self._get_zeta()

            probability_random[~converged] = self._get_random()

            converged = self._calculate_trial_probability() > probability_random

        displacement = self.zeta * self.delta * self._mass_scaling

        self.atoms.set_momenta(self._masses * displacement)
        corrected_displacement = self.atoms.get_momenta() / self._masses

        self.atoms.set_positions(positions + corrected_displacement)

        return forces

    def _get_zeta(self) -> NDArray[np.floating]:
        """Get the zeta parameter for the Monte Carlo step."""
        return self._rng.uniform(-1, 1, self._size)  # type: ignore

    def _get_random(self) -> NDArray[np.floating]:
        """Get the random parameter for the Monte Carlo step."""
        return self._rng.random(self._size)  # type: ignore

    def _calculate_trial_probability(self) -> NDArray[np.floating]:
        """Calculate the trial probability for the Monte Carlo step."""
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
