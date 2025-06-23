<div align="center">
  <img src=https://raw.githubusercontent.com/Atomic-Samplers/quansino/refs/heads/main/docs/images/quansino_logo.png width="500"><br>
</div>

<div align="center">
  <h1><code>quansino</code> 🎲</h1>
  <p><i>Modular Stochastic Simulations for Atomistic Modelling</i></p>
</div>

***

[![PyPI version](https://badge.fury.io/py/quansino.svg)](https://badge.fury.io/py/quansino)
![Python Version](https://img.shields.io/pypi/pyversions/quansino)
[![codecov](https://codecov.io/gh/Atomic-Samplers/quansino/branch/main/graph/badge.svg?token=A864UNYUOG)](https://codecov.io/gh/Atomic-Samplers/quansino)
[![GitHub license](https://img.shields.io/github/license/Atomic-Samplers/quansino)](https://github.com/Atomic-Samplers/quansino/blob/main/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

`quansino` is a Python framework for running Monte Carlo simulations on atomic systems, designed to be modular and work with popular quantum chemistry codes/forcefields. The package offer a flexible interface to build custom algorithms, and is designed to be modular and extensible. It is built to work with the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) `Atoms` object.

## Key Features 🎰

- Perform simulations in various ensemble; (grand-canonical) canonical, isobaric, more will be added in the future.
- The framework allows to design custom simulation algorithms by providing explicit interfaces for each step of the simulation, such as:

  - **Moves**: Moves are the core of the simulation, allowing for the modification of atomic configurations. The framework supports a wide range of move types, including:

    - **Displacement Moves**: Moves that displace atoms in the simulation box.
    - **Cell Moves**: Moves that change the simulation box size or shape.
    - **Exchange Moves**: Moves that add/remove atoms from the simulation box.

  - **Contexts**: Hold the state of the simulation, such as temperature, pressure, and chemical potential.
  - **Criteria**: Criteria objects are used to determine the acceptance of moves based on energy changes.

- The code makes use of Python's type hints and duck typing to ensure that the code is modular and extensible. Along with an extensive documentation, this makes it easy to understand and extend the codebase.

## Documentation 🎱

The full documentation can be found [here](https://atomic-samplers.github.io/quansino/), and includes detailed instructions about:

- 🔧 [Installation](https://atomic-samplers.github.io/quansino/installation/install.html)
- 📖 [Overview](https://atomic-samplers.github.io/quansino/documentation/overview.html)
- 💡 [Examples](https://atomic-samplers.github.io/quansino/documentation/examples.html)

## Citation 🎯

If you use `quansino` in your research, please cite the following repository: https://doi.org/10.5281/zenodo.14854001

## License 🃏

This project is licensed under the terms of the BSD 3-Clause license.
