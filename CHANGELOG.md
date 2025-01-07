# Changelog

## [0.0.1]

### Added

- README.md, setting up the template and cleaning up the documentation.
- Base class for `BaseMove` and `MonteCarlo`.
- `DisplacementMove`, `ExchangeMove` for displacement and exchange moves.
- `DisplacementContext`, `ExchangeContext` for context management between moves and simulation objects.
- `Canonical` class for canonical ensemble NVT simulations.
- Various operations for `DisplacementMove` and `ExchangeMove`, such as `Ball`, `Sphere` and `Box` operations.
- `ForceBias` class based on the work of K. M Bal and E. C. Neyts, J. Chem. Phys. 141, 204104 (2014).
- Added the logo for the project.
- Set up the CI/CD pipeline with GitHub Actions.
- Various utils for supporting the project such as `search_molecules` and `reinsert_atoms`.
