# Changelog

## [0.0.2]

## Added

### Grand Canonical Monte Carlo
- Added implementation of Grand Canonical (μVT) Monte Carlo ensemble
- Introduced `GrandCanonical` class with support for chemical potential calculations
- Added `GrandCanonicalCriteria` for move acceptance in Grand Canonical ensemble
- Implemented particle insertion and deletion moves with volume consideration

### Monte Carlo Core Improvements
- Introduced generics for improved type safety across Monte Carlo implementations
- Added generic `Context` system for managing simulation state
- Implemented `MoveStorage` for better organization of moves and their parameters
- Added comprehensive acceptance criteria framework
- Improved move selection and probability handling

### Moves and Operations
- Added composite displacement moves through `CompositeDisplacementMove`
- Improved exchange moves with better particle tracking
- Enhanced move operations with clearer separation of concerns
- Added support for molecular insertions and deletions
- Improved handling of move constraints and boundaries

### Python Support
- Updated Python version requirements to 3.12+
- Added support for Python 3.13
- Removed support for Python 3.10 and 3.11

### Code Quality
- Added improved type hints throughout the codebase
- Enhanced error handling and validation
- Improved code organization and modularity
- Added more comprehensive unit tests
- Implemented better class inheritance structures

### Logging and I/O
- Enhanced logging system with better formatting
- Improved header handling in log files
- Added more flexible field formatting options
- Better handling of logging contexts
- Added support for stress tensor logging

## Internal Changes
- Reorganized package structure
- Improved test coverage
- Enhanced type safety with generics
- Better separation of concerns in move implementations
- Improved code reusability and maintainability

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
