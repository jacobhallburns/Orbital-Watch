# Contributing to Orbital Watch

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dev dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Create a feature branch: `git checkout -b feat/your-feature`.
4. Make your changes, add tests, run `pytest`.
5. Open a pull request against `main`.

## Guidelines

- Keep the internal `TLERecord` dataclass as the standard exchange format between all modules.
- New data sources go under `orbitalwatch/sources/`.
- All public functions should have type annotations.
- Run `ruff check .` and `black --check .` before submitting.
- Tests live in `tests/` mirroring the package structure.

## Reporting Issues

Please open a GitHub issue with a minimal reproducible example.
