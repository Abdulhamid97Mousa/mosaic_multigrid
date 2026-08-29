# Contributing to mosaic_multigrid

Thanks for your interest in contributing. This document explains how
to report issues, propose changes, and get help.

## Reporting bugs

Open an issue at
<https://github.com/Abdulhamid97Mousa/mosaic_multigrid/issues> and
include:

- The env ID that reproduces the bug (e.g.
  `MosaicMultiGrid-BB-2v2-IndAgObs-v1`)
- A minimal script (`import mosaic_multigrid.envs`, `gym.make`,
  `reset`, `step`)
- Your Python, `gymnasium`, and `mosaic_multigrid` versions
  (`pip show mosaic_multigrid`)
- The full traceback

## Proposing changes

1. Fork the repository and create a feature branch off `main`.
2. Set up a development environment:
   ```bash
   pip install -e ".[dev]"
   ```
3. Add or update tests under `tests/` for your change.
4. Run the full suite locally:
   ```bash
   pytest tests/
   ```
   All tests must pass.
5. Open a pull request describing what you changed and why. Link the
   related issue if one exists.

## Adding new environments

If you are adding a new sport or matchup variant, follow the existing
pattern in `mosaic_multigrid/envs/`:

- Add a `*_game.py` module (or extend an existing one) with a class
  subclassing `MosaicMultiGridEnv`.
- Register the new env ID in `mosaic_multigrid/envs/__init__.py`.
- Add matched tests in `tests/` (see
  `tests/test_asymmetric_variants.py` for a parametrised example).

## Support

Bug reports, feature requests, and questions all go through
[GitHub Issues](https://github.com/Abdulhamid97Mousa/mosaic_multigrid/issues).

## License

By contributing you agree that your contributions will be licensed
under the Apache License 2.0. See
[`LICENSE`](LICENSE) for the full text.
