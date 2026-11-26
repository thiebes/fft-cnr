# Contributing to fft-cnr

Contributions are welcome. This guide covers the basics for getting started.

## Development setup

Clone the repository and install development dependencies:

```bash
git clone https://github.com/thiebes/fft-cnr.git
cd fft-cnr
pip install -e ".[dev]"
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
```

## Running tests

```bash
pytest
```

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Add tests for any new functionality.
3. Make sure all tests pass before opening a pull request.
4. If you change the version or a dependency in `pyproject.toml`, run `uv lock`
   and commit `uv.lock`. CI installs with `--locked`, so a lockfile that does
   not match `pyproject.toml` fails the tests.
5. Keep pull requests focused — one fix or feature per PR.

## Reporting bugs

Open an issue on GitHub with:

- A minimal reproducing example
- Expected vs. actual behavior
- Python version and OS

## Style

- Follow existing code conventions in the project.
- No strict formatter is enforced, but keep code readable.

## Release maintenance

The release commit bumps the version in `pyproject.toml`, so it must also
update `uv.lock` (run `uv lock`). The lockfile records the package's own
version. Without the update the publish workflow's test job, which installs
with `--locked`, fails at the release.

The release build checks the build backend and its dependencies against the
hashes in `.github/build-constraints.txt`. Regenerate that file with the recipe
in its header before each release, and after any change to the hatchling
version in `pyproject.toml`. The file also pins `trove-classifiers`. A
classifier in `pyproject.toml` that is not in the pinned release fails the
build until the file is regenerated with a release that has it. The recipe
keeps every pin at least seven days old, so a classifier added to
trove-classifiers in the last seven days cannot be used yet.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
