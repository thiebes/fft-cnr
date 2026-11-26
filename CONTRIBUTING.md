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
4. Keep pull requests focused — one fix or feature per PR.

## Reporting bugs

Open an issue on GitHub with:

- A minimal reproducing example
- Expected vs. actual behavior
- Python version and OS

## Style

- Follow existing code conventions in the project.
- No strict formatter is enforced, but keep code readable.

## Release maintenance

The release build checks the build backend and its dependencies against the
hashes in `.github/build-constraints.txt`. Regenerate that file with the recipe
in its header before each release, and after any change to the hatchling
version in `pyproject.toml`. The file also pins `trove-classifiers`, so a
classifier added to `pyproject.toml` that is newer than the pinned list fails
the build until the file is regenerated.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
