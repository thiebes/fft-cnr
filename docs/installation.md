# Installation

`fft-cnr` requires Python 3.10 or later (tested on 3.10 through 3.13). The only
runtime dependencies are numpy (>=1.24) and scipy (>=1.10), which the installer
pulls in automatically.

Install into a virtual environment so the package and its dependencies stay
isolated from your other projects:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install fft-cnr
```

## Development install

To work on the package, clone the repository and install the development
dependencies. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/thiebes/fft-cnr.git
cd fft-cnr
uv sync --group dev
```

Or with pip:

```bash
pip install -e ".[dev]"
```

See the [contributing guide](https://github.com/thiebes/fft-cnr/blob/main/CONTRIBUTING.md)
for how to run the tests and submit changes.
