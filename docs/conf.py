"""Sphinx configuration for the fft-cnr documentation."""

from importlib.metadata import version as _installed_version

project = "fft-cnr"
author = "Joseph J. Thiebes"
copyright = "2025-2026, Joseph J. Thiebes"
release = _installed_version("fft-cnr")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Docstrings are numpydoc style.
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

myst_enable_extensions = ["dollarmath", "colon_fence"]

html_theme = "pydata_sphinx_theme"
html_title = "fft-cnr"
html_theme_options = {
    "github_url": "https://github.com/thiebes/fft-cnr",
    "navigation_with_keys": False,
}
