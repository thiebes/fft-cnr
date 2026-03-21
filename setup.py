"""Setup script for fft-cnr package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the version from version.py
version = {}
with open("fft_cnr/version.py") as f:
    exec(f.read(), version)

# Read the long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="fft-cnr",
    version=version["__version__"],
    author="Joseph J. Thiebes",
    author_email="your.email@example.com",
    description="FFT-based Contrast-to-Noise Ratio estimation for 1D signal profiles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fft-cnr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
        ],
    },
    keywords="fft cnr contrast-to-noise-ratio signal-processing noise-estimation",
)
