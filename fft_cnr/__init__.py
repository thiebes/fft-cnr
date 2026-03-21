"""
fft-cnr: FFT-based Contrast-to-Noise Ratio estimation for 1D signal profiles.

This package provides tools for estimating the Contrast-to-Noise Ratio (CNR) of
1D signal profiles using Fast Fourier Transform (FFT) analysis.

Example
-------
>>> import numpy as np
>>> from fft_cnr import estimate_cnr
>>>
>>> # Create a noisy signal
>>> signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
>>> noisy_signal = signal + 0.1 * np.random.randn(100)
>>>
>>> # Estimate CNR
>>> cnr = estimate_cnr(noisy_signal)
>>> print(f"CNR: {cnr}")
"""

from .estimator import estimate_cnr
from .version import __version__

__all__ = ["estimate_cnr", "__version__"]
