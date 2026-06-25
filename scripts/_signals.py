"""Shared synthetic signal generators for the validation scripts.

Each generator returns ``(noisy, clean)`` for a given length, amplitude, width,
noise level, and seed, so the scripts draw identical profiles from one source
rather than carrying their own copies. ``SHAPES`` and ``METHODS`` are the common
sweep axes used by more than one script.
"""

import numpy as np


def _generalized_gaussian(N, amplitude, sigma, noise_std, p, seed):
    """Generalized Gaussian: p=2 is standard Gaussian, p<2 heavy-tailed, p>2 flat-topped."""
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    z = np.abs((x - center) / sigma)
    clean = amplitude * np.exp(-0.5 * z ** p)
    return clean + rng.normal(0, noise_std, N), clean


def _gaussian(N, amplitude, sigma, noise_std, seed):
    """Standard Gaussian peak (generalized Gaussian with p=2)."""
    return _generalized_gaussian(N, amplitude, sigma, noise_std, 2.0, seed)


def _gaussian_mixture(N, amplitude, sigma, noise_std, seed):
    """Sum of a narrow and broad Gaussian at the same center.

    The narrow component contributes 60% and the broad component 40% of the
    peak amplitude, producing a heavy-tailed shape outside the generalized
    Gaussian family.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    narrow = 0.6 * amplitude * np.exp(-0.5 * ((x - center) / (sigma * 0.5)) ** 2)
    broad = 0.4 * amplitude * np.exp(-0.5 * ((x - center) / (sigma * 2.0)) ** 2)
    clean = narrow + broad
    return clean + rng.normal(0, noise_std, N), clean


def _lorentzian(N, amplitude, sigma, noise_std, seed):
    """Lorentzian (Cauchy) peak, not in the generalized Gaussian family."""
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    clean = amplitude / (1.0 + ((x - center) / sigma) ** 2)
    return clean + rng.normal(0, noise_std, N), clean


SHAPES = {
    "Gaussian": lambda N, A, s, ns, seed: _generalized_gaussian(N, A, s, ns, 2.0, seed),
    "Heavy-tailed (p=1.5)": lambda N, A, s, ns, seed: _generalized_gaussian(N, A, s, ns, 1.5, seed),
    "Flat-topped (p=4)": lambda N, A, s, ns, seed: _generalized_gaussian(N, A, s, ns, 4.0, seed),
    "Gaussian mixture": _gaussian_mixture,
    "Lorentzian": _lorentzian,
}

METHODS = ["peak", "generalized_gaussian", "matched_filter"]
