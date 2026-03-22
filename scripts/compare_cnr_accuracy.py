"""Compare CNR estimation accuracy across profile types.

Run before and after code changes to verify that estimation accuracy
and precision are maintained.

Usage:
    uv run python scripts/compare_cnr_accuracy.py
"""

import numpy as np

from fft_cnr import fft_cnr

N_REALIZATIONS = 200
N_POINTS = 512


def _gaussian_profile(N, amplitude, sigma, seed):
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    clean = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    noise_std = 1.0
    return clean + rng.normal(0, noise_std, N), amplitude, noise_std


def _flat_top_profile(N, amplitude, sigma, p, seed):
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    z = np.abs((x - center) / sigma)
    clean = amplitude * np.exp(-0.5 * z**p)
    noise_std = 1.0
    return clean + rng.normal(0, noise_std, N), amplitude, noise_std


def _asymmetric_profile(N, amplitude, sigma_left, sigma_right, seed):
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    sigma = np.where(x < center, sigma_left, sigma_right)
    clean = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    noise_std = 1.0
    return clean + rng.normal(0, noise_std, N), amplitude, noise_std


def _sinusoidal_profile(N, amplitude, freq, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N, endpoint=False)
    clean = amplitude * np.sin(2 * np.pi * freq * t)
    noise_std = 1.0
    return clean + rng.normal(0, noise_std, N), amplitude, noise_std


PROFILES = {
    "Gaussian (A=20, sigma=30)": lambda seed: _gaussian_profile(
        N_POINTS, 20.0, 30.0, seed
    ),
    "Flat-top (A=20, sigma=30, p=4)": lambda seed: _flat_top_profile(
        N_POINTS, 20.0, 30.0, 4.0, seed
    ),
    "Asymmetric (A=15, sL=20, sR=40)": lambda seed: _asymmetric_profile(
        N_POINTS, 15.0, 20.0, 40.0, seed
    ),
    "Sinusoidal (A=10, f=3)": lambda seed: _sinusoidal_profile(
        N_POINTS, 10.0, 3.0, seed
    ),
}


def main():
    print(f"{'Profile':<38} {'True CNR':>10} {'Mean CNR':>10} {'Bias':>10} {'Std':>10}")
    print("-" * 88)

    for name, gen_fn in PROFILES.items():
        cnr_values = []
        for seed in range(N_REALIZATIONS):
            signal, true_amp, noise_std = gen_fn(seed)
            result = fft_cnr(signal)
            cnr_values.append(result.cnr)

        cnr_arr = np.array(cnr_values)
        true_cnr = true_amp / noise_std
        mean_cnr = np.mean(cnr_arr)
        bias = mean_cnr - true_cnr
        std = np.std(cnr_arr)

        print(f"{name:<38} {true_cnr:>10.2f} {mean_cnr:>10.2f} {bias:>+10.2f} {std:>10.2f}")


if __name__ == "__main__":
    main()
