"""Monte Carlo calibration of the amplitude standard error across paths.

The reported ``amplitude_se`` (and the derived ``amplitude_snr``) is computed
differently on each amplitude path: the whitened matched filter returns its
exact Cramer-Rao standard error, the generalized-Gaussian fit returns the
amplitude error from its covariance, and the default peak path returns the
proxy ``sigma / sqrt(kc_full)`` whose calibration was never characterized.

This sweep measures, for each path, the ratio of the predicted standard error
to the empirical standard deviation of the amplitude estimate over many noise
realizations. A ratio near 1 means the reported error matches the actual
scatter; above 1 means it overstates the scatter (so ``amplitude_snr`` is
understated, a conservative error); below 1 means it understates the scatter
(so ``amplitude_snr`` is overstated, the unsafe direction).

Usage:
    uv run python scripts/validate_amplitude_se.py
    uv run python scripts/validate_amplitude_se.py --trials 100   # quick check
"""

import argparse
import sys

import numpy as np

from fft_cnr import fft_cnr


# ---------------------------------------------------------------------------
# Signal generators (shared with validate_accuracy.py)
# ---------------------------------------------------------------------------

def _generalized_gaussian(N, amplitude, sigma, noise_std, p, seed):
    """Generalized Gaussian: p=2 is standard Gaussian, p<2 heavy-tailed, p>2 flat-topped."""
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    z = np.abs((x - center) / sigma)
    clean = amplitude * np.exp(-0.5 * z ** p)
    return clean + rng.normal(0, noise_std, N), clean


def _gaussian_mixture(N, amplitude, sigma, noise_std, seed):
    """Sum of a narrow and broad Gaussian; heavy-tailed, outside the GG family."""
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_condition(shape_fn, N, true_cnr, method, n_trials, sigma=20.0):
    """Calibrate the predicted amplitude SE against the empirical scatter.

    Returns the ratio of the median predicted standard error to the empirical
    standard deviation of the amplitude over the trials, plus the amplitude
    bias, for one (shape, N, CNR, method) condition.
    """
    noise_std = 1.0
    amplitude = true_cnr * noise_std

    amps = []
    ses = []
    for seed in range(n_trials):
        noisy, clean = shape_fn(N, amplitude, sigma, noise_std, seed)
        if method == "matched_filter":
            template = clean / np.max(np.abs(clean))
            result = fft_cnr(noisy, template=template)
        elif method == "generalized_gaussian":
            result = fft_cnr(noisy, fit_model="generalized_gaussian")
        else:
            result = fft_cnr(noisy)
        if np.isfinite(result.amplitude) and np.isfinite(result.amplitude_se):
            amps.append(result.amplitude)
            ses.append(result.amplitude_se)

    amps = np.array(amps)
    ses = np.array(ses)
    if len(amps) < 2:
        return {"ratio": np.nan, "amp_bias": np.nan, "n_used": len(amps)}

    empirical_sd = float(np.std(amps, ddof=1))
    predicted_se = float(np.median(ses))
    ratio = predicted_se / empirical_sd if empirical_sd > 0 else np.nan
    amp_bias = float(np.median(amps)) / amplitude
    return {"ratio": ratio, "amp_bias": amp_bias, "n_used": len(amps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=500,
        help="Monte Carlo trials per condition",
    )
    args = parser.parse_args()
    n_trials = args.trials

    cnr_values = [5, 10, 20, 50]
    n_values = [128, 256, 512]

    # Peak path across all shapes and lengths; the cross-method comparison runs
    # on the Gaussian shape only (all three paths are well posed there).
    peak_results = {}
    cross_results = {}
    total = len(SHAPES) * len(n_values) * len(cnr_values) + (
        len(METHODS) * len(cnr_values)
    )
    done = 0

    for shape_name, shape_fn in SHAPES.items():
        for N in n_values:
            for true_cnr in cnr_values:
                peak_results[(shape_name, N, true_cnr)] = run_condition(
                    shape_fn, N, true_cnr, "peak", n_trials
                )
                done += 1
                sys.stderr.write(f"\r  {done}/{total} conditions complete")
                sys.stderr.flush()

    gaussian_fn = SHAPES["Gaussian"]
    for method in METHODS:
        for true_cnr in cnr_values:
            cross_results[(method, true_cnr)] = run_condition(
                gaussian_fn, 512, true_cnr, method, n_trials
            )
            done += 1
            sys.stderr.write(f"\r  {done}/{total} conditions complete")
            sys.stderr.flush()

    sys.stderr.write("\n")

    cnr_header = f"{'Shape':<22} {'N':<6}"
    for c in cnr_values:
        cnr_header += f" {'CNR=' + str(c):>9}"

    # --- Table 1: peak-path SE calibration ratio (predicted / empirical) ---
    print("=" * 72)
    print(f"Table 1: Peak-path SE calibration ratio (predicted/empirical), "
          f"{n_trials} trials")
    print("=" * 72)
    print("ratio ~1 calibrated; >1 SE overstated (SNR low); <1 SE understated (SNR high)")
    print(cnr_header)
    print("-" * 72)
    for shape_name in SHAPES:
        for N in n_values:
            row = f"{shape_name:<22} {N:<6}"
            for true_cnr in cnr_values:
                r = peak_results[(shape_name, N, true_cnr)]
                row += f" {r['ratio']:>9.2f}"
            print(row)
    print()

    # --- Table 2: peak-path amplitude bias (median estimate / true) ---
    print("=" * 72)
    print("Table 2: Peak-path amplitude bias (median estimate / true)")
    print("=" * 72)
    print(cnr_header)
    print("-" * 72)
    for shape_name in SHAPES:
        for N in n_values:
            row = f"{shape_name:<22} {N:<6}"
            for true_cnr in cnr_values:
                r = peak_results[(shape_name, N, true_cnr)]
                row += f" {r['amp_bias']:>9.3f}"
            print(row)
    print()

    # --- Table 3: cross-method calibration ratio (Gaussian, N=512) ---
    print("=" * 72)
    print("Table 3: SE calibration ratio by method (Gaussian, N=512)")
    print("=" * 72)
    method_header = f"{'Method':<24}"
    for c in cnr_values:
        method_header += f" {'CNR=' + str(c):>9}"
    print(method_header)
    print("-" * 72)
    for method in METHODS:
        row = f"{method:<24}"
        for true_cnr in cnr_values:
            r = cross_results[(method, true_cnr)]
            row += f" {r['ratio']:>9.2f}"
        print(row)
    print()

    # --- Summary ---
    peak_ratios = [
        r["ratio"] for r in peak_results.values() if np.isfinite(r["ratio"])
    ]
    matched_ratios = [
        cross_results[("matched_filter", c)]["ratio"]
        for c in cnr_values
        if np.isfinite(cross_results[("matched_filter", c)]["ratio"])
    ]
    gg_ratios = [
        cross_results[("generalized_gaussian", c)]["ratio"]
        for c in cnr_values
        if np.isfinite(cross_results[("generalized_gaussian", c)]["ratio"])
    ]
    print("=" * 72)
    print("Summary (SE calibration ratio = predicted SE / empirical scatter):")
    print(f"  Peak path (all shapes/N/CNR):        "
          f"{min(peak_ratios):.2f} -- {max(peak_ratios):.2f}  (understates scatter)")
    print(f"  Generalized-Gaussian (Gaussian):     "
          f"{min(gg_ratios):.2f} -- {max(gg_ratios):.2f}")
    print(f"  Matched filter (Gaussian):           "
          f"{min(matched_ratios):.2f} -- {max(matched_ratios):.2f}  "
          f"(noise-only whitening)")
    print("=" * 72)


if __name__ == "__main__":
    main()
