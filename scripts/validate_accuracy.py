"""Monte Carlo validation of fft_cnr accuracy, precision, and CI coverage.

Sweeps over signal shapes, true CNR values, profile lengths, and amplitude
estimation methods.  Reports bias (median estimated/true CNR), precision
(relative standard deviation), and empirical 95% CI coverage.

Usage:
    uv run python scripts/validate_accuracy.py
    uv run python scripts/validate_accuracy.py --trials 50   # quick check
"""

import argparse
import sys

import numpy as np

from fft_cnr import fft_cnr


# ---------------------------------------------------------------------------
# Signal generators
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_condition(shape_name, shape_fn, N, true_cnr, method, n_trials, sigma=20.0):
    """Run n_trials Monte Carlo realizations for a single condition.

    Returns dict with bias, precision, coverage, and CI width statistics.
    """
    noise_std = 1.0
    amplitude = true_cnr * noise_std

    cnr_vals = []
    ci_covers = []
    ci_widths = []

    for seed in range(n_trials):
        noisy, clean = shape_fn(N, amplitude, sigma, noise_std, seed)

        if method == "matched_filter":
            # Normalize template to unit peak so the matched filter
            # returns the amplitude as its scaling factor.
            template = clean / np.max(np.abs(clean))
            result = fft_cnr(noisy, template=template)
        elif method == "generalized_gaussian":
            result = fft_cnr(noisy, fit_model="generalized_gaussian")
        else:
            result = fft_cnr(noisy)

        cnr_vals.append(result.cnr)

        lo, hi = result.cnr_ci95
        if np.isfinite(lo) and np.isfinite(hi):
            ci_covers.append(lo <= true_cnr <= hi)
            ci_widths.append(hi - lo)
        else:
            ci_covers.append(np.nan)
            ci_widths.append(np.nan)

    cnr_arr = np.array(cnr_vals)
    ci_arr = np.array(ci_covers, dtype=float)

    median_ratio = float(np.median(cnr_arr) / true_cnr)
    rel_std = float(np.std(cnr_arr) / true_cnr)

    finite_ci = ci_arr[np.isfinite(ci_arr)]
    coverage = float(np.mean(finite_ci)) if len(finite_ci) > 0 else np.nan
    ci_frac = len(finite_ci) / len(ci_arr)

    return {
        "shape": shape_name,
        "N": N,
        "true_cnr": true_cnr,
        "method": method,
        "median_ratio": median_ratio,
        "rel_std": rel_std,
        "coverage": coverage,
        "ci_fraction": ci_frac,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200, help="Monte Carlo trials per condition")
    args = parser.parse_args()

    n_trials = args.trials
    cnr_values = [3, 5, 10, 20, 50, 100]
    n_values = [128, 256, 512]
    methods = ["peak", "generalized_gaussian", "matched_filter"]

    results = []

    total = len(SHAPES) * len(n_values) * len(cnr_values) * len(methods)
    done = 0

    for shape_name, shape_fn in SHAPES.items():
        for N in n_values:
            for true_cnr in cnr_values:
                for method in methods:
                    r = run_condition(shape_name, shape_fn, N, true_cnr, method, n_trials)
                    results.append(r)
                    done += 1
                    sys.stderr.write(f"\r  {done}/{total} conditions complete")
                    sys.stderr.flush()

    sys.stderr.write("\n")

    # --- Summary tables ---

    # Table 1: CNR bias by shape and method (N=512, all CNR values)
    print("=" * 90)
    print("Table 1: Median CNR ratio (estimated / true) at N=512")
    print("=" * 90)
    header = f"{'Shape':<25} {'Method':<22}"
    for c in cnr_values:
        header += f" {'CNR=' + str(c):>8}"
    print(header)
    print("-" * 90)

    for shape_name in SHAPES:
        for method in methods:
            row = f"{shape_name:<25} {method:<22}"
            for true_cnr in cnr_values:
                match = [r for r in results
                         if r["shape"] == shape_name and r["N"] == 512
                         and r["true_cnr"] == true_cnr and r["method"] == method]
                if match:
                    row += f" {match[0]['median_ratio']:>8.3f}"
                else:
                    row += f" {'---':>8}"
            print(row)
        print()

    # Table 2: Precision (relative std) by shape and method (N=512)
    print()
    print("=" * 90)
    print("Table 2: Relative standard deviation at N=512")
    print("=" * 90)
    print(header)
    print("-" * 90)

    for shape_name in SHAPES:
        for method in methods:
            row = f"{shape_name:<25} {method:<22}"
            for true_cnr in cnr_values:
                match = [r for r in results
                         if r["shape"] == shape_name and r["N"] == 512
                         and r["true_cnr"] == true_cnr and r["method"] == method]
                if match:
                    row += f" {match[0]['rel_std']:>8.3f}"
                else:
                    row += f" {'---':>8}"
            print(row)
        print()

    # Table 3: CI coverage by shape and method (N=512)
    print()
    print("=" * 90)
    print("Table 3: 95% CI coverage at N=512")
    print("=" * 90)
    print(header)
    print("-" * 90)

    for shape_name in SHAPES:
        for method in methods:
            row = f"{shape_name:<25} {method:<22}"
            for true_cnr in cnr_values:
                match = [r for r in results
                         if r["shape"] == shape_name and r["N"] == 512
                         and r["true_cnr"] == true_cnr and r["method"] == method]
                if match:
                    cov = match[0]["coverage"]
                    if np.isfinite(cov):
                        row += f" {cov:>7.1%} "
                    else:
                        row += f" {'N/A':>8}"
                else:
                    row += f" {'---':>8}"
            print(row)
        print()

    # Table 4: Effect of profile length (peak method, Gaussian shape)
    print()
    print("=" * 90)
    print("Table 4: Effect of profile length (peak method, Gaussian shape)")
    print("=" * 90)
    header4 = f"{'N':<8}"
    for c in cnr_values:
        header4 += f" {'CNR=' + str(c):>8}"
    print(f"{'':>8} {'Median ratio':>{8 * len(cnr_values)}}")
    print(header4)
    print("-" * 60)

    for N in n_values:
        row = f"{N:<8}"
        for true_cnr in cnr_values:
            match = [r for r in results
                     if r["shape"] == "Gaussian" and r["N"] == N
                     and r["true_cnr"] == true_cnr and r["method"] == "peak"]
            if match:
                row += f" {match[0]['median_ratio']:>8.3f}"
            else:
                row += f" {'---':>8}"
        print(row)

    print()

    # Summary statistics
    peak_512 = [r for r in results if r["N"] == 512 and r["method"] == "peak"]
    all_ratios = [r["median_ratio"] for r in peak_512]
    all_coverage = [r["coverage"] for r in peak_512 if np.isfinite(r["coverage"])]
    print("=" * 60)
    print("Summary (peak method, N=512, all shapes and CNR values):")
    print(f"  Median CNR ratio range: {min(all_ratios):.3f} -- {max(all_ratios):.3f}")
    print(f"  CI coverage range:      {min(all_coverage):.1%} -- {max(all_coverage):.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
