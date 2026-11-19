"""Record width needed by the generalized-Gaussian and peak paths.

A centred Gaussian (peak 1, zero baseline) sits in a record of half-width
k = N / (2 sigma) feature standard deviations, in white Gaussian noise of
standard deviation 1 / CNR, where CNR is the nominal CNR (the simulation input).
Each k uses three record designs: N = 32 and N = 52 with sigma = N / (2k), and
sigma = 5 samples with N = 2 round(5k). Each (k, design, CNR) cell draws its
profiles as one array from ``np.random.default_rng([35400, ki, ri, ci])``.

For each k and CNR, pooled over the designs, the script reports:

- the share of profiles whose generalized-Gaussian fit is kept with a negative
  amplitude (the wrong sign under this positive feature);
- the median CNR / nominal on the generalized-Gaussian path, fallbacks included;
- the share of fallbacks to the peak read, with the most common refusal reason;
- the median CNR / nominal on the peak path, and its noiseless prediction
  ``1 - sqrt(2 pi) [Phi(k) - Phi(k/2)] / (k/2)``: the edge baseline (the mean of
  the outer N // 4 samples) averages the feature over k/2 to k SD.

Usage:
    uv run python scripts/validate_record_width.py
    uv run python scripts/validate_record_width.py --trials 100   # quick
"""

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import norm

from fft_cnr import fft_cnr

HALF_WIDTHS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
DESIGNS = ("N = 32", "N = 52", "sigma = 5")
CNR_VALUES = (5, 8, 13, 21, 34)


def _record(ki, ri):
    """Record length and feature standard deviation of one design."""
    k = HALF_WIDTHS[ki]
    if ri == 2:
        return 2 * round(5 * k), 5.0
    N = (32, 52)[ri]
    return N, N / (2 * k)


def _run_cell(cell, trials):
    """Per-profile outcomes of one (k, design, CNR) cell."""
    ki, ri, ci = cell
    N, sigma = _record(ki, ri)
    cnr = CNR_VALUES[ci]
    grid = np.arange(N, dtype=float)
    feature = np.exp(-0.5 * ((grid - (N - 1) / 2) / sigma) ** 2)
    profiles = feature + np.random.default_rng([35400, ki, ri, ci]).normal(0, 1 / cnr, (trials, N))
    outcomes = []
    for profile in profiles:
        fit = fft_cnr(profile, fit_model="generalized_gaussian")
        peak = fft_cnr(profile)
        kept = fit.diagnostics["amplitude_method"] == "generalized_gaussian_fit"
        outcomes.append({
            "wrong_sign": kept and fit.amplitude < 0,
            "fallback": not kept,
            "reason": fit.diagnostics.get("gaussian_fit_rejection_reason", "fit failed"),
            "fit_ratio": fit.cnr / cnr,
            "peak_ratio": peak.cnr / cnr,
        })
    return cell, outcomes


def _predicted_peak_read(k):
    return 1.0 - np.sqrt(2 * np.pi) * (norm.cdf(k) - norm.cdf(k / 2)) / (k / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=600, help="profiles per cell")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="worker processes (1 runs serially)")
    args = parser.parse_args()

    cells = [(ki, ri, ci) for ki in range(len(HALF_WIDTHS)) for ri in range(len(DESIGNS))
             for ci in range(len(CNR_VALUES))]
    trials = [args.trials] * len(cells)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = dict(pool.map(_run_cell, cells, trials))
    else:
        results = {}
        for done, (cell, outcomes) in enumerate(map(_run_cell, cells, trials), 1):
            results[cell] = outcomes
            sys.stderr.write(f"\r  {done}/{len(cells)} cells")
        sys.stderr.write("\n")

    print(f"Centred Gaussian in a record of +/-k SD, {args.trials} profiles per (k, design, CNR) "
          f"cell, designs {', '.join(DESIGNS)} pooled.")
    print("Columns per CNR: wrong-sign share / fit-path median / fallback share / "
          "peak-path median (CNR / nominal).")
    print(f"{'record':<10} {'CNR':>4} {'wrong sign':>11} {'fit median':>11} {'fallback':>9} "
          f"{'peak median':>12} {'predicted':>10}  most common refusal")
    for ki, k in enumerate(HALF_WIDTHS):
        for ci, cnr in enumerate(CNR_VALUES):
            pooled = [o for ri in range(len(DESIGNS)) for o in results[(ki, ri, ci)]]
            reasons = Counter(o["reason"] for o in pooled if o["fallback"])
            common = reasons.most_common(1)[0][0] if reasons else "-"
            print(f"+/-{k:<7g} {cnr:>4} {np.mean([o['wrong_sign'] for o in pooled]):>11.3f} "
                  f"{np.median([o['fit_ratio'] for o in pooled]):>11.2f} "
                  f"{np.mean([o['fallback'] for o in pooled]):>9.3f} "
                  f"{np.median([o['peak_ratio'] for o in pooled]):>12.2f} "
                  f"{_predicted_peak_read(k):>10.3f}  {common}")
    print()
    print("Wrong-sign share by design, CNR " + ", ".join(map(str, CNR_VALUES)) + ":")
    for ki, k in enumerate(HALF_WIDTHS):
        parts = []
        for ri, design in enumerate(DESIGNS):
            N, _ = _record(ki, ri)
            shares = [np.mean([o["wrong_sign"] for o in results[(ki, ri, ci)]])
                      for ci in range(len(CNR_VALUES))]
            parts.append(f"{design} (N {N}): " + " ".join(f"{s:.3f}" for s in shares))
        print(f"+/-{k:g} SD  " + "; ".join(parts))


if __name__ == "__main__":
    main()
