"""Trace the amplitude standard error through to the CNR confidence interval.

The companion sweep ``validate_amplitude_se.py`` shows that ``amplitude_se`` is
calibrated differently on each amplitude path: the matched filter returns the
exact closed-form error of its white-noise projection, while the peak path uses
the proxy ``sigma / sqrt(kc_full)`` that understates the empirical scatter. That
sweep stops at the standard error. This one asks the next question: does an
amplitude-SE miscalibration actually reach the shipped ``cnr_ci95``, or is it
swallowed by the chi-squared noise term?

The delta-method interval in ``fft_cnr`` combines two variance terms (see
``core.py``)::

    var_cnr = var_A / sigma**2  +  Amp**2 * var_sigma / sigma**4
              \-- amplitude --/    \------- noise (sigma) -------/

If the amplitude term is a small fraction of ``var_cnr``, then even a large
error in ``amplitude_se`` barely moves ``cnr_ci95``: the miscalibration is
latent, surfacing only when ``amplitude_snr`` exposes the standard error
directly. If the amplitude term dominates, the miscalibration is a live defect
in the shipped confidence interval.

For each path this sweep reports three quantities over many noise realizations:

- coverage: fraction of trials whose ``cnr_ci95`` brackets the true CNR
  (nominal 0.95);
- relative CI width: median full interval width divided by the true CNR;
- amplitude-term fraction: median of ``term_A / var_cnr``, the share of the
  CNR variance contributed by the amplitude standard error.

All three are reconstructed from the public ``CNREstimate`` fields, so the
script reproduces the internal computation without reaching into private state.

Usage:
    uv run python scripts/validate_cnr_ci_decomposition.py
    uv run python scripts/validate_cnr_ci_decomposition.py --trials 100   # quick
"""

import argparse
import sys

import numpy as np

from _signals import METHODS, _gaussian
from fft_cnr import fft_cnr

# The Gaussian cross-method comparison is well posed for all three amplitude
# paths, matching validate_amplitude_se.py Table 3.


def _ci_terms(result):
    """Reconstruct the two delta-method variance terms from public fields.

    Mirrors the computation in ``fft_cnr``: the amplitude term ``var_A /
    sigma**2`` and the noise term ``Amp**2 * var_sigma / sigma**4``. Returns
    (term_A, term_sigma), or (nan, nan) if the interval is undefined.
    """
    sigma = result.noise_rms
    amp = result.amplitude
    amp_se = result.amplitude_se
    lo, hi = result.noise_ci95
    if not (np.isfinite(amp_se) and np.isfinite(sigma) and sigma > 0):
        return np.nan, np.nan
    se_sigma = 0.5 * (hi - lo) / 1.96
    var_a = amp_se ** 2
    var_sigma = se_sigma ** 2
    term_a = var_a / sigma ** 2
    term_sigma = amp ** 2 * var_sigma / sigma ** 4
    return term_a, term_sigma


def run_condition(N, true_cnr, method, n_trials, sigma=20.0):
    """Coverage, CI width, and variance decomposition for one condition.

    The true CNR equals the input amplitude because the noise standard
    deviation is 1.0, so the interval should bracket ``true_cnr``.
    """
    noise_std = 1.0
    amplitude = true_cnr * noise_std

    covered = []
    rel_widths = []
    amp_fracs = []
    for seed in range(n_trials):
        noisy, clean = _gaussian(N, amplitude, sigma, noise_std, seed)
        if method == "matched_filter":
            template = clean / np.max(np.abs(clean))
            result = fft_cnr(noisy, template=template)
        elif method == "generalized_gaussian":
            result = fft_cnr(noisy, fit_model="generalized_gaussian")
        else:
            result = fft_cnr(noisy)

        lo, hi = result.cnr_ci95
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        covered.append(lo <= true_cnr <= hi)
        rel_widths.append((hi - lo) / true_cnr)

        term_a, term_sigma = _ci_terms(result)
        total = term_a + term_sigma
        if np.isfinite(total) and total > 0:
            amp_fracs.append(term_a / total)

    if not covered:
        return {"coverage": np.nan, "rel_width": np.nan,
                "amp_frac": np.nan, "n_used": 0}
    return {
        "coverage": float(np.mean(covered)),
        "rel_width": float(np.median(rel_widths)),
        "amp_frac": float(np.median(amp_fracs)) if amp_fracs else np.nan,
        "n_used": len(covered),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=500,
        help="Monte Carlo trials per condition",
    )
    args = parser.parse_args()
    n_trials = args.trials

    cnr_values = [5, 10, 20, 50]
    n_values = [256, 512]

    results = {}
    total = len(n_values) * len(METHODS) * len(cnr_values)
    done = 0
    for N in n_values:
        for method in METHODS:
            for true_cnr in cnr_values:
                results[(N, method, true_cnr)] = run_condition(
                    N, true_cnr, method, n_trials
                )
                done += 1
                sys.stderr.write(f"\r  {done}/{total} conditions complete")
                sys.stderr.flush()
    sys.stderr.write("\n")

    header = f"{'Method':<24}"
    for c in cnr_values:
        header += f" {'CNR=' + str(c):>9}"

    def _table(title, key, fmt):
        print("=" * 72)
        print(title)
        print("=" * 72)
        for N in n_values:
            print(f"-- Gaussian, N={N} " + "-" * (72 - len(f"-- Gaussian, N={N} ")))
            print(header)
            for method in METHODS:
                row = f"{method:<24}"
                for true_cnr in cnr_values:
                    v = results[(N, method, true_cnr)][key]
                    row += f" {v:>9{fmt}}" if np.isfinite(v) else f" {'nan':>9}"
                print(row)
        print()

    _table("Table 1: cnr_ci95 empirical coverage (nominal 0.95)",
           "coverage", ".3f")
    _table("Table 2: cnr_ci95 relative width (full width / true CNR)",
           "rel_width", ".3f")
    _table("Table 3: amplitude-term share of var_cnr (term_A / var_cnr)",
           "amp_frac", ".4f")

    # --- Interpretation ---
    print("=" * 72)
    print("Reading:")
    print("  Table 3 is the decisive one. If the matched-filter amplitude")
    print("  fraction is small, the inflated amplitude_se is masked by the")
    print("  noise term and barely reaches cnr_ci95 (latent). If it is large,")
    print("  the inflation is a live defect in the shipped interval -- visible")
    print("  as inflated width (Table 2) and over-coverage (Table 1).")
    print("=" * 72)


if __name__ == "__main__":
    main()
