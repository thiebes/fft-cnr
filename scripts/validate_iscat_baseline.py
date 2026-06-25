"""Monte Carlo validation of the low-frequency-baseline guard on iSCAT-like data.

Interferometric scattering (iSCAT) is the hard case for the baseline-leakage
behavior tracked in issue #7: the particle contrast is a small point-scatterer
signal sitting on a much larger structured background. This script checks the
``lowfreq_dominated`` diagnostic and the ``roi`` remedy against an iSCAT
surrogate built *after* background subtraction, where a residual structured
field remains.

The surrogate is deliberately not the single cosine the guard's threshold was
calibrated on -- that would be circular. Instead:

- Signal: an iSCAT interferometric point-spread function on a line cut through
  the particle center -- a Gaussian core with damped oscillatory side rings.
- Residual background: a broadband, Lorentzian-correlated random field (power
  spectrum ``1 / (k**2 + k0**2)``, DC removed) with fresh random phases every
  trial, scaled to a chosen multiple of the noise RMS. This represents the
  structured residual left by imperfect ratiometric/differential subtraction
  (slow drift plus residual speckle), and carries power across many low-to-mid
  spatial frequencies rather than at one frequency.
- Noise: white Gaussian (the shot-plus-read floor after subtraction), RMS 1, so
  the true CNR equals the central contrast amplitude.

This is a simulation standing in for real iSCAT frames; treat the numbers as a
behavioral check of the guard against realistic structured background, not as a
calibration against measured data.

Usage:
    uv run python scripts/validate_iscat_baseline.py
    uv run python scripts/validate_iscat_baseline.py --trials 40   # quick check
"""

import argparse
import sys

import numpy as np

from fft_cnr import fft_cnr


# ---------------------------------------------------------------------------
# iSCAT surrogate generators
# ---------------------------------------------------------------------------

def _ipsf(N, amplitude, center, sigma, ring):
    """iSCAT interferometric PSF line cut: Gaussian core with damped rings.

    ``ring`` is the ring wavelength in units of ``sigma`` (the cosine factor is
    unity at the center, so the central contrast is ``amplitude``); larger
    ``ring`` weakens the side lobes. The rings sit within a few ``sigma`` of the
    center, well inside the peak window the guard excludes.
    """
    x = np.arange(N, dtype=float)
    z = (x - center) / sigma
    envelope = np.exp(-0.5 * z**2)
    return amplitude * envelope * np.cos(2 * np.pi * (x - center) / (ring * sigma))


def _structured_background(N, rms, corr_len, rng):
    """Zero-mean Lorentzian-correlated random field (residual after subtraction).

    The power spectrum ``1 / (k**2 + k0**2)`` with ``k0 = 1 / corr_len`` gives a
    field with exponential spatial correlation of length ``corr_len`` and most
    of its power at low frequency, like residual drift plus speckle. DC is
    removed (background subtraction removes the mean). Random phases make each
    realization independent.
    """
    k = np.fft.rfftfreq(N)
    k0 = 1.0 / max(corr_len, 1)
    amp = 1.0 / np.sqrt(k**2 + k0**2)
    amp[0] = 0.0
    phases = rng.uniform(0.0, 2 * np.pi, len(k))
    field = np.fft.irfft(amp * np.exp(1j * phases), n=N)
    std = float(np.std(field))
    return field * (rms / std) if std > 0 else field


def _profile(N, true_cnr, beta, *, center, sigma, ring, corr_len, seed):
    """One iSCAT line cut: iPSF + residual background (beta*noise) + white noise."""
    rng = np.random.default_rng(seed)
    signal = _ipsf(N, float(true_cnr), center, sigma, ring)
    background = (
        _structured_background(N, beta, corr_len, rng) if beta > 0 else 0.0
    )
    noise = rng.normal(0.0, 1.0, N)  # noise RMS 1 -> true CNR == amplitude
    return signal + background + noise


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_condition(true_cnr, beta, n_trials, *, N, center, sigma, ring, corr_len):
    """Estimate CNR full-profile, with auto ROI, and an explicit peak window."""
    half = max(8, int(round(2.5 * sigma)))
    roi_explicit = (int(center - half), int(center + half))

    cnr_full, cnr_auto, cnr_roi, ratios, flags = [], [], [], [], []
    for trial in range(n_trials):
        y = _profile(
            N, true_cnr, beta, center=center, sigma=sigma, ring=ring,
            corr_len=corr_len, seed=trial,
        )
        full = fft_cnr(y)
        cnr_full.append(full.cnr)
        ratios.append(full.diagnostics["lowfreq_offpeak_ratio"])
        flags.append(int(full.diagnostics["lowfreq_dominated"]))
        cnr_auto.append(fft_cnr(y, roi="auto").cnr)
        cnr_roi.append(fft_cnr(y, roi=roi_explicit).cnr)

    return {
        "cnr_full": float(np.mean(cnr_full)),
        "cnr_auto": float(np.mean(cnr_auto)),
        "cnr_roi": float(np.mean(cnr_roi)),
        "ratio": float(np.nanmean(ratios)),
        "flag_rate": float(np.mean(flags)),
    }


def run_matched_condition(true_cnr, beta, n_trials, *, N, center, sigma, ring,
                          corr_len):
    """Matched-filter path: project onto the known iPSF template.

    iSCAT detection is matched-filtering against the interferometric PSF. The
    projection rejects background orthogonal to the template, so the amplitude
    and its SNR (``amplitude_snr``) resist structured background that the
    peak-method ``cnr`` cannot.
    """
    template = _ipsf(N, 1.0, center, sigma, ring)
    amps, snrs = [], []
    for trial in range(n_trials):
        y = _profile(
            N, true_cnr, beta, center=center, sigma=sigma, ring=ring,
            corr_len=corr_len, seed=trial,
        )
        result = fft_cnr(y, template=template)
        amps.append(result.amplitude)
        snrs.append(result.amplitude_snr)
    return {"amp": float(np.mean(amps)), "amp_snr": float(np.nanmean(snrs))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Monte Carlo trials per condition",
    )
    args = parser.parse_args()
    n_trials = args.trials

    N, center, sigma, ring, corr_len = 128, 64.0, 4.0, 5.0, 16.0
    true_cnrs = [8.0, 4.0, 2.0, 1.0, 0.0]   # 0.0 is the peakless control
    betas = [0.0, 1.0, 3.0]                  # residual background / noise RMS

    results = {}
    total = len(true_cnrs) * len(betas)
    done = 0
    for beta in betas:
        for c in true_cnrs:
            results[(beta, c)] = run_condition(
                c, beta, n_trials,
                N=N, center=center, sigma=sigma, ring=ring, corr_len=corr_len,
            )
            done += 1
            sys.stderr.write(f"\r  {done}/{total} conditions complete")
            sys.stderr.flush()
    sys.stderr.write("\n")

    print("=" * 78)
    print("iSCAT surrogate (after background subtraction), "
          f"{n_trials} trials, N={N}")
    print(f"iPSF sigma={sigma:g}, ring={ring:g} sigma; "
          f"background: Lorentzian field, corr_len={corr_len:g} px")
    print("beta = residual background RMS / noise RMS; true CNR = central "
          "contrast / noise")
    print("=" * 78)

    # --- Table 1: CNR estimate vs true, full profile vs ROI remedies ---
    print("\nTable 1: estimated CNR (full / auto-ROI / explicit-ROI) vs true")
    print("-" * 78)
    print(f"{'beta':>5} {'true':>6} {'full':>9} {'auto-ROI':>10} "
          f"{'expl-ROI':>10}")
    for beta in betas:
        for c in true_cnrs:
            r = results[(beta, c)]
            print(f"{beta:>5.0f} {c:>6.1f} {r['cnr_full']:>9.2f} "
                  f"{r['cnr_auto']:>10.2f} {r['cnr_roi']:>10.2f}")
        print("-" * 78)

    # --- Table 2: guard behavior (off-peak ratio and flag rate) ---
    print("\nTable 2: lowfreq guard -- off-peak ratio (threshold 2.5) and "
          "flag rate")
    print("-" * 78)
    print(f"{'beta':>5} {'true':>6} {'ratio':>9} {'flag_rate':>11}")
    for beta in betas:
        for c in true_cnrs:
            r = results[(beta, c)]
            print(f"{beta:>5.0f} {c:>6.1f} {r['ratio']:>9.2f} "
                  f"{r['flag_rate']:>10.0%}")
        print("-" * 78)

    # --- Table 3: where the guard fires vs background correlation length ---
    # Peakless control at fixed beta: smooth (long-correlation) drift puts power
    # below the knee and trips the guard; short-correlation speckle leaks above
    # the knee into the noise estimate and is missed.
    print("\nTable 3: peakless control (true CNR 0, beta=3) vs correlation "
          "length")
    print("-" * 78)
    print(f"{'corr_len':>9} {'full_cnr':>9} {'roi_cnr':>9} {'ratio':>8} "
          f"{'flag_rate':>11}")
    for cl in [8.0, 16.0, 32.0, 64.0, 128.0]:
        r = run_condition(
            0.0, 3.0, n_trials,
            N=N, center=center, sigma=sigma, ring=ring, corr_len=cl,
        )
        print(f"{cl:>9.0f} {r['cnr_full']:>9.2f} {r['cnr_roi']:>9.2f} "
              f"{r['ratio']:>8.2f} {r['flag_rate']:>10.0%}")
    print("-" * 78)

    # --- Table 4: matched-filter path with the known iPSF template ---
    print("\nTable 4: matched filter vs the iPSF -- amplitude and amplitude_snr")
    print("(true amplitude == true CNR; the projection rejects orthogonal "
          "background)")
    print("-" * 78)
    print(f"{'beta':>5} {'true':>6} {'mf_amp':>9} {'amp_snr':>10}")
    for beta in betas:
        for c in true_cnrs:
            r = run_matched_condition(
                c, beta, n_trials,
                N=N, center=center, sigma=sigma, ring=ring, corr_len=corr_len,
            )
            print(f"{beta:>5.0f} {c:>6.1f} {r['amp']:>9.2f} "
                  f"{r['amp_snr']:>10.2f}")
        print("-" * 78)

    # --- Findings summary ---
    clean_flagrate = max(results[(0.0, c)]["flag_rate"] for c in true_cnrs)
    clean_tracks = all(
        abs(results[(0.0, c)]["cnr_full"] - c) <= 0.2 * c + 0.5
        for c in true_cnrs if c > 0
    )
    # Ordering separation at heavy background: a true peak vs a peakless frame.
    sep_full = results[(3.0, 8.0)]["cnr_full"] - results[(3.0, 0.0)]["cnr_full"]
    sep_roi = results[(3.0, 8.0)]["cnr_roi"] - results[(3.0, 0.0)]["cnr_roi"]
    print("\nFindings:")
    print(f"  clean (beta=0) tracks true CNR, no false flags: "
          f"{'yes' if clean_tracks and clean_flagrate <= 0.10 else 'no'} "
          f"(max flag rate {clean_flagrate:.0%})")
    print("  broadband background corrupts CNR by inflating the noise estimate")
    print("  and adding low-frequency amplitude, compressing the ordering:")
    print(f"    beta=3  true-8 minus peakless:  full {sep_full:+.2f}   "
          f"explicit-ROI {sep_roi:+.2f}")
    print("  the lowfreq guard (tuned to smooth baselines) fires on long-")
    print("  correlation drift but misses short-correlation speckle residual")
    print("  (see Table 3); broadband structured background is correlated noise,")
    print("  which single-frame estimation cannot characterize (see NoiseModel).")
    print("  Recommended for iSCAT: the matched-filter path with the iPSF")
    print("  template (Table 4) -- its amplitude and amplitude_snr stay")
    print("  background-robust where the peak-method cnr does not.")


if __name__ == "__main__":
    main()
