"""Monte Carlo validation of the real-space noise-model detector.

Checks the signal-dependence flag's false-positive rate on white noise
(target: the 5% test level), its power and gain recovery on photon-limited
(Poisson) profiles, and the white-noise consistency between the fitted read
floor and the true noise level.

Each trial passes its own seeded Generator to fft_cnr so the null draws are
independent across trials; the default fixed-seed rng would correlate the
detection threshold between trials and distort the measured rates.

Usage:
    uv run python scripts/validate_noise_model.py
    uv run python scripts/validate_noise_model.py --trials 25   # quick check
"""

import argparse
import sys

import numpy as np

from fft_cnr import fft_cnr


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _generalized_gaussian(N, amplitude, sigma, noise_std, p, seed):
    """Generalized Gaussian with additive white noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    z = np.abs((x - center) / sigma)
    clean = amplitude * np.exp(-0.5 * z ** p)
    return clean + rng.normal(0, noise_std, N)


def _poisson_profile(N, peak_counts, sigma, seed):
    """Poisson profile whose peak holds peak_counts photons.

    In count units the true photon-transfer gain is 1.0 with no read floor,
    and the true peak SNR is sqrt(peak_counts).
    """
    rng = np.random.default_rng(seed)
    x = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    lam = peak_counts * np.exp(-0.5 * ((x - center) / sigma) ** 2) + 1e-9
    return rng.poisson(lam).astype(float)


SHAPES = {
    "Gaussian": 2.0,
    "Flat-topped (p=4)": 4.0,
}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_white_condition(p, N, true_cnr, n_trials, sigma=20.0):
    """White noise: flag rate is the false-positive rate (target 5%)."""
    flagged = 0
    skipped = 0
    gains = []
    reads = []
    for trial in range(n_trials):
        noisy = _generalized_gaussian(N, float(true_cnr), sigma, 1.0, p, trial)
        null_rng = np.random.default_rng(10_000_000 + trial)
        result = fft_cnr(noisy, estimate_noise_model=True, rng=null_rng)
        model = result.noise_model
        if model.signal_dependent is None:
            skipped += 1
            continue
        flagged += int(model.signal_dependent)
        gains.append(model.gain)
        reads.append(model.read)
    tested = n_trials - skipped
    return {
        "fpr": flagged / tested if tested else np.nan,
        "skipped": skipped,
        "median_gain": float(np.median(gains)) if gains else np.nan,
        "median_read": float(np.median(reads)) if reads else np.nan,
    }


def run_shot_condition(N, true_cnr, n_trials, sigma=20.0):
    """Poisson noise: flag rate is the power; truth is gain=1, read=0."""
    peak_counts = float(true_cnr) ** 2
    flagged = 0
    skipped = 0
    gains = []
    snr_ratios = []
    for trial in range(n_trials):
        x = _poisson_profile(N, peak_counts, sigma, trial)
        null_rng = np.random.default_rng(20_000_000 + trial)
        result = fft_cnr(x, estimate_noise_model=True, rng=null_rng)
        model = result.noise_model
        if model.signal_dependent is None:
            skipped += 1
            continue
        flagged += int(model.signal_dependent)
        gains.append(model.gain)
        snr_ratios.append(model.peak_snr(result.amplitude) / true_cnr)
    tested = n_trials - skipped
    return {
        "power": flagged / tested if tested else np.nan,
        "skipped": skipped,
        "median_gain": float(np.median(gains)) if gains else np.nan,
        "median_snr_ratio": (
            float(np.median(snr_ratios)) if snr_ratios else np.nan
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Monte Carlo trials per condition",
    )
    args = parser.parse_args()
    n_trials = args.trials

    white_cnr_values = [5, 20]
    shot_cnr_values = [5, 10, 20]
    n_values = [256, 512]

    white_results = {}
    shot_results = {}
    total = len(SHAPES) * len(n_values) * len(white_cnr_values) + (
        len(n_values) * len(shot_cnr_values)
    )
    done = 0

    for shape_name, p in SHAPES.items():
        for N in n_values:
            for true_cnr in white_cnr_values:
                white_results[(shape_name, N, true_cnr)] = run_white_condition(
                    p, N, true_cnr, n_trials
                )
                done += 1
                sys.stderr.write(f"\r  {done}/{total} conditions complete")
                sys.stderr.flush()

    for N in n_values:
        for true_cnr in shot_cnr_values:
            shot_results[(N, true_cnr)] = run_shot_condition(
                N, true_cnr, n_trials
            )
            done += 1
            sys.stderr.write(f"\r  {done}/{total} conditions complete")
            sys.stderr.flush()

    sys.stderr.write("\n")

    # --- Table 1: false-positive rate on white noise ---
    print("=" * 72)
    print(f"Table 1: Flag rate on white noise (target ~5%), {n_trials} trials")
    print("=" * 72)
    header = f"{'Shape':<22} {'N':<6}"
    for c in white_cnr_values:
        header += f" {'CNR=' + str(c):>10}"
    header += f" {'skipped':>9}"
    print(header)
    print("-" * 72)
    for shape_name in SHAPES:
        for N in n_values:
            row = f"{shape_name:<22} {N:<6}"
            n_skipped = 0
            for true_cnr in white_cnr_values:
                r = white_results[(shape_name, N, true_cnr)]
                row += f" {r['fpr']:>9.1%} "
                n_skipped += r["skipped"]
            row += f" {n_skipped:>9}"
            print(row)
    print()

    # --- Table 2: white-noise estimates (truth: gain=0, read=1) ---
    print("=" * 72)
    print("Table 2: Median estimates on white noise (truth: gain 0, read 1)")
    print("=" * 72)
    print(header)
    print("-" * 72)
    for shape_name in SHAPES:
        for N in n_values:
            row = f"{shape_name:<22} {N:<6}"
            for true_cnr in white_cnr_values:
                r = white_results[(shape_name, N, true_cnr)]
                row += f" {r['median_gain']:>+5.3f}/{r['median_read']:.2f}"
            print(row + "   gain/read")
    print()

    # --- Table 3: shot noise (truth: gain=1, read=0) ---
    print("=" * 72)
    print(f"Table 3: Shot noise (Poisson; truth gain 1, read 0), {n_trials} trials")
    print("=" * 72)
    header3 = f"{'N':<6} {'Metric':<22}"
    for c in shot_cnr_values:
        header3 += f" {'CNR=' + str(c):>10}"
    print(header3)
    print("-" * 72)
    for N in n_values:
        for metric, key, fmt in [
            ("flag rate (power)", "power", "{:>9.1%} "),
            ("median gain", "median_gain", "{:>10.3f}"),
            ("peak SNR ratio", "median_snr_ratio", "{:>10.3f}"),
        ]:
            row = f"{N:<6} {metric:<22}"
            for true_cnr in shot_cnr_values:
                row += fmt.format(shot_results[(N, true_cnr)][key])
            print(row)
        skipped = sum(
            shot_results[(N, c)]["skipped"] for c in shot_cnr_values
        )
        if skipped:
            print(f"{'':<6} ({skipped} trials skipped)")
        print()

    # --- Summary ---
    fprs = [r["fpr"] for r in white_results.values() if np.isfinite(r["fpr"])]
    powers = [
        r["power"] for r in shot_results.values() if np.isfinite(r["power"])
    ]
    gains = [
        r["median_gain"] for r in shot_results.values()
        if np.isfinite(r["median_gain"])
    ]
    print("=" * 72)
    print("Summary:")
    print(f"  White-noise flag rate range:  {min(fprs):.1%} -- {max(fprs):.1%}"
          f"  (nominal 5%)")
    print(f"  Shot-noise power range:       {min(powers):.1%} -- {max(powers):.1%}")
    print(f"  Shot-noise median gain range: {min(gains):.3f} -- {max(gains):.3f}"
          f"  (truth 1.0)")
    print("=" * 72)


if __name__ == "__main__":
    main()
