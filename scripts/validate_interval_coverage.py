"""Coverage of the noise and CNR confidence intervals, by degrees-of-freedom rule.

``noise_ci95`` is a chi-squared interval on the noise variance, so its width is
set by the degrees of freedom assigned to the estimate. The estimate is the
tapered, band-limited power of the full record, and ``fft_cnr`` takes the
degrees of freedom of that quadratic form (``_noise_band_dof`` in ``core.py``).
This sweep measures the coverage that results and compares the rule with three
alternatives, all evaluated on the same draws from public ``CNREstimate``
fields:

- ``welch``: twice the Welch segment count times the kept-band fraction. The
  Welch spectrum only locates the knee, so this counts the wrong sample.
- ``taper``: ``N * mean(w**2)**2 / mean(w**4)`` times the kept-band fraction.
- ``band``: ``2 * (N/2 + 1 - kc) * mean(w**2)**2 / mean(w**4)``.
- ``exact``: the quadratic-form value reported in ``diagnostics["dof"]``.

The ``welch`` rule is kept for comparison only: its columns pair the old
degrees of freedom with the amplitude standard errors the package reports now,
so they do not reproduce an earlier release.

Four checks, all with seeded generators:

1. ``noise_ci95`` coverage of the true noise level on white Gaussian noise,
   alone and with a Gaussian feature, over profile length.
2. ``cnr_ci95`` coverage and relative width on each amplitude path, with the
   calibration ratios (mean reported standard error over empirical standard
   deviation) of the amplitude, the noise level, and the CNR.
3. The same quantities under Poisson shot noise, where the white-noise forms
   are not exact, with and without ``estimate_noise_model``. With it, the
   intervals use the fitted photon-transfer variance when the detector finds
   signal-dependent noise (``diagnostics["interval_noise"]``).
4. Over a grid of background level and CNR under shot noise, how often that
   variance is applied, how often the detector declines to fit, and the
   coverage in each subset, against two reference noise levels.

The CNR interval is rebuilt from public fields with the delta-method formula
of ``fft_cnr``::

    var_cnr = amplitude_se**2 / sigma**2 + amplitude**2 * se_sigma**2 / sigma**4
    se_sigma = (noise_ci95[1] - noise_ci95[0]) / (2 * 1.96)
    cnr_ci95 = cnr -/+ 1.96 * sqrt(var_cnr)

which treats the amplitude and noise estimates as independent. The script
checks that the rebuilt ``exact`` interval equals the one the package returns.

Usage:
    uv run python scripts/validate_interval_coverage.py
    uv run python scripts/validate_interval_coverage.py --trials 200   # quick
    uv run python scripts/validate_interval_coverage.py --checks 1 3
    uv run python scripts/validate_interval_coverage.py --checks 1 2 --lengths 28 40 52
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal.windows import tukey
from scipy.stats import chi2

from _signals import METHODS, _gaussian
from fft_cnr import fft_cnr

RULES = ["welch", "taper", "band", "exact"]
TUKEY_ALPHA = 0.25
# Feature width as a fraction of the record, 25 samples at N = 512.
WIDTH_TO_LENGTH = 25.0 / 512.0


def _candidate_dofs(result):
    """Degrees of freedom under each rule, from public fields of one result."""
    diags = result.diagnostics
    N = diags["N"]
    nbins = N // 2 + 1
    kept = diags["kept_bins"]
    frac_kept = kept / nbins
    w = tukey(N, alpha=TUKEY_ALPHA)
    taper_factor = np.mean(w**2) ** 2 / np.mean(w**4)
    step = diags["welch_nperseg"] - diags["welch_noverlap"]
    n_segments = len(range(0, N - diags["welch_nperseg"] + 1, step))
    return {
        "welch": float(max(2, int(2 * n_segments * frac_kept))),
        "taper": N * taper_factor * frac_kept,
        "band": 2.0 * kept * taper_factor,
        "exact": float(diags["dof"]),
    }


def _noise_interval(sigma, nu):
    """Chi-squared 95% interval on the noise level (arrays broadcast)."""
    lower = np.sqrt(nu * sigma**2 / chi2.ppf(0.975, nu))
    upper = np.sqrt(nu * sigma**2 / chi2.ppf(0.025, nu))
    return lower, upper


def _cnr_interval(amplitude, amplitude_se, sigma, noise_lo, noise_hi):
    """Delta-method 95% interval on the CNR, as combined in ``fft_cnr``."""
    se_sigma = 0.5 * (noise_hi - noise_lo) / 1.96
    var_cnr = amplitude_se**2 / sigma**2 + amplitude**2 * se_sigma**2 / sigma**4
    se_cnr = np.sqrt(var_cnr)
    cnr = np.abs(amplitude) / sigma
    return cnr - 1.96 * se_cnr, cnr + 1.96 * se_cnr, se_cnr, se_sigma


def _estimate(noisy, clean, method, noise_model=False):
    kwargs = {"estimate_noise_model": noise_model}
    if method == "matched_filter":
        return fft_cnr(noisy, template=clean / np.max(np.abs(clean)), **kwargs)
    if method == "generalized_gaussian":
        return fft_cnr(noisy, fit_model="generalized_gaussian", **kwargs)
    return fft_cnr(noisy, **kwargs)


def _collect(profiles, method, noise_model=False):
    """Run one cell and return per-trial arrays of the public fields."""
    rows = {key: [] for key in ("amplitude", "amplitude_se", "sigma", "cnr",
                                "cnr_lo", "cnr_hi", "noise_lo", "noise_hi",
                                "signal_dependent", "model_skipped")}
    dofs = {rule: [] for rule in RULES}
    for noisy, clean in profiles:
        result = _estimate(noisy, clean, method, noise_model)
        diags = result.diagnostics
        rows["signal_dependent"].append(diags["interval_noise"] == "signal_dependent")
        rows["model_skipped"].append("noise_model_skipped" in diags)
        rows["amplitude"].append(result.amplitude)
        rows["amplitude_se"].append(result.amplitude_se)
        rows["sigma"].append(result.noise_rms)
        rows["cnr"].append(result.cnr)
        rows["cnr_lo"].append(result.cnr_ci95[0])
        rows["cnr_hi"].append(result.cnr_ci95[1])
        rows["noise_lo"].append(result.noise_ci95[0])
        rows["noise_hi"].append(result.noise_ci95[1])
        for rule, value in _candidate_dofs(result).items():
            dofs[rule].append(value)
    cell = {key: np.asarray(values, float) for key, values in rows.items()}
    cell["dof"] = {rule: np.asarray(values, float) for rule, values in dofs.items()}
    return cell


def _profiles(spec, seeds):
    """Profiles of one cell. spec is ("white", (N, cnr)) or
    ("shot", (N, background, bl_cnr, width)); each seed gives one draw."""
    kind, params = spec
    if kind == "white":
        N, true_cnr = params
        for seed in seeds:
            yield _gaussian(N, float(true_cnr), WIDTH_TO_LENGTH * N, 1.0, seed)
        return
    N, background, bl_cnr, width = params
    grid = np.arange(N, dtype=float)
    shape = np.exp(-0.5 * ((grid - (N - 1) / 2.0) / width) ** 2)
    rate = background + bl_cnr * np.sqrt(background) * shape
    for seed in seeds:
        rng = np.random.default_rng(seed)
        yield rng.poisson(rate).astype(float), shape


def _collect_chunk(spec, method, noise_model, seeds):
    return _collect(_profiles(spec, seeds), method, noise_model)


# Worker pool for _run_cell, set in main(); None runs serially.
_POOL = None


def _run_cell(spec, method, n_trials, noise_model=False):
    """Collect one cell, splitting its seeds across the worker pool. Results
    are concatenated in seed order, so they match a serial run."""
    if _POOL is None:
        return _collect_chunk(spec, method, noise_model, range(n_trials))
    n_chunks = min(n_trials, 4 * _POOL._max_workers)
    chunks = [list(c) for c in np.array_split(np.arange(n_trials), n_chunks) if len(c)]
    parts = list(_POOL.map(_collect_chunk, [spec] * len(chunks), [method] * len(chunks),
                           [noise_model] * len(chunks), chunks))
    cell = {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if key != "dof"}
    cell["dof"] = {rule: np.concatenate([p["dof"][rule] for p in parts]) for rule in RULES}
    return cell


def _summarize(cell, true_sigma, true_cnr):
    """Coverage, width, and calibration ratios for one cell, per rule."""
    sigma = cell["sigma"]
    amplitude = cell["amplitude"]
    amplitude_se = cell["amplitude_se"]
    summary = {
        "sigma_ratio": float(np.median(sigma) / true_sigma),
        "sigma_rsd": float(np.std(sigma) / np.mean(sigma)),
        "cnr_ratio": float(np.median(cell["cnr"]) / true_cnr) if true_cnr else np.nan,
        "amp_se_ratio": float(np.nanmean(amplitude_se) / np.std(amplitude)),
    }
    summary["implied_dof"] = 1.0 / (2.0 * summary["sigma_rsd"] ** 2)
    for rule in RULES:
        nu = cell["dof"][rule]
        noise_lo, noise_hi = _noise_interval(sigma, nu)
        cnr_lo, cnr_hi, se_cnr, se_sigma = _cnr_interval(
            amplitude, amplitude_se, sigma, noise_lo, noise_hi
        )
        if rule == "exact":
            # The rebuilt interval must be the one the package returns.
            assert np.allclose(noise_lo, cell["noise_lo"], rtol=1e-12)
            assert np.allclose(noise_hi, cell["noise_hi"], rtol=1e-12)
            assert np.allclose(cnr_lo, cell["cnr_lo"], rtol=1e-9, atol=1e-12)
            assert np.allclose(cnr_hi, cell["cnr_hi"], rtol=1e-9, atol=1e-12)
        summary[rule] = {
            "mean_dof": float(np.mean(nu)),
            "noise_coverage": float(
                np.mean((noise_lo <= true_sigma) & (true_sigma <= noise_hi))
            ),
            "noise_se_ratio": float(np.mean(se_sigma) / np.std(sigma)),
            "cnr_coverage": (
                float(np.mean((cnr_lo <= true_cnr) & (true_cnr <= cnr_hi)))
                if true_cnr else np.nan
            ),
            "cnr_rel_width": (
                float(np.median(cnr_hi - cnr_lo) / true_cnr) if true_cnr else np.nan
            ),
            "cnr_se_ratio": float(np.mean(se_cnr) / np.std(cell["cnr"])),
        }
    return summary



def _progress(done, total):
    sys.stderr.write(f"\r  {done}/{total} cells complete")
    sys.stderr.flush()
    if done == total:
        sys.stderr.write("\n")


def check_noise_interval(n_trials, lengths=None):
    lengths = lengths or [64, 128, 256, 512, 1024]
    cnr_values = [0, 5, 20, 100]
    print("=" * 100)
    print("Check 1: noise_ci95 coverage of the true noise level (nominal 0.95)")
    print(f"white Gaussian noise, Gaussian feature of width {WIDTH_TO_LENGTH:.4f} N, "
          f"peak path, {n_trials} trials per cell; CNR 0 is noise alone")
    print("=" * 100)
    header = (f"{'N':>5} {'CNR':>4} {'med s/true':>10} {'rel sd':>7} {'implied':>8}"
              + "".join(f" {'nu ' + r:>9}" for r in RULES)
              + "".join(f" {'cov ' + r:>9}" for r in RULES))
    print(header)
    total, done = len(lengths) * len(cnr_values), 0
    lines = []
    for N in lengths:
        for true_cnr in cnr_values:
            cell = _run_cell(("white", (N, true_cnr)), "peak", n_trials)
            s = _summarize(cell, 1.0, float(true_cnr))
            lines.append(
                f"{N:>5} {true_cnr:>4} {s['sigma_ratio']:>10.3f} {s['sigma_rsd']:>7.4f} "
                f"{s['implied_dof']:>8.0f}"
                + "".join(f" {s[r]['mean_dof']:>9.1f}" for r in RULES)
                + "".join(f" {s[r]['noise_coverage']:>9.3f}" for r in RULES)
            )
            done += 1
            _progress(done, total)
    print("\n".join(lines))
    print()


def _print_cnr_rows(label, s):
    print(
        f"{label} {s['cnr_ratio']:>8.3f} {s['sigma_ratio']:>8.3f}"
        f" {s['welch']['cnr_coverage']:>9.3f} {s['exact']['cnr_coverage']:>9.3f}"
        f" {s['welch']['cnr_rel_width']:>9.3f} {s['exact']['cnr_rel_width']:>9.3f}"
        f" {s['amp_se_ratio']:>8.2f}"
        f" {s['welch']['noise_se_ratio']:>9.2f} {s['exact']['noise_se_ratio']:>9.2f}"
        f" {s['welch']['cnr_se_ratio']:>9.2f} {s['exact']['cnr_se_ratio']:>9.2f}"
    )


_CNR_COLUMNS = (
    f" {'med CNR':>8} {'med s':>8} {'cov welch':>9} {'cov exact':>9}"
    f" {'wid welch':>9} {'wid exact':>9} {'SE/SD A':>8}"
    f" {'s welch':>9} {'s exact':>9} {'CNR welch':>9} {'CNR exact':>9}"
)


def check_cnr_interval(n_trials, lengths=None):
    lengths = lengths or [256, 512]
    cnr_values = [2, 5, 10, 20, 100]
    print("=" * 124)
    print("Check 2: cnr_ci95 coverage of the true CNR (nominal 0.95), relative width")
    print("(median full width / true CNR), and calibration ratios (mean reported")
    print("standard error / empirical standard deviation) of the amplitude (A), the")
    print(f"noise level (s), and the CNR. White Gaussian noise, {n_trials} trials per cell.")
    print("med CNR and med s are the median estimates over the true values.")
    print("=" * 124)
    print(f"{'N':>5} {'method':<21} {'CNR':>4}" + _CNR_COLUMNS)
    total, done = len(lengths) * len(METHODS) * len(cnr_values), 0
    lines = []
    for N in lengths:
        for method in METHODS:
            for true_cnr in cnr_values:
                cell = _run_cell(("white", (N, true_cnr)), method, n_trials)
                s = _summarize(cell, 1.0, float(true_cnr))
                lines.append((f"{N:>5} {method:<21} {true_cnr:>4}", s))
                done += 1
                _progress(done, total)
    for label, s in lines:
        _print_cnr_rows(label, s)
    print()


def _shot_references(N, width, background, bl_cnr):
    """Amplitude and the two reference noise levels of one shot-noise cell.

    The root mean rate is the plain average. The noise estimate converges to
    the taper-weighted mean variance mean(w**2 * rate) / mean(w**2), which is
    higher when the feature sits where the taper is 1 and the background
    fills the tapered ends.
    """
    grid = np.arange(N, dtype=float)
    shape = np.exp(-0.5 * ((grid - (N - 1) / 2.0) / width) ** 2)
    amplitude = bl_cnr * np.sqrt(background)
    rate = background + amplitude * shape
    w = tukey(N, alpha=TUKEY_ALPHA)
    references = {
        "mean": float(np.sqrt(np.mean(rate))),
        "taper": float(np.sqrt(np.sum(w**2 * rate) / np.sum(w**2))),
    }
    return amplitude, references


def check_shot_noise(n_trials):
    N, width, background, bl_cnr = 512, 20.0, 10.0, 20.0
    amplitude, references = _shot_references(N, width, background, bl_cnr)
    print("=" * 124)
    print("Check 3: Poisson shot noise, Gaussian peak of width 20 on a background of")
    print(f"10 counts, background-limited CNR 20, N = 512, {n_trials} trials. Reference")
    print(f"noise levels: root mean rate {references['mean']:.3f} (CNR "
          f"{amplitude / references['mean']:.2f}) and taper-weighted root mean rate")
    print(f"{references['taper']:.3f} (CNR {amplitude / references['taper']:.2f}); "
          "columns as in check 2, one row per reference. 'model' runs with")
    print("estimate_noise_model=True, so the intervals use the fitted photon-transfer")
    print("variance when the detector finds signal-dependent noise.")
    print("=" * 124)
    print(f"{'N':>5} {'method':<15} {'run':<7} {'ref':<5} {'CNR':>4}" + _CNR_COLUMNS
          + f" {'sig.dep':>7}")
    for method in ("matched_filter", "peak"):
        for run, noise_model in (("default", False), ("model", True)):
            cell = _run_cell(("shot", (N, background, bl_cnr, width)), method,
                             n_trials, noise_model)
            applied = float(np.mean(cell["signal_dependent"]))
            for name, true_sigma in references.items():
                true_cnr = amplitude / true_sigma
                s = _summarize(cell, true_sigma, true_cnr)
                label = f"{N:>5} {method:<15} {run:<7} {name:<5} {true_cnr:>4.1f}"
                print(f"{label} {s['cnr_ratio']:>8.3f} {s['sigma_ratio']:>8.3f}"
                      f" {s['welch']['cnr_coverage']:>9.3f} {s['exact']['cnr_coverage']:>9.3f}"
                      f" {s['welch']['cnr_rel_width']:>9.3f} {s['exact']['cnr_rel_width']:>9.3f}"
                      f" {s['amp_se_ratio']:>8.2f}"
                      f" {s['welch']['noise_se_ratio']:>9.2f} {s['exact']['noise_se_ratio']:>9.2f}"
                      f" {s['welch']['cnr_se_ratio']:>9.2f} {s['exact']['cnr_se_ratio']:>9.2f}"
                      f" {applied:>7.3f}")
    print()


def _subset_coverage(cell, mask, true_cnr):
    """CNR-interval coverage over the trials selected by mask (NaN if none)."""
    if not np.any(mask):
        return float("nan")
    lo, hi = cell["cnr_lo"][mask], cell["cnr_hi"][mask]
    return float(np.mean((lo <= true_cnr) & (true_cnr <= hi)))


def check_shot_grid(n_trials):
    """Where the noise-model intervals apply on a grid of count levels."""
    N, width = 512, 20.0
    backgrounds = [1.0, 10.0, 100.0, 1000.0]
    bl_cnr_values = [2.0, 5.0, 20.0, 50.0]
    print("=" * 124)
    print("Check 4: Poisson shot noise over background level and background-limited CNR,")
    print(f"Gaussian peak of width 20, N = 512, estimate_noise_model=True, {n_trials} trials")
    print("per cell. taper/mean: ratio of the taper-weighted to the plain root mean rate.")
    print("applied: fraction of trials whose intervals use the photon-transfer variance;")
    print("skipped: fraction where the detector declined to fit (signal range under 3")
    print("noise RMS); the rest tested not significant. Coverage of the CNR interval")
    print("against each reference, overall and split by whether the variance was applied.")
    print("=" * 124)
    print(f"{'method':<15} {'bg':>6} {'BL-CNR':>6} {'taper/mean':>10} {'applied':>8} "
          f"{'skipped':>8} | {'cov mean':>8} {'cov taper':>9} | "
          f"{'taper appl':>10} {'taper fall':>10} | {'SE/SD A':>7} {'s exact':>7}")
    for method in ("matched_filter", "peak"):
        for background in backgrounds:
            for bl_cnr in bl_cnr_values:
                amplitude, refs = _shot_references(N, width, background, bl_cnr)
                cell = _run_cell(("shot", (N, background, bl_cnr, width)), method,
                                 n_trials, noise_model=True)
                applied = cell["signal_dependent"].astype(bool)
                skipped = cell["model_skipped"].astype(bool)
                taper_cnr = amplitude / refs["taper"]
                s = _summarize(cell, refs["taper"], taper_cnr)
                print(f"{method:<15} {background:>6.0f} {bl_cnr:>6.0f} "
                      f"{refs['taper'] / refs['mean']:>10.3f} {np.mean(applied):>8.3f} "
                      f"{np.mean(skipped):>8.3f} | "
                      f"{_subset_coverage(cell, np.ones_like(applied), amplitude / refs['mean']):>8.3f} "
                      f"{_subset_coverage(cell, np.ones_like(applied), taper_cnr):>9.3f} | "
                      f"{_subset_coverage(cell, applied, taper_cnr):>10.3f} "
                      f"{_subset_coverage(cell, ~applied, taper_cnr):>10.3f} | "
                      f"{s['amp_se_ratio']:>7.2f} {s['exact']['noise_se_ratio']:>7.2f}",
                      flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000,
                        help="Monte Carlo trials per cell")
    parser.add_argument("--grid-trials", type=int, default=200,
                        help="trials per cell for check 4 (runs the noise-model detector)")
    parser.add_argument("--checks", type=int, nargs="+", default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4], help="which checks to run")
    parser.add_argument("--lengths", type=int, nargs="+", default=None,
                        help="profile lengths for checks 1 and 2 (overrides the defaults)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="worker processes (1 runs serially)")
    args = parser.parse_args()
    global _POOL
    if args.workers > 1:
        _POOL = ProcessPoolExecutor(max_workers=args.workers)
    if 1 in args.checks:
        check_noise_interval(args.trials, args.lengths)
    if 2 in args.checks:
        check_cnr_interval(args.trials, args.lengths)
    if 3 in args.checks:
        check_shot_noise(args.trials)
    if 4 in args.checks:
        check_shot_grid(args.grid_trials)


if __name__ == "__main__":
    main()
