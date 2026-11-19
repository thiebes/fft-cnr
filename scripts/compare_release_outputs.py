"""Check that fft_cnr outputs are bit-identical to a reference release.

Loads ``src/fft_cnr/core.py`` at a git ref (default ``v0.4.1``) beside the
working-tree package and runs both on a fixed set of profiles, with and
without ``estimate_noise_model``, and on a seeded batch of profiles (centred
Gaussians in records of +/-1.5 to +/-4 standard deviations, and in long
records) on the peak, matched-filter, and generalized-Gaussian paths. Every
``CNREstimate`` field and property and every ``NoiseModel`` field that the
reference defines must match bit for bit. Fields that only the working tree
defines are listed, not compared, and ``peak_snr`` is reported per profile for
information. Exits 1 on any difference.

With ``--allow-fit-changes``, a generalized-Gaussian run may differ when the
working tree's fit reaches a strictly lower sum of squared residuals than the
reference's fit on the same profile (the second start of 0.4.2). Every other
difference still fails, and the counts of identical and changed runs are
reported.

Both versions run in one interpreter, so they share numpy and scipy and only
the package code differs.

Usage:
    uv run python scripts/compare_release_outputs.py
    uv run python scripts/compare_release_outputs.py --ref v0.4.1 --allow-fit-changes
"""

import argparse
import dataclasses
import importlib.util
import inspect
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

import fft_cnr.core as candidate
from _signals import _generalized_gaussian, _lorentzian

REPO = Path(__file__).resolve().parents[1]


def _load_reference_core(ref):
    """Import core.py as it was at ``ref``, under its own module name."""
    source = subprocess.run(
        ["git", "show", f"{ref}:src/fft_cnr/core.py"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    name = "fft_cnr_reference_core"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "core.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def _poisson_profile(N, background, peak, sigma, seed):
    """Poisson counts on ``background + peak * gaussian``; a negative peak is a dip."""
    x = np.arange(N, dtype=float)
    rate = background + peak * np.exp(-0.5 * ((x - (N - 1) / 2.0) / sigma) ** 2)
    return np.random.default_rng(seed).poisson(rate).astype(float)


def _cases():
    """(label, profile, fft_cnr keyword arguments) covering every path and option."""
    gaussian, gaussian_clean = _generalized_gaussian(512, 20.0, 25.0, 1.0, 2.0, 1)
    flat_top, _ = _generalized_gaussian(512, 20.0, 30.0, 1.0, 4.0, 2)
    cusp, _ = _generalized_gaussian(512, 50.0, 20.0, 1.0, 1.3, 3)
    lorentzian, _ = _lorentzian(512, 20.0, 23.5, 1.0, 4)
    dip, _ = _generalized_gaussian(512, -15.0, 25.0, 1.0, 2.0, 5)
    short, _ = _generalized_gaussian(64, 10.0, 5.0, 1.0, 2.0, 6)
    broad, _ = _generalized_gaussian(128, 20.0, 20.0, 1.0, 2.0, 7)
    weak, _ = _generalized_gaussian(512, 1.5, 25.0, 1.0, 2.0, 3)
    shot = _poisson_profile(512, 25.0, 100.0, 25.0, 8)
    shot_dip = _poisson_profile(512, 300.0, -100.0, 25.0, 10)
    unit_gaussian = np.exp(-0.5 * ((np.arange(512) - 255.5) / 25.0) ** 2)
    filling = _record_profiles(32, 2.5, 8.0, 2026, 4)[3]
    peak_path_cases = [
        ("Gaussian", gaussian, {}),
        ("Gaussian, hann window", gaussian, {"window": "hann"}),
        ("Gaussian, no window", gaussian, {"window": "none"}),
        ("Gaussian, matched filter", gaussian, {"template": gaussian_clean}),
        ("Gaussian, explicit roi", gaussian, {"roi": (128, 384)}),
        ("Gaussian, roi auto", gaussian, {"roi": "auto"}),
        ("Gaussian, bandpassed noise", gaussian, {"return_bandpassed_noise": True}),
        ("flat-top p 4", flat_top, {}),
        ("cusp p 1.3", cusp, {}),
        ("Lorentzian", lorentzian, {}),
        ("dip", dip, {}),
        ("N 64", short, {}),
        ("N 128, sigma 20", broad, {}),
        ("weak feature (noise model skipped)", weak, {}),
        ("Poisson on background", shot, {}),
        ("Poisson on background, matched filter", shot, {"template": unit_gaussian}),
        ("Poisson dip on background", shot_dip, {}),
        ("feature filling the record (+/-2.5 SD)", filling, {}),
    ]
    fit_cases = [
        (f"{label}, generalized-Gaussian fit", profile, {"fit_model": "generalized_gaussian"})
        for label, profile, kwargs in peak_path_cases
        if not kwargs
    ]
    fit_cases.append(
        ("dip filling the record, generalized-Gaussian fit", -filling,
         {"fit_model": "generalized_gaussian"})
    )
    return peak_path_cases + fit_cases


def _record_profiles(N, half_width_sds, cnr, seed, n_profiles):
    """Centred Gaussians (peak 1, zero baseline) in a record of +/-``half_width_sds``
    standard deviations, in white noise of SD 1 / ``cnr``."""
    x = np.arange(N, dtype=float)
    sigma = N / (2.0 * half_width_sds)
    feature = np.exp(-0.5 * ((x - (N - 1) / 2.0) / sigma) ** 2)
    noise = np.random.default_rng(seed).normal(0, 1.0 / cnr, (n_profiles, N))
    return feature + noise


def _batch_cases(n_profiles):
    """Seeded (label, profile, kwargs) runs on the peak, matched-filter, and
    generalized-Gaussian paths: records of +/-1.5 to +/-4 SD at N = 32, and a
    Gaussian of SD 25 in a record of 512 samples."""
    records = [(32, k, cnr) for k in (1.5, 2.0, 2.5, 3.0, 4.0) for cnr in (5.0, 13.0, 34.0)]
    records += [(512, 512 / 50.0, cnr) for cnr in (2.0, 5.0, 20.0)]
    for i, (N, k, cnr) in enumerate(records):
        x = np.arange(N, dtype=float)
        template = np.exp(-0.5 * ((x - (N - 1) / 2.0) / (N / (2.0 * k))) ** 2)
        for j, profile in enumerate(_record_profiles(N, k, cnr, [41, i], n_profiles)):
            label = f"batch N {N} +/-{k:g} SD CNR {cnr:g} #{j}"
            yield label, profile, {}
            yield label, profile, {"template": template}
            yield label, profile, {"fit_model": "generalized_gaussian"}


def _bits_equal(a, b):
    """Exact equality, with floats and arrays compared by their bytes (NaN equals NaN)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and a.tobytes() == b.tobytes()
        )
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_bits_equal(a[key], b[key]) for key in a)
        )
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_bits_equal, a, b))
    if isinstance(a, (float, np.floating)):
        return type(a) is type(b) and np.float64(a).tobytes() == np.float64(b).tobytes()
    return type(a) is type(b) and a == b


def _attribute_names(cls):
    """Dataclass fields, then public properties."""
    fields = [field.name for field in dataclasses.fields(cls)]
    properties = [name for name, member in inspect.getmembers(cls) if isinstance(member, property)]
    return fields + properties


def _differing_attributes(reference, result, prefix=""):
    """Names of the reference's attributes that ``result`` lacks or changes."""
    differing = []
    for name in _attribute_names(type(reference)):
        ref_value = getattr(reference, name)
        if not hasattr(result, name):
            differing.append(prefix + name)
        elif dataclasses.is_dataclass(ref_value) and dataclasses.is_dataclass(getattr(result, name)):
            differing.extend(_differing_attributes(ref_value, getattr(result, name), f"{name}."))
        elif not _bits_equal(ref_value, getattr(result, name)):
            differing.append(prefix + name)
    return differing


def _added_noise_model_fields(reference_core):
    reference_names = {field.name for field in dataclasses.fields(reference_core.NoiseModel)}
    return [
        field.name
        for field in dataclasses.fields(candidate.NoiseModel)
        if field.name not in reference_names
    ]


def _fit_ssr(core_module, profile):
    """Sum of squared residuals of a release's generalized-Gaussian fit to the
    profile, as fft_cnr fits it with default options (inf if the fit fails).
    Releases up to 0.4.1 take the profile only; 0.4.2 also takes the peak read."""
    d = core_module._spectral_decomposition(
        profile, window="tukey", tukey_alpha=0.25, welch_nperseg=None,
        welch_noverlap=None, cutoff_guard=(0.05, 0.5), fallback_cut_frac=0.25,
    )
    x_raw = d.x + d.x_mean
    fit = core_module._fit_generalized_gaussian_amplitude
    if len(inspect.signature(fit).parameters) == 1:
        _, _, params = fit(x_raw)
    else:
        baseline, excursion, idx = core_module._peak_read(x_raw, d.x_lp + d.x_mean)
        _, _, params = fit(x_raw, (float(excursion[idx]), idx, baseline))
    if not params:
        return np.inf
    grid = np.arange(x_raw.size, dtype=float)
    z = np.abs((grid - params["center"]) / params["sigma"])
    model = params["amplitude"] * np.exp(-0.5 * z ** params["shape"]) + params["baseline"]
    return float(np.sum((x_raw - model) ** 2))


def _fit_change_is_lower_residual(reference_core, profile, kwargs):
    """True when a differing generalized-Gaussian run comes from a working-tree
    fit with a strictly lower sum of squared residuals than the reference's."""
    if set(kwargs) != {"fit_model"}:
        return False
    return _fit_ssr(candidate, profile) < _fit_ssr(reference_core, profile)


def _peak_snr(result):
    if result.noise_model is None:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return result.noise_model.peak_snr(result.amplitude)


def _path_name(kwargs):
    if "template" in kwargs:
        return "matched filter"
    if kwargs.get("fit_model") == "generalized_gaussian":
        return "generalized Gaussian"
    return "peak and options"


def _compare_run(reference_core, profile, kwargs, reference, result, allow_fit_changes):
    """'identical', 'lower residual' (an accepted fit change), or 'fail'."""
    if not _differing_attributes(reference, result):
        return "identical"
    if allow_fit_changes and _fit_change_is_lower_residual(reference_core, profile, kwargs):
        return "lower residual"
    return "fail"


def _record_outcome(path_to_counts, kwargs, outcome):
    counts = path_to_counts.setdefault(
        _path_name(kwargs), {"identical": 0, "lower residual": 0, "fail": 0}
    )
    counts[outcome] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="v0.4.1", help="git ref of the reference release")
    parser.add_argument("--allow-fit-changes", action="store_true",
                        help="accept generalized-Gaussian differences from a lower-residual fit")
    parser.add_argument("--batch", type=int, default=40,
                        help="profiles per seeded batch cell (0 skips the batch)")
    args = parser.parse_args()

    reference_core = _load_reference_core(args.ref)
    print(f"Reference: {args.ref}; candidate: working tree ({candidate.__file__})")
    print(f"NoiseModel fields added by the candidate: {_added_noise_model_fields(reference_core)}")
    print()
    print(f"{'profile':<40} {'noise model':>11} {'outputs':>10} "
          f"{'peak_snr ref':>13} {'peak_snr new':>13}")

    failures = 0
    path_to_counts = {}
    for label, profile, kwargs in _cases():
        for estimate_noise_model in (False, True):
            reference = reference_core.fft_cnr(
                profile, estimate_noise_model=estimate_noise_model, **kwargs
            )
            result = candidate.fft_cnr(
                profile, estimate_noise_model=estimate_noise_model, **kwargs
            )
            outcome = _compare_run(reference_core, profile, kwargs, reference, result,
                                   args.allow_fit_changes)
            failures += outcome == "fail"
            _record_outcome(path_to_counts, kwargs, outcome)
            differing = _differing_attributes(reference, result)
            status = {"identical": "identical", "lower residual": "CHANGED (lower residual)",
                      "fail": "DIFFER: " + ", ".join(differing)}[outcome]
            snr_columns = (
                f"{_peak_snr(reference):>13.4f} {_peak_snr(result):>13.4f}"
                if estimate_noise_model
                else ""
            )
            print(f"{label:<40} {str(estimate_noise_model):>11} {status:>10} {snr_columns}")

    batch_failures = []
    for label, profile, kwargs in _batch_cases(args.batch) if args.batch else ():
        reference = reference_core.fft_cnr(profile, **kwargs)
        result = candidate.fft_cnr(profile, **kwargs)
        outcome = _compare_run(reference_core, profile, kwargs, reference, result,
                               args.allow_fit_changes)
        _record_outcome(path_to_counts, kwargs, outcome)
        if outcome == "fail":
            batch_failures.append(f"{label} ({_path_name(kwargs)})")
    failures += len(batch_failures)
    for line in batch_failures:
        print(f"DIFFER: {line}")

    print()
    print(f"{'path':<24} {'identical':>10} {'changed, lower residual':>24} {'failed':>8}")
    for path, counts in path_to_counts.items():
        print(f"{path:<24} {counts['identical']:>10} {counts['lower residual']:>24} "
              f"{counts['fail']:>8}")
    print()
    if failures:
        print(f"{failures} run(s) differ from {args.ref}.")
        sys.exit(1)
    if any(counts["lower residual"] for counts in path_to_counts.values()):
        print(f"All runs bit-identical to {args.ref} apart from the added fields, peak_snr, "
              f"and generalized-Gaussian fits with a lower residual.")
    else:
        print(f"All runs bit-identical to {args.ref} apart from the added fields and peak_snr.")


if __name__ == "__main__":
    main()
