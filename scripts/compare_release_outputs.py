"""Check that fft_cnr outputs are bit-identical to a reference release.

Loads ``src/fft_cnr/core.py`` at a git ref (default ``v0.4.1``) beside the
working-tree package and runs both on a fixed set of profiles, with and
without ``estimate_noise_model``, and on a seeded batch of profiles (centered
Gaussians in records of +/-1.5 to +/-4 standard deviations, and in long
records) on the peak, matched-filter, and generalized-Gaussian paths, without
the noise model. Every ``CNREstimate`` field and property and every
``NoiseModel`` field that the reference defines must match bit for bit. Fields
that only the working tree defines are listed, not compared, and ``peak_snr``
is reported per profile for information. Exits 1 on any difference.

With ``--allow-fit-changes``, a generalized-Gaussian run may differ when the
working tree's fit has a strictly lower sum of squared residuals than the
reference's fit on the same profile (the second start of 0.4.2), and only in
the outputs that follow the fit: ``cnr``, ``cnr_ci95``, ``amplitude``,
``amplitude_se``, ``amplitude_snr``, and the diagnostics
``gaussian_fit_params``, ``gaussian_fit_rejection_reason``,
``amplitude_method``, and ``amplitude_sign_ambiguous``. Every other difference
still fails. An accepted change is counted by its relative residual gain:
below ``_SAME_MINIMUM_RTOL`` the two fits are the same minimum reached from two
starts, above it the working tree's fit is a lower minimum, and a reference fit
that failed to converge is counted on its own.

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
from _signals import _gaussian_batch, _generalized_gaussian, _lorentzian

REPO = Path(__file__).resolve().parents[1]

# Outputs that follow the generalized-Gaussian fit, the only ones an accepted
# fit change may alter. Everything else (the noise level and its interval, the
# cutoff, the noise model, and the remaining diagnostics) must stay bit-identical.
FIT_FIELDS = {"cnr", "cnr_ci95", "amplitude", "amplitude_se", "amplitude_snr", "diagnostics"}
FIT_DIAGNOSTICS = {"gaussian_fit_params", "gaussian_fit_rejection_reason",
                   "amplitude_method", "amplitude_sign_ambiguous"}
# Relative residual gain below which an accepted change is the same minimum
# reached from two starts. curve_fit stops at a relative tolerance of 1e-8, and
# in the 0.4.2 batch every gain lies below 1e-8 or above 1e-6.
_SAME_MINIMUM_RTOL = 1e-6
OUTCOMES = ("identical", "same minimum", "lower minimum", "reference fit failed", "fail")


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
    filling = _gaussian_batch(32, 1.0, 32 / 5.0, 1.0 / 8.0, 2026, 4)[0][3]
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


def _batch_cases(n_profiles):
    """Seeded (label, profile, kwargs) runs on the peak, matched-filter, and
    generalized-Gaussian paths: records of +/-1.5 to +/-4 SD at N = 32, and a
    Gaussian of SD 25 in a record of 512 samples."""
    records = [(32, k, cnr) for k in (1.5, 2.0, 2.5, 3.0, 4.0) for cnr in (5.0, 13.0, 34.0)]
    records += [(512, 512 / 50.0, cnr) for cnr in (2.0, 5.0, 20.0)]
    for i, (N, k, cnr) in enumerate(records):
        profiles, template = _gaussian_batch(N, 1.0, N / (2.0 * k), 1.0 / cnr, [41, i],
                                             n_profiles)
        for j, profile in enumerate(profiles):
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
    profile, as its fft_cnr fits it with default options (inf if the fit fails).

    The decomposition options are read from that release's fft_cnr signature.
    Releases up to 0.4.1 fit the profile alone; from 0.4.2 the fit also takes
    the peak read. The model is written out because the fit's model function is
    local to it."""
    defaults = inspect.signature(core_module.fft_cnr).parameters
    options = ("window", "tukey_alpha", "welch_nperseg", "welch_noverlap",
               "cutoff_guard", "fallback_cut_frac")
    d = core_module._spectral_decomposition(
        profile, **{name: defaults[name].default for name in options}
    )
    x_raw = d.x + d.x_mean
    fit = core_module._fit_generalized_gaussian_amplitude
    if hasattr(core_module, "_peak_read"):
        baseline, excursion, idx = core_module._peak_read(x_raw, d.x_lp + d.x_mean)
        _, _, params = fit(x_raw, (float(excursion[idx]), idx, baseline))
    else:
        _, _, params = fit(x_raw)
    if not params:
        return np.inf
    grid = np.arange(x_raw.size, dtype=float)
    z = np.abs((grid - params["center"]) / params["sigma"])
    model = params["amplitude"] * np.exp(-0.5 * z ** params["shape"]) + params["baseline"]
    return float(np.sum((x_raw - model) ** 2))


def _without_fit_diagnostics(diagnostics):
    return {key: value for key, value in diagnostics.items() if key not in FIT_DIAGNOSTICS}


def _differs_only_in_fit_outputs(reference, result, differing):
    """True when every differing output is one that follows the fit."""
    if not set(differing) <= FIT_FIELDS:
        return False
    return "diagnostics" not in differing or _bits_equal(
        _without_fit_diagnostics(reference.diagnostics),
        _without_fit_diagnostics(result.diagnostics),
    )


def _classify_fit_change(reference_core, profile):
    """The accepted-change outcome for a differing generalized-Gaussian run, or
    'fail' when the working tree's fit does not have a lower residual."""
    reference_ssr = _fit_ssr(reference_core, profile)
    candidate_ssr = _fit_ssr(candidate, profile)
    if not candidate_ssr < reference_ssr:
        return "fail"
    if np.isinf(reference_ssr):
        return "reference fit failed"
    gain = (reference_ssr - candidate_ssr) / reference_ssr
    return "same minimum" if gain < _SAME_MINIMUM_RTOL else "lower minimum"


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
    """One of OUTCOMES, and the names of the differing outputs."""
    differing = _differing_attributes(reference, result)
    if not differing:
        return "identical", differing
    fit_run = kwargs == {"fit_model": "generalized_gaussian"}
    if allow_fit_changes and fit_run and _differs_only_in_fit_outputs(reference, result, differing):
        return _classify_fit_change(reference_core, profile), differing
    return "fail", differing


def _record_outcome(path_to_counts, kwargs, outcome):
    counts = path_to_counts.setdefault(_path_name(kwargs), dict.fromkeys(OUTCOMES, 0))
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
    cases = _cases()
    width = max(len(label) for label, _, _ in cases)
    print(f"{'profile':<{width}} {'noise model':>11} {'peak_snr ref':>13} {'peak_snr new':>13}"
          f"  outputs")

    failures = 0
    path_to_counts = {}
    for label, profile, kwargs in cases:
        for estimate_noise_model in (False, True):
            reference = reference_core.fft_cnr(
                profile, estimate_noise_model=estimate_noise_model, **kwargs
            )
            result = candidate.fft_cnr(
                profile, estimate_noise_model=estimate_noise_model, **kwargs
            )
            outcome, differing = _compare_run(reference_core, profile, kwargs, reference,
                                              result, args.allow_fit_changes)
            failures += outcome == "fail"
            _record_outcome(path_to_counts, kwargs, outcome)
            status = {"identical": "identical", "fail": "DIFFER: " + ", ".join(differing)}.get(
                outcome, f"changed: {outcome}")
            snr_columns = (
                f"{_peak_snr(reference):>13.4f} {_peak_snr(result):>13.4f}"
                if estimate_noise_model
                else f"{'':>13} {'':>13}"
            )
            print(f"{label:<{width}} {str(estimate_noise_model):>11} {snr_columns}  {status}")

    batch_failures = []
    for label, profile, kwargs in _batch_cases(args.batch) if args.batch else ():
        reference = reference_core.fft_cnr(profile, **kwargs)
        result = candidate.fft_cnr(profile, **kwargs)
        outcome, differing = _compare_run(reference_core, profile, kwargs, reference, result,
                                          args.allow_fit_changes)
        _record_outcome(path_to_counts, kwargs, outcome)
        if outcome == "fail":
            batch_failures.append(f"{label} ({_path_name(kwargs)}): {', '.join(differing)}")
    failures += len(batch_failures)
    for line in batch_failures:
        print(f"DIFFER: {line}")

    print()
    print(f"{'path':<24}" + "".join(f" {outcome:>21}" for outcome in OUTCOMES))
    for path, counts in path_to_counts.items():
        print(f"{path:<24}" + "".join(f" {counts[outcome]:>21}" for outcome in OUTCOMES))
    print()
    if failures:
        print(f"{failures} run(s) differ from {args.ref}.")
        sys.exit(1)
    changed = sum(counts[o] for counts in path_to_counts.values() for o in OUTCOMES[1:4])
    if changed:
        print(f"All runs bit-identical to {args.ref} apart from the added fields, peak_snr, "
              f"and {changed} generalized-Gaussian runs whose fit has a lower residual and "
              f"which differ only in the outputs that follow the fit.")
    else:
        print(f"All runs bit-identical to {args.ref} apart from the added fields and peak_snr.")


if __name__ == "__main__":
    main()
