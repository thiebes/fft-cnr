"""Check that fft_cnr outputs are bit-identical to a reference release.

Loads ``src/fft_cnr/core.py`` at a git ref (default ``v0.4.0``) beside the
working-tree package and runs both on a fixed set of profiles, with and
without ``estimate_noise_model``. Every ``CNREstimate`` field and property and
every ``NoiseModel`` field that the reference defines must match bit for bit.
Fields that only the working tree defines are listed, not compared, and
``peak_snr`` is reported per profile for information. Exits 1 on any
difference.

Both versions run in one interpreter, so they share numpy and scipy and only
the package code differs.

Usage:
    uv run python scripts/compare_release_outputs.py
    uv run python scripts/compare_release_outputs.py --ref v0.4.0
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
    return [
        ("Gaussian", gaussian, {}),
        ("Gaussian, hann window", gaussian, {"window": "hann"}),
        ("Gaussian, no window", gaussian, {"window": "none"}),
        ("Gaussian, matched filter", gaussian, {"template": gaussian_clean}),
        ("Gaussian, generalized-Gaussian fit", gaussian, {"fit_model": "generalized_gaussian"}),
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
    ]


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


def _peak_snr(result):
    if result.noise_model is None:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return result.noise_model.peak_snr(result.amplitude)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="v0.4.0", help="git ref of the reference release")
    args = parser.parse_args()

    reference_core = _load_reference_core(args.ref)
    print(f"Reference: {args.ref}; candidate: working tree ({candidate.__file__})")
    print(f"NoiseModel fields added by the candidate: {_added_noise_model_fields(reference_core)}")
    print()
    print(f"{'profile':<40} {'noise model':>11} {'outputs':>10} "
          f"{'peak_snr ref':>13} {'peak_snr new':>13}")

    failures = 0
    for label, profile, kwargs in _cases():
        for estimate_noise_model in (False, True):
            reference = reference_core.fft_cnr(
                profile, estimate_noise_model=estimate_noise_model, **kwargs
            )
            result = candidate.fft_cnr(
                profile, estimate_noise_model=estimate_noise_model, **kwargs
            )
            differing = _differing_attributes(reference, result)
            failures += bool(differing)
            status = "identical" if not differing else "DIFFER: " + ", ".join(differing)
            snr_columns = (
                f"{_peak_snr(reference):>13.4f} {_peak_snr(result):>13.4f}"
                if estimate_noise_model
                else ""
            )
            print(f"{label:<40} {str(estimate_noise_model):>11} {status:>10} {snr_columns}")

    print()
    if failures:
        print(f"{failures} run(s) differ from {args.ref}.")
        sys.exit(1)
    print(f"All runs bit-identical to {args.ref} apart from the added fields and peak_snr.")


if __name__ == "__main__":
    main()
