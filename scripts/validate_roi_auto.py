"""How roi="auto" interacts with the peak method's edge baseline.

``roi="auto"`` sizes a window of about 2.2 FWHM (full width at half maximum)
around the largest feature. The peak method subtracts a baseline taken from the
outer quarter of the window on each side, so in a window that narrow the
baseline sits on the feature's shoulders and the amplitude reads low. The
matched filter does not use that baseline.

A centered Gaussian of peak 1 and SD sigma sits in a record of N = 512 samples,
in white Gaussian noise of SD 1 / CNR, where CNR is the nominal CNR (the
simulation input). Each (sigma, CNR) cell draws its profiles as one array from
``np.random.default_rng([3433, si, ci])`` (indices of sigma and CNR).

For each cell the script prints the median width of the auto window in FWHM,
and the median amplitude, in units of the nominal peak, of:

- the peak method on the full record, with ``roi="auto"``, and with explicit
  bounds 5.4 FWHM wide (+/-2.7 FWHM, clipped to the record);
- the matched filter (the exact template) with ``roi="auto"``.

Usage:
    uv run python scripts/validate_roi_auto.py
    uv run python scripts/validate_roi_auto.py --trials 50   # quick
"""

import argparse

import numpy as np

from fft_cnr import fft_cnr

N = 512
CENTER = 255
SIGMAS = (3, 5, 10, 20, 40)
CNR_VALUES = (5, 20)
EXPLICIT_HALF_WIDTH_FWHM = 2.7
FWHM_PER_SD = 2 * np.sqrt(2 * np.log(2))


def _explicit_bounds(sigma):
    half = int(round(EXPLICIT_HALF_WIDTH_FWHM * FWHM_PER_SD * sigma))
    return (max(0, CENTER - half), min(N, CENTER + half + 1))


def _run_cell(si, ci, trials):
    """Median auto-window width (FWHM) and median amplitudes of one (sigma, CNR) cell."""
    sigma = SIGMAS[si]
    x = np.arange(N, dtype=float)
    template = np.exp(-0.5 * ((x - CENTER) / sigma) ** 2)
    noise_sd = 1.0 / CNR_VALUES[ci]
    profiles = template + np.random.default_rng([3433, si, ci]).normal(0, noise_sd, (trials, N))
    explicit = _explicit_bounds(sigma)
    widths, full, auto, wide, matched = [], [], [], [], []
    for profile in profiles:
        auto_result = fft_cnr(profile, roi="auto")
        start, stop = auto_result.diagnostics["roi"]
        widths.append((stop - start) / (FWHM_PER_SD * sigma))
        auto.append(auto_result.amplitude)
        full.append(fft_cnr(profile).amplitude)
        wide.append(fft_cnr(profile, roi=explicit).amplitude)
        matched.append(fft_cnr(profile, template=template, roi="auto").amplitude)
    return [np.median(values) for values in (widths, full, auto, wide, matched)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=200, help="profiles per cell")
    args = parser.parse_args()

    print(f"Centered Gaussian, N = {N}, {args.trials} profiles per cell. Median amplitude / "
          "nominal peak.")
    print(f"{'SD':>4} {'CNR':>4} {'auto width (FWHM)':>18} {'peak, full':>11} {'peak, auto':>11} "
          f"{'peak, 5.4 FWHM':>15} {'matched, auto':>14}")
    for si, sigma in enumerate(SIGMAS):
        for ci, cnr in enumerate(CNR_VALUES):
            width, full, auto, wide, matched = _run_cell(si, ci, args.trials)
            print(f"{sigma:>4} {cnr:>4} {width:>18.2f} {full:>11.3f} {auto:>11.3f} "
                  f"{wide:>15.3f} {matched:>14.3f}")


if __name__ == "__main__":
    main()
