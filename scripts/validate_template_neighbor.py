"""How a neighboring feature shifts the matched-filter amplitude, and what roi does.

The matched filter projects the demeaned, tapered profile onto the demeaned,
tapered template. Away from the target the demeaned template is negative, so a
second feature in the profile moves the amplitude in proportion to the area
under it: a neighbor of the same sign lowers it. The taper weights the
projection, so a neighbor that reaches the Tukey ramp at the end of the record
counts less.

The amplitude is linear in the profile, and with explicit ``roi`` bounds every
step of the matched-filter path is fixed by the record length. So the shift a
neighbor causes is exactly the amplitude that ``fft_cnr`` returns for the
neighbor alone, with the target's template. It does not depend on the noise or
on the target's height, and a neighbor h times as high shifts the amplitude h
times as much. Shifts are in units of the target's peak, which is 1.

Gaussian target and neighbor; the template is the target alone, at full length.
The script prints three tables:

1. Separation and width: N = 512, target of SD sigma centered on sample 200,
   neighbor of the same SD and height at separation d to its right, on the full
   record and with ``roi`` at +/-3 SD around the target. At sigma 20 and
   d = 12 SD the neighbor reaches the Tukey ramp.
2. Record length and neighbor width: target of SD 10 centered on sample
   round(0.4 N), neighbor of equal height and SD 10 or 20, with 4 SD of each
   feature between them (8 SD for equal widths), on the full record.
3. Cost of the window, for the table 1 target alone in white noise of SD 0.1
   (nominal CNR 10): the scatter (SD) of the amplitude with ``roi`` at +/-3 SD
   over its scatter on the full record, and the same ratio of the median
   reported ``amplitude_se``. Each sigma draws its profiles as one array from
   ``np.random.default_rng([4758, si])`` (index of sigma).

Usage:
    uv run python scripts/validate_template_neighbor.py
"""

import numpy as np

from fft_cnr import fft_cnr

TARGET_CENTER = 200
N = 512
SIGMAS = (5, 10, 20)
SEPARATIONS_IN_SD = (4, 5, 6, 8, 12)
ROI_HALF_WIDTH_IN_SD = 3
LENGTH_TABLE_TARGET_SD = 10
LENGTH_TABLE_CASES = ((256, 10), (512, 10), (1024, 10), (2048, 10), (512, 20))
NOISE_SD = 0.1
SE_TRIALS = 200


def _gaussian(n, center, sigma):
    x = np.arange(n, dtype=float)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _roi_around(center, sigma):
    half = ROI_HALF_WIDTH_IN_SD * sigma
    return (center - half, center + half + 1)


def _percent(shift):
    """Shift as a percentage, with -0.00 printed as 0.00."""
    return round(100 * shift, 2) + 0.0


def _neighbor_shift(neighbor, target, roi=None):
    """Amplitude the neighbor adds, in units of the target's peak (linear, so exact)."""
    return fft_cnr(neighbor, template=target, roi=roi).amplitude


def _print_separation_table():
    print(f"1. N = {N}, target on sample {TARGET_CENTER}, neighbor of the same SD and height. "
          "Shift, % of the target's peak.")
    print(f"{'SD':>4} {'d / SD':>6} {'full record':>12} "
          f"{f'roi +/-{ROI_HALF_WIDTH_IN_SD} SD':>13}")
    for sigma in SIGMAS:
        target = _gaussian(N, TARGET_CENTER, sigma)
        roi = _roi_around(TARGET_CENTER, sigma)
        for separation in SEPARATIONS_IN_SD:
            neighbor = _gaussian(N, TARGET_CENTER + separation * sigma, sigma)
            full = _percent(_neighbor_shift(neighbor, target))
            windowed = _percent(_neighbor_shift(neighbor, target, roi))
            print(f"{sigma:>4} {separation:>6} {full:>11.2f}% {windowed:>12.2f}%")


def _print_length_table():
    target_sd = LENGTH_TABLE_TARGET_SD
    print(f"\n2. Target of SD {target_sd}, neighbor of equal height 4 SD of each feature "
          "away. Shift on the full record, % of the target's peak.")
    print(f"{'N':>5} {'neighbor SD':>11} {'shift':>8}")
    for n, neighbor_sd in LENGTH_TABLE_CASES:
        center = round(0.4 * n)
        target = _gaussian(n, center, target_sd)
        neighbor = _gaussian(n, center + 4 * target_sd + 4 * neighbor_sd, neighbor_sd)
        print(f"{n:>5} {neighbor_sd:>11} {_percent(_neighbor_shift(neighbor, target)):>7.2f}%")


def _print_window_cost_table():
    print(f"\n3. Target alone, N = {N}, noise SD {NOISE_SD}, {SE_TRIALS} profiles per SD. "
          f"Ratio with roi +/-{ROI_HALF_WIDTH_IN_SD} SD to the full record.")
    print(f"{'SD':>4} {'amplitude scatter':>17} {'median amplitude_se':>19}")
    for si, sigma in enumerate(SIGMAS):
        target = _gaussian(N, TARGET_CENTER, sigma)
        roi = _roi_around(TARGET_CENTER, sigma)
        profiles = target + np.random.default_rng([4758, si]).normal(0, NOISE_SD, (SE_TRIALS, N))
        full = [fft_cnr(profile, template=target) for profile in profiles]
        windowed = [fft_cnr(profile, template=target, roi=roi) for profile in profiles]
        scatter_ratio = (np.std([r.amplitude for r in windowed])
                         / np.std([r.amplitude for r in full]))
        se_ratio = (np.median([r.amplitude_se for r in windowed])
                    / np.median([r.amplitude_se for r in full]))
        print(f"{sigma:>4} {scatter_ratio:>17.2f} {se_ratio:>19.2f}")


def main():
    _print_separation_table()
    _print_length_table()
    _print_window_cost_table()


if __name__ == "__main__":
    main()
