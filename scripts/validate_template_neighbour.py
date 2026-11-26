"""How a neighbouring feature shifts the matched-filter amplitude, and what roi does.

The matched filter projects the demeaned profile onto the demeaned template.
Away from the target the demeaned template is negative, so a second feature in
the window moves the amplitude in proportion to its own height: a neighbour of
the same sign lowers it. The projection is linear in the profile, so the shift
does not depend on the noise. The script measures it exactly, as the
difference between two runs on the same noise draw, one with the neighbour and
one without.

Profiles: N = 512, a Gaussian target of peak 1 and SD sigma centered on sample
200, and a Gaussian neighbour of the same SD at a separation d to its right.
The template is the target alone, at full length. Noise is white with SD 0.1
(nominal CNR 10), drawn per (sigma, d) cell from
``np.random.default_rng([4758, si, di])`` (indices of sigma and d); it cancels
in the difference.

For each cell the script prints the shift per unit neighbour height, as a
percentage of the target's peak, on the full record and with ``roi`` set to
+/-3 sigma around the target. A neighbour h times as high shifts the amplitude
h times as much.

Usage:
    uv run python scripts/validate_template_neighbour.py
"""

import numpy as np

from fft_cnr import fft_cnr

N = 512
TARGET_CENTER = 200
SIGMAS = (5, 10, 20)
SEPARATIONS_IN_SD = (4, 5, 6, 8, 12)
NOISE_SD = 0.1
ROI_HALF_WIDTH_IN_SD = 3


def _gaussian(center, sigma):
    x = np.arange(N, dtype=float)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _neighbour_shift(target, neighbour, noise, roi):
    """Amplitude change caused by the neighbour alone, in units of the target's peak."""
    with_neighbour = fft_cnr(target + neighbour + noise, template=target, roi=roi)
    without = fft_cnr(target + noise, template=target, roi=roi)
    return with_neighbour.amplitude - without.amplitude


def main():
    print(f"N = {N}, target peak 1 at sample {TARGET_CENTER}, neighbour of height 1 and the "
          "same SD. Shift of the matched-filter amplitude, % of the target's peak.")
    print(f"{'sigma':>5} {'d / sigma':>9} {'full record':>12} "
          f"{f'roi +/-{ROI_HALF_WIDTH_IN_SD} sigma':>15}")
    for si, sigma in enumerate(SIGMAS):
        target = _gaussian(TARGET_CENTER, sigma)
        half = ROI_HALF_WIDTH_IN_SD * sigma
        roi = (TARGET_CENTER - half, TARGET_CENTER + half + 1)
        for di, separation in enumerate(SEPARATIONS_IN_SD):
            neighbour = _gaussian(TARGET_CENTER + separation * sigma, sigma)
            noise = np.random.default_rng([4758, si, di]).normal(0, NOISE_SD, N)
            full = _neighbour_shift(target, neighbour, noise, None)
            windowed = _neighbour_shift(target, neighbour, noise, roi)
            print(f"{sigma:>5} {separation:>9} {100 * full:>11.2f}% {100 * windowed:>14.2f}%")


if __name__ == "__main__":
    main()
