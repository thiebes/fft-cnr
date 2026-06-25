"""Generate the project hero image: one signal, two domains, one number.

The figure shows what ``fft_cnr`` does as a mechanism, not just a result. The
left panel is the measured profile in real space; the right panel is its power
spectrum, where the method finds the knee that separates signal from noise.
Zeroing the bins above the knee and inverting gives the smooth reconstruction
whose peak, divided by the noise RMS, is the CNR.

Every annotated number is read from a real ``fft_cnr`` run on the demo profile
defined below (a narrow Gaussian peak in white noise, seed 0), so the picture
stays correct if the method changes.

The hero uses a Tufte range-frame: open top/right spines, with the remaining
spines drawn only across the span the data occupy.

Re-render with:

    uv run --group assets python assets/hero.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
import matplotlib.patheffects as pe

from curved_text import curved_text

from fft_cnr import fft_cnr
from fft_cnr.core import (
    _break_knee_loglog,
    _spectral_decomposition,
    _welch_psd_unitary,
)

# --- Plot style: Spanish flag palette.
# Two flag colors (red, yellow) plus complementary accents. ------------------
BLUE = "#0077c8"        # Spanish-flag accent blue: signal / data
GOLD = "#FFC400"        # flag_yellow: noise fills and labels (with a shadow)
RED = "#C60B1E"        # flag_red: salient markers (knee, amplitude)
GRAY = "#9aa0a6"        # neutral: receding raw data and baseline
INK = "#333333"         # near-black: the CNR headline / synthesized result
inch = 1 / 2.54
FONT = 9

# --- Real data: the README quick-start profile ------------------------------
rng = np.random.default_rng(0)
N = 384
pos = np.arange(N, dtype=float)
signal = 10.0 * np.exp(-0.5 * ((pos - N // 2) / 5) ** 2)
noisy = signal + rng.normal(0, 1.0, N)

result = fft_cnr(noisy)
d = _spectral_decomposition(
    noisy,
    window="tukey",
    tukey_alpha=0.25,
    welch_nperseg=None,
    welch_noverlap=None,
    cutoff_guard=(0.05, 0.5),
    fallback_cut_frac=0.25,
)

recon = d.x_lp + d.x_mean                     # smooth low-pass reconstruction
kc = d.kc_full                                # signal/noise knee (full grid)
sigma = result.noise_rms
amplitude = result.amplitude
cnr = result.cnr
cnr_lo, cnr_hi = result.cnr_ci95             # delta-method 95% CI on the CNR

# Baseline exactly as fft_cnr computes it (outer-quarter mean of the raw input).
margin = max(1, N // 4)
baseline = float(np.mean(np.concatenate([noisy[:margin], noisy[-margin:]])))
peak_x = int(np.argmax(d.x_lp))
peak_y = baseline + amplitude

eps = np.finfo(float).eps

# Panel B shows the knee detection on the spectrum the detector actually uses:
# the Welch PSD (recomputed identically to _spectral_decomposition), the
# log-log two-segment fit at every candidate breakpoint the AIC search tries,
# and the breakpoint AIC selects. Frequencies are bin / nperseg (cycles/pixel).
Pxx, _dof = _welch_psd_unitary(
    d.x, d.welch_nperseg, d.welch_noverlap, win="hann"
)
welch_freq = np.arange(len(Pxx)) / d.welch_nperseg

# Replicate the search grid and per-bin fits exactly (see _break_knee_loglog):
# x is the log bin index, y is log power, and each candidate k splits the band
# into [1, k) and [k, K]. The winner comes from the real function, not this
# replica, so the highlighted fit is exactly what the detector chose.
guard = (0.05, 0.5)
Kw = len(Pxx) - 1
kmin = max(1, int(guard[0] * Kw))
kmax = max(kmin + 2, int(guard[1] * Kw))
bx_log = np.log(np.arange(1, Kw + 1))
by_log = np.log(Pxx[1:] + eps)


def _line(xs, ys):
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    return slope, intercept


def _candidate_fit(k):
    """Two segments split at bin ``k`` as (freq, power) polylines, plus AIC."""
    m1, b1 = _line(bx_log[:k], by_log[:k])
    m2, b2 = _line(bx_log[k:], by_log[k:])
    lo_bins, hi_bins = np.arange(1, k + 1), np.arange(k, Kw + 1)
    lo = (lo_bins / d.welch_nperseg, np.exp(m1 * np.log(lo_bins) + b1))
    hi = (hi_bins / d.welch_nperseg, np.exp(m2 * np.log(hi_bins) + b2))
    r1 = by_log[:k] - (m1 * bx_log[:k] + b1)
    r2 = by_log[k:] - (m2 * bx_log[k:] + b2)
    sse = float(r1 @ r1 + r2 @ r2)
    aic = len(by_log) * np.log(sse / len(by_log) + eps) + 2 * 4
    return lo, hi, aic


candidate_ks = list(range(kmin, kmax))
candidates = {k: _candidate_fit(k) for k in candidate_ks}
kc_welch = _break_knee_loglog(Pxx, guard=guard)     # the breakpoint AIC selects
f_knee = kc_welch / d.welch_nperseg

# The full-resolution FFT power spectrum, in the same unitary convention as the
# Welch PSD (|X|^2 / window energy) so the two share one power scale. It
# resolves the signal band into many bins; the Welch PSD is the coarser
# estimate (segment-averaged, sampled at fewer frequencies) the knee search
# runs on.
fft_freq = np.arange(len(d.X)) / N
fft_power = np.abs(d.X) ** 2 / d.w_rms**2


def render():
    """Build and save the figure."""
    fig, (axA, axB) = plt.subplots(
        1, 2, layout="constrained", figsize=(18 * inch, 7.5 * inch)
    )
    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)

    # === Panel A: real space ================================================
    axA.plot(pos, noisy, color=GRAY, lw=0.8, alpha=0.8, zorder=1)
    axA.plot(pos, recon, color=BLUE, lw=2, zorder=4)

    # Noise RMS as a symmetric +/- sigma envelope about the baseline: this is
    # what zero-mean noise is, and it visibly contains the bulk of the gray
    # scatter. Labelled +/- sigma so the magnitude is explicit (Tufte: a
    # labelled symmetric band is faithful and commits no lie of proportion).
    # Flag-yellow edges at +/- sigma with a fainter fill between, kept at the
    # same weight as the faint FFT spectrum in panel B so the noise yellow reads
    # consistently across both panels.
    axA.axhspan(baseline - sigma, baseline + sigma, color=GOLD, alpha=0.3,
                lw=0, zorder=0)
    for yb in (baseline - sigma, baseline + sigma):
        axA.axhline(yb, color=GOLD, lw=0.8, alpha=0.5, zorder=2)
    axA.axhline(baseline, color=GRAY, lw=0.8, zorder=2)
    # Label centered just below the band, in the void under the peak.
    axA.text(N / 2, baseline - sigma - 1.8,
             "noise RMS\n" + r"$\pm\sigma_{\mathrm{rms}}$ = " + f"{sigma:.2f}",
             color=GOLD, fontsize=FONT, va="center", ha="center",
             path_effects=[pe.withSimplePatchShadow(offset=(0.5, -0.5),
                           shadow_rgbFace="#444444", alpha=0.95)])

    # Amplitude as a plain range bar from baseline to peak (no arrowheads), in
    # flag red so it reads as the key measurement, distinct from the blue
    # recovered-signal curve.
    bx = peak_x + 16
    cap = 5
    axA.plot([bx, bx], [baseline, peak_y], color=RED, lw=1.3, zorder=5)
    axA.plot([bx - cap, bx + cap], [peak_y, peak_y], color=RED, lw=1.3, zorder=5)
    axA.plot([bx - cap, bx + cap], [baseline, baseline], color=RED, lw=1.3, zorder=5)

    # Amplitude label near the top of its bracket, in flag red.
    axA.text(bx + 9, baseline + 0.72 * amplitude,
             f"amplitude\n$A$ = {amplitude:.1f}",
             color=RED, fontsize=FONT, va="center", ha="left")

    # CNR headline as three columns so the two equals signs align exactly
    # (proportional fonts ignore leading-space alignment).
    cnr_y, cnr_dy, x_eq = 0.95, 0.082, 0.20
    cnr_label = axA.text(x_eq - 0.05, cnr_y, "CNR", transform=axA.transAxes,
                         fontsize=FONT + 2, va="top", ha="right", color=INK)
    for row, rhs in enumerate((r"$A/\sigma_{\mathrm{rms}}$", f"{cnr:.1f}")):
        yy = cnr_y - row * cnr_dy
        axA.text(x_eq, yy, "=", transform=axA.transAxes, fontsize=FONT + 2,
                 va="top", ha="center", color=INK)
        axA.text(x_eq + 0.035, yy, rhs, transform=axA.transAxes,
                 fontsize=FONT + 2, va="top", ha="left", color=INK)
    # The 95% CI is placed after layout (see below), aligned under the CNR label.

    # "recovered signal" runs up the steep rising flank of the peak, near
    # vertical, offset to the left so it clears the blue curve. The guide is the
    # real reconstruction over the flank (not an idealized Gaussian).
    flank = (pos >= peak_x - 12) & (pos <= peak_x)
    curved_text(axA, pos[flank], recon[flank], "recovered signal",
                anchor="center", pos=0.6, offset=8, color=BLUE,
                fontsize=FONT, zorder=6)

    axA.set_xlim(0, N - 1)
    axA.set_xlabel("position (pixel)", fontsize=FONT)
    axA.set_ylabel("intensity (arb. units)", fontsize=FONT)

    # === Panel B: the AIC knee search ======================================
    # Full-resolution FFT spectrum (faint line) resolves the signal band into
    # many bins; the Welch PSD (markers) is the coarser per-bin point estimate
    # the knee search runs on. The selected two-segment fit is drawn on the
    # spectrum; the search itself (the objective it minimizes) is the inset.
    MODEL = "#4a4f54"
    # Raw FFT split at the knee: blue on the signal side, flag yellow on the
    # noise side, so the line itself carries the signal/noise division.
    ksplit = int(np.searchsorted(fft_freq, f_knee))
    axB.loglog(fft_freq[1:ksplit + 1], fft_power[1:ksplit + 1], color=BLUE,
               lw=0.7, alpha=0.30, zorder=0)
    axB.loglog(fft_freq[ksplit:], fft_power[ksplit:], color=GOLD, lw=0.7,
               alpha=0.5, zorder=0)
    # The selected two-segment fit -- the model whose breakpoint is the knee --
    # in flag red. The steep signal segment rides on top of the Welch points it
    # was fitted to (zorder above the markers); the flat noise-floor segment
    # stays beneath, so its points scatter on top of it.
    lo, hi, _ = candidates[kc_welch]
    if len(lo[0]) >= 2:
        axB.loglog(lo[0], lo[1], color=RED, lw=1.6, zorder=4)
    if len(hi[0]) >= 2:
        axB.loglog(hi[0], hi[1], color=RED, lw=1.6, zorder=2)
    # The Welch PSD as markers: one point estimate per frequency bin, each an
    # average over segments at that frequency -- coarse because short segments
    # give wide bin spacing (sparse frequency sampling, not band-averaging).
    # These are exactly the (log freq, log power) points the AIC two-segment fit
    # minimizes residuals to.
    axB.loglog(welch_freq[1:], Pxx[1:], ls="none", marker="o", ms=4,
               color=BLUE, zorder=3)

    axB.set_xlim(fft_freq[1], welch_freq[-1])
    axB.set_ylim(0.25, max(fft_power[1], Pxx[1]) * 1.6)
    axB.set_yticks([1, 10, 100])         # decades within the data; no stray tick
    ymin, ymax = axB.get_ylim()
    # Knee divider, full height -- the search inset moved to the lower-left, so
    # the line is clear to run to the top.
    axB.axvline(f_knee, color=RED, lw=1.0, ls=(0, (4, 3)), zorder=5)
    # The signal/noise split is carried by the FFT line's two colours and the
    # legend, so no separate region labels are needed -- just the knee marker.
    axB.text(f_knee * 1.18, ymin * (ymax / ymin) ** 0.62, "AIC knee",
             color=RED, fontsize=FONT, rotation=90, rotation_mode="anchor",
             va="center", ha="center")
    # Series legend for the two spectra (the knee/fit annotations stay labelled
    # in situ -- they do not belong in the legend). The FFT line is split at the
    # knee into signal (blue) and noise (gold) halves.
    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=0.8, label="FFT (signal)"),
        Line2D([0], [0], color=GOLD, lw=0.8, label="FFT (noise)"),
        Line2D([0], [0], color=BLUE, ls="none", marker="o", ms=4,
               label="Welch PSD"),
    ]
    axB.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=FONT - 2,
               frameon=True, facecolor="white", framealpha=0.85,
               edgecolor="#bbbbbb", handlelength=1.4, labelspacing=0.3)

    # The AIC search as its objective: Delta-AIC vs candidate breakpoint
    # frequency, dipping to its minimum at the chosen knee. In the lower-left,
    # the open corner below the signal descent.
    a = np.array([candidates[k][2] for k in candidate_ks])
    daic = a - float(a.min())
    bf = np.array([k / d.welch_nperseg for k in candidate_ks])
    axi = axB.inset_axes([0.09, 0.12, 0.42, 0.32])
    axi.axvline(f_knee, color=RED, lw=0.8, ls=(0, (3, 3)), alpha=0.5, zorder=1)
    axi.plot(bf, daic, color=MODEL, lw=1.0, marker="o", ms=2.5, zorder=2)
    axi.plot([f_knee], [0.0], marker="o", ms=5, color=RED, zorder=3)
    axi.set_xscale("log")
    axi.set_xlim(bf.min() * 0.85, bf.max() * 1.12)
    axi.set_ylim(-0.06 * daic.max(), daic.max() * 1.2)
    # Schematic of the objective: the shape (dip at the chosen breakpoint) is
    # the message, so the axes carry labels but no numeric values.
    # Schematic: only the dip shape carries meaning, so no ticks at all -- the
    # light frame and the curve falling to the chosen knee tell the whole story.
    axi.tick_params(axis="both", which="both", bottom=False, left=False,
                    top=False, right=False, labelleft=False, labelbottom=False)
    for s in axi.spines.values():        # full 4-sided frame, kept light
        s.set_color("#bbbbbb")
        s.set_linewidth(0.8)
    axi.set_ylabel(r"$\Delta$AIC", fontsize=FONT - 2, labelpad=2)
    axi.text(0.5, 1.04, "knee search", transform=axi.transAxes,
             fontsize=FONT - 2, color=RED, va="bottom", ha="center")

    axB.set_xlabel(r"$\log_{10}$ spatial frequency (cycles / pixel)",
                   fontsize=FONT)
    axB.set_ylabel(r"$\log_{10}$ power (arb. units)", fontsize=FONT)

    # === Frame treatment: Tufte range-frame =================================
    # Drop the top/right spines and draw the remaining spines only across the
    # span the data occupy.
    yA = (min(noisy.min(), baseline), max(noisy.max(), peak_y))
    axA.set_ylim(yA[0] - 0.3, yA[1] + 0.3)
    yB_hi = max(float(fft_power[1:].max()), float(Pxx[1:].max()))
    spans = {axA: ((0, N - 1), yA),
             axB: ((fft_freq[1], welch_freq[-1]), (0.25, yB_hi))}
    for ax, ((xlo, xhi), (ylo, yhi)) in spans.items():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_bounds(xlo, xhi)
        ax.spines["left"].set_bounds(ylo, yhi)
        ax.tick_params(axis="both", which="major", direction="in",
                       top=False, right=False, labelsize=FONT,
                       length=4, width=1)
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)

    # Per-axis value policy:
    # - panel A x (position): no ticks or numbers -- the raw trace shows the
    #   resolution, and pixel index carries no message here.
    # - panel A y (intensity): keep numbers so A and sigma connect to the scale.
    # - panel B (log-log): keep decade ticks but drop numbers; the axis labels
    #   say log10, so with minor ticks off each tick reads as one decade.
    for ax in (axA, axB):        # no ticks on the top or right of either panel
        ax.tick_params(axis="both", which="both", top=False, right=False)
    axA.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axA.minorticks_off()         # linear panel: decade-style minors would mislead
    axB.tick_params(axis="both", which="both", labelbottom=False,
                    labelleft=False)

    # A quiet log-scale cue on panel B: short, light minor ticks. With the
    # decade numbers removed the major ticks alone are evenly spaced and could
    # read as linear; the minor-tick bunching marks the axis as logarithmic
    # without competing with the data or the major decades.
    axB.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    axB.tick_params(axis="both", which="minor", direction="in",
                    bottom=True, left=True, top=False, right=False,
                    length=2, width=0.6, color="#b0b0b0")

    # The 95% CI beneath the headline, its left edge aligned with the "CNR"
    # label (measured after layout): the CNR is a measurement with uncertainty,
    # not a point, and here the interval is wide -- the noise term dominates.
    fig.canvas.draw()
    cnr_x_left = axA.transAxes.inverted().transform(
        (cnr_label.get_window_extent().x0, 0.0))[0]
    axA.text(cnr_x_left, cnr_y - 2 * cnr_dy - 0.02,
             f"95% CI [{cnr_lo:.1f}, {cnr_hi:.1f}]", transform=axA.transAxes,
             fontsize=FONT - 2, va="top", ha="left", color="#6b6f73")

    out = "assets/hero.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


print("wrote", render())
print(f"CNR={cnr:.1f}, A={amplitude:.2f}, sigma={sigma:.3f}, knee bin={kc}")
