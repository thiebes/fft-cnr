"""Core FFT-based CNR estimation routines."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import get_window
from scipy.signal.windows import tukey
from scipy.stats import chi2


@dataclass
class NoiseModel:
    """Estimated structure of the noise, beyond a single RMS level.

    Characterizes the noise along two orthogonal axes. The real-space axis
    (``read``, ``gain``) captures signal-dependent noise via the
    photon-transfer relation ``var = gain * signal + read**2`` and is
    populated by the ``estimate_noise_model`` detector. The spectral axis
    (``spectral_exponent``, ``white_floor``, ``correlated``) names spatially
    correlated, ``1/f``-type noise; these fields are reserved and are not
    populated. Single-frame quantitative correction of correlated
    noise is unsupported: the low-frequency model error left by an estimated
    signal shape is indistinguishable from ``1/f`` noise in one frame, so the
    spectral exponent and white floor cannot be recovered without bias. Use
    multiple frames, interleaved acquisition, or a reference channel to
    characterize and correct correlated noise. White, signal-independent
    noise is the degenerate case of the real-space axis (zero gain).

    Real-space numeric fields are NaN and ``signal_dependent`` is None until
    the detector has run, so "not tested" is distinguishable from "tested,
    not significant"; the spectral-axis fields stay at those sentinels.

    Attributes
    ----------
    read : float
        Read-noise floor (intercept of the var-vs-signal fit).
    gain : float
        Photon-transfer slope (var-vs-signal).
    spectral_exponent : float
        Reserved (correlated-noise axis); always NaN. Single-frame ``1/f``
        correction is unsupported (see the class notes above).
    white_floor : float
        Reserved (correlated-noise axis); always NaN.
    signal_dependent : bool or None
        Whether the gain is significantly above the pipeline null.
    correlated : bool or None
        Reserved flag for a correlated-noise detector; always None
        (detection deferred, no single-frame correction).
    """

    read: float
    gain: float
    spectral_exponent: float
    white_floor: float
    signal_dependent: bool | None
    correlated: bool | None

    def peak_snr(self, amplitude: float) -> float:
        """Peak signal-to-noise ratio under the fitted real-space noise model.

        Uses the amplitude magnitude, so a negative (absorption / dark-contrast)
        amplitude gives the same positive SNR as a peak of equal depth.
        """
        a = abs(amplitude)
        return float(a / np.sqrt(self.gain * a + self.read**2))


@dataclass
class CNREstimate:
    """Result of an FFT-based CNR estimation.

    Attributes
    ----------
    cnr : float
        Estimated contrast-to-noise ratio.
    cnr_ci95 : tuple[float, float]
        95% confidence interval on CNR (delta-method approximation).
    amplitude : float
        Estimated signal amplitude. Signed on the peak and generalized-Gaussian
        paths: negative for a downward (absorption / dark-contrast) feature.
        ``cnr`` uses its magnitude.
    amplitude_se : float
        Standard error of the amplitude estimate (NaN if unavailable). On the
        matched-filter and generalized-Gaussian paths this is a derived
        standard error; on the default peak path it is an uncharacterized
        proxy (``sigma / sqrt(kc_full)``) that understates the true scatter,
        so it feeds ``cnr_ci95`` (where the noise term dominates) but is not
        exposed through ``amplitude_snr``.
    noise_rms : float
        RMS noise estimated from high-frequency spectral region.
    noise_ci95 : tuple[float, float]
        95% confidence interval on noise RMS.
    cutoff_index : int
        Spectral index separating signal from noise.
    diagnostics : dict
        Additional diagnostic information.
    noise_model : NoiseModel or None
        Estimated noise structure, or None when no detector has run.
    """

    cnr: float
    cnr_ci95: tuple[float, float]
    amplitude: float
    amplitude_se: float
    noise_rms: float
    noise_ci95: tuple[float, float]
    cutoff_index: int
    diagnostics: dict
    noise_model: NoiseModel | None = None

    @property
    def amplitude_snr(self) -> float:
        """Matched-filter signal-to-noise ratio: amplitude over its standard error.

        Defined only on the matched-filter (``template``) path, where the
        standard error is the whitened estimator's, so the ratio is the
        efficient detectability member of the contrast-to-noise-ratio family.
        It is NaN on every other path: the peak proxy standard error is
        uncharacterized, and the generalized-Gaussian standard error is
        denominated in the fit residual rather than the noise spectrum, so
        neither is the same statistical object and the values are not
        comparable. Those paths still expose ``amplitude_se`` directly for any
        caller that wants the raw ratio.
        """
        is_matched = self.diagnostics.get("amplitude_method") == "matched_filter"
        if is_matched and np.isfinite(self.amplitude_se):
            return float(self.amplitude / self.amplitude_se)
        return float("nan")


def _welch_psd_unitary(
    x: np.ndarray, nperseg: int, noverlap: int, win: str = "hann"
) -> tuple[np.ndarray, int]:
    """Unitary Welch PSD estimator with window-energy normalization.

    Parameters
    ----------
    x : np.ndarray
        Demeaned input signal.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of overlapping points between segments.
    win : str
        Window function name.

    Returns
    -------
    Pxx : np.ndarray
        Power spectral density estimate.
    dof : int
        Approximate degrees of freedom (2 * number of segments).
    """
    w = get_window(win, nperseg, fftbins=True)
    W2 = np.sum(w**2)
    step = nperseg - noverlap
    segs = []
    for start in range(0, len(x) - nperseg + 1, step):
        seg = x[start : start + nperseg]
        seg = seg * w
        X = np.fft.rfft(seg, norm="ortho")
        P = (np.abs(X) ** 2) * (len(seg) / W2)
        segs.append(P)
    Pxx = (
        np.mean(segs, axis=0)
        if segs
        else np.abs(np.fft.rfft(x, norm="ortho")) ** 2
    )
    dof = 2 * len(segs)
    return Pxx, dof


def _break_knee_loglog(
    P: np.ndarray, guard: tuple[float, float] = (0.05, 0.5)
) -> int:
    """Two-segment least-squares fit on log-log spectrum with AIC selection.

    Parameters
    ----------
    P : np.ndarray
        Power spectral density array.
    guard : tuple[float, float]
        Fractional bounds (min, max) for the knee search range.

    Returns
    -------
    int
        Index of the spectral knee (signal/noise boundary).
    """
    eps = np.finfo(float).eps
    K = len(P) - 1
    kmin = max(1, int(guard[0] * K))
    kmax = max(kmin + 2, int(guard[1] * K))
    x = np.log(np.arange(1, K + 1))
    y = np.log(P[1:] + eps)
    best_aic, best_k = np.inf, max(kmin, int(0.25 * K))
    for k in range(kmin, kmax):
        x1, y1 = x[:k], y[:k]
        x2, y2 = x[k:], y[k:]
        A1 = np.vstack([x1, np.ones_like(x1)]).T
        A2 = np.vstack([x2, np.ones_like(x2)]).T
        b1, a1 = np.linalg.lstsq(A1, y1, rcond=None)[0]
        b2, a2 = np.linalg.lstsq(A2, y2, rcond=None)[0]
        r1 = y1 - (b1 * x1 + a1)
        r2 = y2 - (b2 * x2 + a2)
        sse = r1.dot(r1) + r2.dot(r2)
        k_params = 4
        n = len(y)
        aic = n * np.log(sse / n + eps) + 2 * k_params
        if aic < best_aic:
            best_aic, best_k = aic, k
    return best_k


class _SpectralDecomposition(NamedTuple):
    """Signal/noise band split of a profile: steps 1-4 of the fft_cnr pipeline.

    Shared by fft_cnr and the noise-model null calibration so that observed
    and null statistics travel the identical path.
    """

    x: np.ndarray  # demeaned input
    x_mean: float  # subtracted mean (physical offset)
    w: np.ndarray  # taper window
    w_rms: float
    X: np.ndarray  # unitary rFFT of the tapered signal
    Pxx_full: np.ndarray  # Welch PSD interpolated to the full rFFT grid
    welch_nperseg: int  # resolved segment length
    welch_noverlap: int  # resolved overlap
    kc_full: int  # signal/noise knee index on the full grid
    x_bp: np.ndarray  # high-frequency residual of the tapered signal
    kept: int  # noise-band bin count
    frac_kept: float  # noise-band fraction (shared attenuation convention)
    nu: int  # chi-squared degrees of freedom for the noise CI
    sigma: float  # noise RMS
    sigma_ci: tuple[float, float]
    x_lp: np.ndarray  # low-pass reconstruction (pre-window signal estimate)


def _spectral_decomposition(
    x: np.ndarray,
    *,
    window: str,
    tukey_alpha: float,
    welch_nperseg: int | None,
    welch_noverlap: int | None,
    cutoff_guard: tuple[float, float],
    fallback_cut_frac: float,
) -> _SpectralDecomposition:
    """Detrend, taper, estimate the PSD, locate the knee, and split bands."""
    N = x.size

    # Detrend and taper
    x_mean = float(np.mean(x))
    x = x - x_mean
    if window == "tukey":
        w = tukey(N, alpha=tukey_alpha)
    elif window == "hann":
        w = get_window("hann", N, fftbins=True)
    else:
        w = np.ones(N)
    xw = x * w
    w_rms = np.sqrt(np.mean(w**2))

    # Unitary rFFT
    X = np.fft.rfft(xw, norm="ortho")

    # Welch PSD (unitary)
    if welch_nperseg is None:
        welch_nperseg = max(16, int(N // 8))
        welch_nperseg += welch_nperseg % 2  # ensure even
    if welch_noverlap is None:
        welch_noverlap = welch_nperseg // 2
    Pxx, dof = _welch_psd_unitary(x, welch_nperseg, welch_noverlap, win="hann")

    # Interpolate Welch PSD to full FFT frequency grid
    nfft_bins = N // 2 + 1
    if len(Pxx) != nfft_bins:
        welch_freqs = np.linspace(0, 1, len(Pxx))
        full_freqs = np.linspace(0, 1, nfft_bins)
        Pxx_full = np.interp(full_freqs, welch_freqs, Pxx)
    else:
        Pxx_full = Pxx

    # Knee (cutoff) with guardrails
    kc = _break_knee_loglog(Pxx, guard=cutoff_guard)
    if not (1 <= kc <= len(Pxx) - 1):
        kc = max(1, int(fallback_cut_frac * (len(Pxx) - 1)))
    # Scale cutoff index to full FFT grid
    kc_full = int(round(kc * (nfft_bins - 1) / (len(Pxx) - 1)))
    kc_full = max(1, min(kc_full, nfft_bins - 1))

    # Noise RMS via bandpass iFFT
    X_bp = X.copy()
    X_bp[:kc_full] = 0.0
    x_bp = np.fft.irfft(X_bp, n=N, norm="ortho")
    kept = max(1, len(X_bp) - kc_full)
    frac_kept = kept / max(1, nfft_bins)
    sigma = float(
        np.sqrt(np.mean(x_bp**2))
        / ((w_rms * np.sqrt(frac_kept)) if w_rms > 0 else 1.0)
    )

    # Confidence interval for sigma via Welch degrees of freedom
    frac = kept / max(1, len(X_bp))
    nu = max(2, int(dof * frac))
    alpha_ci = 0.05
    lower = (nu * sigma**2) / chi2.ppf(1 - alpha_ci / 2, nu)
    upper = (nu * sigma**2) / chi2.ppf(alpha_ci / 2, nu)
    sigma_ci = (np.sqrt(lower), np.sqrt(upper))

    # Low-pass reconstruction of the signal via spectral low-pass filter:
    # FFT the original (pre-window) demeaned signal, zero the noise
    # frequencies, and inverse FFT.  The window used for PSD estimation is
    # not applied here so that the reconstruction preserves the true peak
    # height.  Computed for every amplitude method: the peak method reads
    # its peak from it, and the noise-model detectors bin the
    # high-frequency residual by it.
    X_lp = np.fft.rfft(x, norm="ortho")
    X_lp[kc_full:] = 0.0
    x_lp = np.fft.irfft(X_lp, n=N, norm="ortho")

    return _SpectralDecomposition(
        x=x,
        x_mean=x_mean,
        w=w,
        w_rms=w_rms,
        X=X,
        Pxx_full=Pxx_full,
        welch_nperseg=welch_nperseg,
        welch_noverlap=welch_noverlap,
        kc_full=kc_full,
        x_bp=x_bp,
        kept=kept,
        frac_kept=frac_kept,
        nu=nu,
        sigma=sigma,
        sigma_ci=sigma_ci,
        x_lp=x_lp,
    )


def _fit_photon_transfer(
    signal_estimate: np.ndarray, residual: np.ndarray, frac_kept: float
) -> tuple[float, float]:
    """Fit var = gain * signal + read**2 to per-pixel squared residuals.

    The residual carries only the noise power above the spectral cutoff, so
    the raw slope and intercept are attenuated by the kept band fraction;
    both are corrected with the same ``frac_kept`` convention the noise RMS
    estimate uses, which keeps the two estimators consistent (under white
    noise the corrected intercept matches the squared noise RMS).

    Returns
    -------
    gain : float
        Photon-transfer slope (var-vs-signal).
    read2 : float
        Squared read-noise floor (intercept; may be negative from noise).
    """
    A = np.vstack([signal_estimate, np.ones_like(signal_estimate)]).T
    slope, intercept = np.linalg.lstsq(A, residual**2, rcond=None)[0]
    return float(slope / frac_kept), float(intercept / frac_kept)


def _estimate_noise_model(
    decomp: _SpectralDecomposition,
    rng: np.random.Generator | None,
    *,
    window: str,
    tukey_alpha: float,
    cutoff_guard: tuple[float, float],
    fallback_cut_frac: float,
) -> tuple[NoiseModel, dict]:
    """Estimate the real-space noise model and test for signal dependence.

    Fits the photon-transfer relation to the estimator's own residual, then
    calibrates the significance of the fitted gain against a null built by
    adding white noise at the estimated RMS to the reconstructed signal and
    re-running the full spectral decomposition.  The null travels the same
    path as the observed statistic, so signal-curvature leakage into the
    residual, filter-induced correlation, and knee-placement jitter are all
    represented in the null distribution rather than assumed away.

    Spectral-axis fields stay NaN/None: single-frame correlated-noise
    correction is unsupported (see ``NoiseModel``).

    Returns
    -------
    model : NoiseModel
        Real-space fields populated, or all-NaN/None when the signal range
        cannot constrain the fit.
    diags : dict
        Detector diagnostics to merge into the result diagnostics:
        ``var_signal_p`` (rank p-value of the gain against the null) or
        ``noise_model_skipped`` (reason the fit was not attempted).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    signal_estimate = decomp.x_lp + decomp.x_mean
    residual = decomp.x - decomp.x_lp

    if float(np.ptp(signal_estimate)) < _NOISE_MODEL_MIN_RANGE_SIGMAS * decomp.sigma:
        model = NoiseModel(
            read=float("nan"),
            gain=float("nan"),
            spectral_exponent=float("nan"),
            white_floor=float("nan"),
            signal_dependent=None,
            correlated=None,
        )
        reason = (
            "signal range too small relative to the noise to constrain "
            "the var-vs-signal slope"
        )
        return model, {"noise_model_skipped": reason}

    gain, read2 = _fit_photon_transfer(
        signal_estimate, residual, decomp.frac_kept
    )

    N = decomp.x.size
    null_gains = np.empty(_NOISE_MODEL_NULL_DRAWS)
    for i in range(_NOISE_MODEL_NULL_DRAWS):
        y = signal_estimate + rng.normal(0.0, decomp.sigma, N)
        d = _spectral_decomposition(
            y,
            window=window,
            tukey_alpha=tukey_alpha,
            welch_nperseg=decomp.welch_nperseg,
            welch_noverlap=decomp.welch_noverlap,
            cutoff_guard=cutoff_guard,
            fallback_cut_frac=fallback_cut_frac,
        )
        null_gains[i], _ = _fit_photon_transfer(
            d.x_lp + d.x_mean, d.x - d.x_lp, d.frac_kept
        )

    p_value = float((1 + np.sum(null_gains >= gain)) / (_NOISE_MODEL_NULL_DRAWS + 1))
    model = NoiseModel(
        read=float(np.sqrt(max(0.0, read2))),
        gain=gain,
        spectral_exponent=float("nan"),
        white_floor=float("nan"),
        signal_dependent=bool(p_value <= _NOISE_MODEL_TEST_LEVEL),
        correlated=None,
    )
    return model, {"var_signal_p": p_value}


def _fit_generalized_gaussian_amplitude(x: np.ndarray) -> tuple[float, float, dict]:
    """Fit a generalized Gaussian peak + constant baseline to a 1-D profile.

    The model is ``A * exp(-0.5 * |z|^p) + baseline`` where ``z = (x - center) / sigma``.
    The shape parameter ``p`` controls kurtosis: p=2 is a standard Gaussian,
    p<2 produces heavy tails, and p>2 produces a flat top.

    Parameters
    ----------
    x : np.ndarray
        Input 1-D profile (original, not demeaned).

    Returns
    -------
    amplitude : float
        Fitted peak height above the baseline (NaN on failure).
    amplitude_se : float
        Standard error of the amplitude from the covariance matrix (NaN on failure).
    fit_params : dict
        Fitted parameters {amplitude, center, sigma, baseline, shape} (empty on failure).
    """
    N = len(x)
    x_grid = np.arange(N, dtype=float)

    def _gen_gaussian(xg, amplitude, center, sigma, baseline, shape):
        z = np.abs((xg - center) / sigma)
        return amplitude * np.exp(-0.5 * z ** shape) + baseline

    baseline_init = float(np.median(x))
    residual = x - baseline_init
    peak_idx = int(np.argmax(np.abs(residual)))
    amplitude_init = float(residual[peak_idx])
    center_init = float(peak_idx)
    sigma_init = float(N) / 10.0

    p0 = [amplitude_init, center_init, sigma_init, baseline_init, 2.0]
    bounds = (
        [-np.inf, -0.5 * N, 0.5, -np.inf, 0.5],
        [np.inf, 1.5 * N, float(N), np.inf, 10.0],
    )

    try:
        popt, pcov = curve_fit(_gen_gaussian, x_grid, x, p0=p0, bounds=bounds)
        amplitude = float(popt[0])
        amplitude_se = float(np.sqrt(pcov[0, 0]))
        fit_params = {
            "amplitude": float(popt[0]),
            "center": float(popt[1]),
            "sigma": float(popt[2]),
            "baseline": float(popt[3]),
            "shape": float(popt[4]),
        }
        return amplitude, amplitude_se, fit_params
    except (RuntimeError, ValueError):
        return np.nan, np.nan, {}


_LOWFREQ_OFFPEAK_HALFWIDTH_FRAC = 0.25
_LOWFREQ_DOMINANCE_THRESHOLD = 2.5

# Auto-roi window sizing: half-width is the feature's half-prominence width
# scaled by this factor (1.1 FWHM is about +/- 2.5 sigma), floored so a very
# narrow feature still yields a usable window, and the total span is never
# allowed below the minimum profile length the estimator accepts.
_ROI_HALFWIDTH_FWHM_FACTOR = 1.1
_ROI_HALFWIDTH_FLOOR = 8
_MIN_ROI_SPAN = 16

# Noise-model detector (_estimate_noise_model): number of parametric-bootstrap
# null realizations, the significance level for the gain rank test, and the
# minimum signal range (in noise-RMS units) below which the photon-transfer
# fit is skipped as unconstrained.
_NOISE_MODEL_NULL_DRAWS = 199
_NOISE_MODEL_TEST_LEVEL = 0.05
_NOISE_MODEL_MIN_RANGE_SIGMAS = 3.0


def _feature_peak_and_width(x_lp: np.ndarray) -> tuple[int, int]:
    """Locate the largest-magnitude feature and measure its half-prominence width.

    Both are measured on the signed deviation from the median, so a downward
    (absorption / dark-contrast) feature is handled the same way as an upward
    one. Returns ``(peak_idx, width)`` with ``width >= 2``; ``width`` is the
    full width at half prominence in samples. Shared by the auto-roi window
    sizing and the off-peak ratio so the two locate the feature identically.
    """
    n = x_lp.size
    dev = x_lp - float(np.median(x_lp))
    peak_idx = int(np.argmax(np.abs(dev)))
    peak_dev = float(dev[peak_idx])
    if peak_dev == 0.0:
        return peak_idx, 2
    left = peak_idx
    while left > 0 and dev[left] / peak_dev > 0.5:
        left -= 1
    right = peak_idx
    while right < n - 1 and dev[right] / peak_dev > 0.5:
        right += 1
    return peak_idx, max(2, right - left)


def _lowfreq_offpeak_ratio(x_lp: np.ndarray, sigma: float) -> float:
    """Off-peak low-frequency structure relative to the noise RMS.

    A localized peak sits on a flat baseline, so the low-pass reconstruction
    is flat away from the peak; a smooth baseline (or fringe) carries
    low-frequency structure across the whole profile. This statistic is the
    RMS of ``x_lp`` outside a window around its largest excursion, in units of
    ``sigma``. It is near zero for a clean peak (the off-peak region is just
    the in-band tail of the white noise) and large when a baseline dominates,
    independent of the peak amplitude, so it flags the peakless-baseline case
    without false-flagging a genuinely small peak. It cannot separate a peak
    whose width approaches the profile length from a baseline -- the two are
    indistinguishable in a single frame (see ``NoiseModel``).

    The excluded zone is scaled to the feature's own width (with a fixed
    fraction of the profile as a floor), so the peak's shoulders do not leak
    into the off-peak region. This keeps the statistic honest whether ``x_lp``
    is the full profile or a window already cropped to the feature by ``roi``:
    on a tight window the exclusion covers the whole feature and the off-peak
    region vanishes, returning NaN ("not applicable") rather than a false flag.

    Returns NaN when the off-peak region is too small to estimate or the noise
    RMS is zero.
    """
    N = x_lp.size
    if sigma <= 0:
        return float("nan")
    peak_idx, fwhm = _feature_peak_and_width(x_lp)
    half_width = max(
        int(_LOWFREQ_OFFPEAK_HALFWIDTH_FRAC * N),
        int(round(_ROI_HALFWIDTH_FWHM_FACTOR * fwhm)),
    )
    off_peak = np.abs(np.arange(N) - peak_idx) > half_width
    if int(np.sum(off_peak)) < 4:
        return float("nan")
    off = x_lp[off_peak]
    baseline = float(np.median(off))
    return float(np.sqrt(np.mean((off - baseline) ** 2)) / sigma)


def _resolve_roi(
    x: np.ndarray,
    roi: str | tuple[int, int],
    *,
    window: str,
    tukey_alpha: float,
    welch_nperseg: int | None,
    welch_noverlap: int | None,
    cutoff_guard: tuple[float, float],
    fallback_cut_frac: float,
) -> tuple[int, int]:
    """Resolve a region-of-interest specification to integer ``(start, stop)``.

    ``"auto"`` runs a pre-pass spectral decomposition, locates the peak of the
    low-pass reconstruction, and takes a window scaled to the peak's own width
    (its full width at half prominence): roughly +/- 2.5 sigma, clamped to the
    profile bounds. Sizing the window to the peak is what lets it exclude
    off-center structure; a fixed fraction of the profile would either keep too
    much baseline or clip a broad peak. A two-element sequence is taken as
    explicit ``(start, stop)`` bounds. The window restricts every downstream
    step -- knee detection, noise RMS, and amplitude -- and is the remedy for
    off-center baseline structure that would otherwise leak into the CNR. It
    cannot help against a baseline whose structure spans the peak itself; the
    ``lowfreq_dominated`` diagnostic flags that residual case.
    """
    N = x.size
    if isinstance(roi, str):
        if roi != "auto":
            raise ValueError(
                f"Unsupported roi={roi!r}. Use 'auto' or a (start, stop) pair."
            )
        pre = _spectral_decomposition(
            x,
            window=window,
            tukey_alpha=tukey_alpha,
            welch_nperseg=welch_nperseg,
            welch_noverlap=welch_noverlap,
            cutoff_guard=cutoff_guard,
            fallback_cut_frac=fallback_cut_frac,
        )
        peak_idx, fwhm = _feature_peak_and_width(pre.x_lp)
        # FWHM is ~2.355 sigma; ~2.5 sigma each side (about 1.1 FWHM) keeps the
        # peak and its shoulders while dropping off-center structure. Enforce a
        # usable span and slide it inside the profile so a peak near an edge
        # (common when there is no real peak to find) still yields a window.
        half_width = max(
            _ROI_HALFWIDTH_FLOOR, int(round(_ROI_HALFWIDTH_FWHM_FACTOR * fwhm))
        )
        span = min(N, max(_MIN_ROI_SPAN, 2 * half_width + 1))
        start = max(0, min(peak_idx - span // 2, N - span))
        stop = start + span
    else:
        bounds = tuple(int(v) for v in roi)
        if len(bounds) != 2:
            raise ValueError(
                f"Region of interest must be 'auto' or a (start, stop) pair; "
                f"got {len(bounds)} values."
            )
        start, stop = bounds
        if stop <= start:
            raise ValueError(
                f"Region of interest bounds must be increasing; got "
                f"(start={start}, stop={stop})."
            )
        start = max(0, start)
        stop = min(N, stop)
        if stop - start < _MIN_ROI_SPAN:
            raise ValueError(
                f"Region of interest spans fewer than {_MIN_ROI_SPAN} points; "
                f"widen the window or pass the full profile."
            )
    return start, stop


def fft_cnr(
    x: np.ndarray,
    template: np.ndarray | None = None,
    *,
    fit_model: str | None = None,
    window: str = "tukey",
    tukey_alpha: float = 0.25,
    welch_nperseg: int | None = None,
    welch_noverlap: int | None = None,
    cutoff_guard: tuple[float, float] = (0.05, 0.5),
    fallback_cut_frac: float = 0.25,
    roi: str | tuple[int, int] | None = None,
    return_bandpassed_noise: bool = False,
    estimate_noise_model: bool = False,
    rng: np.random.Generator | None = None,
) -> CNREstimate:
    """Estimate contrast-to-noise ratio from a single 1-D profile using FFT methods.

    Uses unitary FFT normalization, Welch PSD estimation with degrees-of-freedom
    tracking, AIC-based objective cutoff selection, white-noise matched-filter
    amplitude estimation, and analytical confidence intervals.

    Parameters
    ----------
    x : np.ndarray
        Input 1-D signal (e.g., a line profile or spectrum).
    template : np.ndarray or None
        Expected signal shape for matched-filter amplitude estimation.
        If None, amplitude is estimated using the method specified by
        ``fit_model`` (default: ``"peak"``).
    fit_model : str or None
        Amplitude estimation method when no template is provided.
        ``"peak"`` (default) applies a spectral low-pass filter and
        reads the peak from the smoothed signal -- robust across
        arbitrary profile shapes. ``"generalized_gaussian"`` fits a
        5-parameter generalized Gaussian with a shape exponent that
        accommodates non-zero excess kurtosis, providing fitted
        parameters (center, width, shape) in diagnostics.
        Ignored when ``template`` is given.
    window : str
        Tapering window: ``"tukey"``, ``"hann"``, or ``"none"``.
    tukey_alpha : float
        Shape parameter for the Tukey window (0 = rectangular, 1 = Hann).
    welch_nperseg : int or None
        Segment length for Welch PSD estimation. Defaults to max(16, N//8).
        With 50% overlap this heuristic produces approximately 15 Welch
        segments at any signal length, giving consistent degrees of freedom
        for the noise confidence interval. Longer segments reduce the segment
        count and widen the noise CI substantially without improving CNR
        accuracy.
    welch_noverlap : int or None
        Overlap for Welch segments. Defaults to nperseg // 2.
    cutoff_guard : tuple[float, float]
        Fractional bounds for the AIC knee search range.
    fallback_cut_frac : float
        Fallback cutoff fraction if AIC selection fails.
    roi : str, tuple[int, int], or None
        Restrict the estimate to a region of interest. ``None`` (default)
        uses the full profile. A ``(start, stop)`` index pair estimates on
        that slice. ``"auto"`` locates the largest feature (peak or dip) and
        takes a window scaled to its own width (about +/- 2.5 sigma). Windowing
        removes off-center low-frequency baseline structure that would otherwise
        be counted as signal; the chosen bounds are reported in
        ``diagnostics["roi"]``. ``"auto"`` locates the largest feature, so when
        an off-center baseline exceeds the peak of interest, pass explicit
        bounds instead. The window must span at least 16 points.
    return_bandpassed_noise : bool
        If True, include the bandpassed noise array in diagnostics.
    estimate_noise_model : bool
        If True, fit the photon-transfer relation (var = gain * signal +
        read**2) to the residual, test the fitted gain for significance
        against a null calibrated through this same pipeline, and attach
        the result as ``noise_model``.  Works with every amplitude method.
        Adds a Monte Carlo cost of about 200 re-runs of the spectral
        decomposition.
    rng : numpy.random.Generator or None
        Random generator for the noise-model null calibration.  None
        (default) uses a fixed seed, so repeated calls on the same input
        give identical results; pass a Generator for independent draws.
        Unused unless ``estimate_noise_model`` is True.

    Returns
    -------
    CNREstimate
        Dataclass containing CNR, amplitude, noise, confidence intervals,
        and diagnostic information. On the localized-peak methods (``peak`` and
        ``generalized_gaussian``) the diagnostics carry
        ``lowfreq_offpeak_ratio`` -- the RMS of low-frequency structure away
        from the peak, in units of the noise RMS -- and the boolean
        ``lowfreq_dominated`` (true above an off-peak ratio of 2.5). A true
        flag means smooth baseline or fringe structure dominates the profile,
        so the reported CNR may reflect baseline power rather than the peak;
        narrow the estimate with ``roi``. The ratio is NaN on the
        matched-filter (``template``) path, where the template defines the
        signal and the off-peak statistic does not apply.

    Raises
    ------
    ValueError
        If the input profile has fewer than 16 points, or a region of interest
        spans fewer than 16 points.
    """
    x = np.asarray(x, float).ravel()
    N = x.size
    if N < 16:
        raise ValueError("Profile too short for stable PSD estimation.")
    _valid_fit_models = {"generalized_gaussian", "peak"}
    if fit_model is not None and fit_model not in _valid_fit_models:
        raise ValueError(
            f"Unsupported fit_model={fit_model!r}. "
            f"Supported values: {sorted(_valid_fit_models)}."
        )

    roi_bounds: tuple[int, int] | None = None
    if roi is not None:
        roi_bounds = _resolve_roi(
            x,
            roi,
            window=window,
            tukey_alpha=tukey_alpha,
            welch_nperseg=welch_nperseg,
            welch_noverlap=welch_noverlap,
            cutoff_guard=cutoff_guard,
            fallback_cut_frac=fallback_cut_frac,
        )
        # Slice a full-length template to the same window as the data so the two
        # stay aligned; without this a full-profile template would be truncated
        # from its start and the matched filter would project onto the wrong
        # samples. A template already sized to the ROI is left untouched.
        if template is not None:
            t_full = np.asarray(template, float).ravel()
            if t_full.size == N:
                template = t_full[roi_bounds[0] : roi_bounds[1]]
        x = x[roi_bounds[0] : roi_bounds[1]]
        N = x.size

    d = _spectral_decomposition(
        x,
        window=window,
        tukey_alpha=tukey_alpha,
        welch_nperseg=welch_nperseg,
        welch_noverlap=welch_noverlap,
        cutoff_guard=cutoff_guard,
        fallback_cut_frac=fallback_cut_frac,
    )
    x = d.x
    x_mean = d.x_mean
    w = d.w
    kc_full = d.kc_full
    sigma = d.sigma
    sigma_ci = d.sigma_ci
    x_lp = d.x_lp

    noise_model = None
    noise_model_diags: dict = {}
    if estimate_noise_model:
        noise_model, noise_model_diags = _estimate_noise_model(
            d,
            rng,
            window=window,
            tukey_alpha=tukey_alpha,
            cutoff_guard=cutoff_guard,
            fallback_cut_frac=fallback_cut_frac,
        )

    # Amplitude estimation
    gfit_params: dict = {}
    if template is not None:
        t = np.asarray(template, float).ravel()
        if t.size != N:
            if t.size > N:
                t = t[:N]
            else:
                t = np.pad(t, (0, N - t.size))
        t = t - np.mean(t)
        # White-noise matched filter: project the windowed, demeaned data onto
        # the windowed template.  The earlier version whitened by the full data
        # power spectrum, but that spectrum is signal-contaminated in the signal
        # band, which inflated the standard error (and hence cnr_ci95) on this
        # path.  For the broadband-white noise the package targets, the optimal
        # weighting is flat, so the unweighted projection is the efficient
        # estimator and its standard error follows in closed form.
        tw = t * w
        xw = x * w
        denom = float(np.sum(tw * tw))
        Amp = float(np.sum(xw * tw) / denom)
        # Exact standard error of the projection under white noise of level
        # sigma: var(Amp) = sigma**2 * sum(w**2 * tw**2) / denom**2, which
        # reduces to the textbook sigma / ||t|| when the window is flat.
        Amp_se = float(sigma * np.sqrt(np.sum((w * tw) ** 2)) / denom)
        amp_method = "matched_filter"
    elif fit_model == "generalized_gaussian":
        x_raw = x + x_mean
        Amp, Amp_se, gfit_params = _fit_generalized_gaussian_amplitude(x_raw)
        if np.isnan(Amp):
            amp_method = "generalized_gaussian_fit_fallback"
        else:
            amp_method = "generalized_gaussian_fit"
    else:
        # Default: non-parametric peak read from the low-pass reconstruction.
        amp_method = "peak"

    if amp_method in ("peak", "generalized_gaussian_fit_fallback"):
        margin = max(1, N // 4)
        x_raw = x + x_mean
        baseline = float(np.mean(np.concatenate([x_raw[:margin], x_raw[-margin:]])))
        # Read the largest-magnitude excursion from the baseline so a negative
        # (absorption / dark-contrast) feature is measured with the right sign;
        # for a positive peak this is the peak height, unchanged.
        dev = (x_lp + x_mean) - baseline
        Amp = float(dev[int(np.argmax(np.abs(dev)))])
        Amp_se = float(sigma / np.sqrt(max(1, kc_full)))

    # CNR and confidence interval (delta-method)
    cnr_val = np.inf if sigma == 0 else float(np.abs(Amp) / sigma)
    if np.isfinite(cnr_val) and np.isfinite(Amp_se):
        var_A = Amp_se**2
        se_sigma = 0.5 * (sigma_ci[1] - sigma_ci[0]) / 1.96
        var_sigma = se_sigma**2
        var_cnr = var_A / (sigma**2) + (Amp**2) * var_sigma / (sigma**4)
        se_cnr = np.sqrt(max(0.0, var_cnr))
        cnr_ci = (cnr_val - 1.96 * se_cnr, cnr_val + 1.96 * se_cnr)
    else:
        cnr_ci = (np.nan, np.nan)

    # The flat-baseline assumption only applies to the localized-peak methods;
    # the matched filter defines its signal through the template and rejects
    # non-matching structure by projection, so the off-peak statistic (which
    # would read an extended template's own lobes as baseline) does not apply.
    if template is None:
        offpeak_ratio = _lowfreq_offpeak_ratio(x_lp, sigma)
    else:
        offpeak_ratio = float("nan")
    lowfreq_dominated = bool(
        np.isfinite(offpeak_ratio)
        and offpeak_ratio > _LOWFREQ_DOMINANCE_THRESHOLD
    )

    diags: dict = {
        "N": N,
        "window_rms": d.w_rms,
        "dof": d.nu,
        "welch_nperseg": d.welch_nperseg,
        "welch_noverlap": d.welch_noverlap,
        "kept_bins": d.kept,
        "amplitude_method": amp_method,
        "lowfreq_offpeak_ratio": offpeak_ratio,
        "lowfreq_dominated": lowfreq_dominated,
    }
    if roi_bounds is not None:
        diags["roi"] = roi_bounds
    if gfit_params:
        diags["gaussian_fit_params"] = gfit_params
    if return_bandpassed_noise:
        diags["x_bp"] = d.x_bp
    diags.update(noise_model_diags)

    return CNREstimate(
        cnr=cnr_val,
        cnr_ci95=cnr_ci,
        amplitude=Amp,
        amplitude_se=Amp_se,
        noise_rms=sigma,
        noise_ci95=sigma_ci,
        cutoff_index=kc_full,
        diagnostics=diags,
        noise_model=noise_model,
    )
