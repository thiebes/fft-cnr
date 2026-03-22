"""Core FFT-based CNR estimation routines."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import get_window
from scipy.signal.windows import tukey
from scipy.stats import chi2


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
        Estimated signal amplitude.
    amplitude_se : float
        Standard error of the amplitude estimate (NaN if unavailable).
    noise_rms : float
        RMS noise estimated from high-frequency spectral region.
    noise_ci95 : tuple[float, float]
        95% confidence interval on noise RMS.
    cutoff_index : int
        Spectral index separating signal from noise.
    diagnostics : dict
        Additional diagnostic information.
    """

    cnr: float
    cnr_ci95: tuple[float, float]
    amplitude: float
    amplitude_se: float
    noise_rms: float
    noise_ci95: tuple[float, float]
    cutoff_index: int
    diagnostics: dict


def _welch_psd_unitary(
    xw: np.ndarray, nperseg: int, noverlap: int, win: str = "hann"
) -> tuple[np.ndarray, int]:
    """Unitary Welch PSD estimator with window-energy normalization.

    Parameters
    ----------
    xw : np.ndarray
        Windowed input signal.
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
    for start in range(0, len(xw) - nperseg + 1, step):
        seg = xw[start : start + nperseg]
        seg = seg * w
        X = np.fft.rfft(seg, norm="ortho")
        P = (np.abs(X) ** 2) * (len(seg) / W2)
        segs.append(P)
    Pxx = (
        np.mean(segs, axis=0)
        if segs
        else np.abs(np.fft.rfft(xw, norm="ortho")) ** 2
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
    return_bandpassed_noise: bool = False,
) -> CNREstimate:
    """Estimate contrast-to-noise ratio from a single 1-D profile using FFT methods.

    Uses unitary FFT normalization, Welch PSD estimation with degrees-of-freedom
    tracking, AIC-based objective cutoff selection, whitened matched-filter amplitude
    estimation, and analytical confidence intervals.

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
    return_bandpassed_noise : bool
        If True, include the bandpassed noise array in diagnostics.

    Returns
    -------
    CNREstimate
        Dataclass containing CNR, amplitude, noise, confidence intervals,
        and diagnostic information.

    Raises
    ------
    ValueError
        If the input profile has fewer than 16 points.
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
    Pxx, dof = _welch_psd_unitary(xw, welch_nperseg, welch_noverlap, win="hann")

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
        Tw = t * w
        T = np.fft.rfft(Tw, norm="ortho")
        Sn = np.maximum(Pxx_full, np.finfo(float).tiny)
        num = np.sum(X * np.conj(T) / Sn)
        den = np.sum(np.abs(T) ** 2 / Sn)
        Ahat = np.real(num / den)
        Avar = 1.0 / den
        Amp = Ahat
        Amp_se = np.sqrt(Avar)
        amp_method = "matched_filter"
    elif fit_model == "generalized_gaussian":
        x_raw = x + x_mean
        Amp, Amp_se, gfit_params = _fit_generalized_gaussian_amplitude(x_raw)
        if np.isnan(Amp):
            amp_method = "generalized_gaussian_fit_fallback"
        else:
            amp_method = "generalized_gaussian_fit"
    else:
        # Default: non-parametric peak via spectral low-pass filter.
        # FFT the original (pre-window) demeaned signal, zero the noise
        # frequencies, and inverse FFT to get a smoothed signal.  The window
        # used for PSD estimation is not applied here so that the smoothed
        # signal preserves the true peak height.
        amp_method = "peak"

    if amp_method in ("peak", "generalized_gaussian_fit_fallback"):
        X_raw = np.fft.rfft(x, norm="ortho")
        X_lp = X_raw.copy()
        X_lp[kc_full:] = 0.0
        x_lp = np.fft.irfft(X_lp, n=N, norm="ortho")
        peak_val = float(np.max(x_lp)) + x_mean
        margin = max(1, N // 4)
        x_raw = x + x_mean
        baseline = float(np.mean(np.concatenate([x_raw[:margin], x_raw[-margin:]])))
        Amp = peak_val - baseline
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

    diags: dict = {
        "N": N,
        "window_rms": w_rms,
        "cutoff_index": kc_full,
        "dof": nu,
        "welch_nperseg": welch_nperseg,
        "welch_noverlap": welch_noverlap,
        "kept_bins": kept,
        "amplitude_method": amp_method,
    }
    if gfit_params:
        diags["gaussian_fit_params"] = gfit_params
    if return_bandpassed_noise:
        diags["x_bp"] = x_bp

    return CNREstimate(
        cnr=cnr_val,
        cnr_ci95=cnr_ci,
        amplitude=Amp,
        amplitude_se=Amp_se,
        noise_rms=sigma,
        noise_ci95=sigma_ci,
        cutoff_index=kc_full,
        diagnostics=diags,
    )
