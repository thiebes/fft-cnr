# fft-cnr

FFT-based contrast-to-noise ratio estimation from a single frame.

`fft-cnr` measures the contrast-to-noise ratio (CNR) of a 1-D signal profile
without requiring multiple acquisitions or a separate background region. It
decomposes the signal into low-frequency (signal) and high-frequency (noise)
components using a unitary FFT, locates the spectral boundary between the two
via an information-theoretic criterion (AIC), and returns a CNR estimate with
a 95% confidence interval.

## Installation

```bash
pip install fft-cnr
```

## Quick start

```python
import numpy as np
from fft_cnr import fft_cnr

# Simulate a Gaussian peak with additive white noise
rng = np.random.default_rng(0)
x = np.arange(256, dtype=float)
signal = 10.0 * np.exp(-0.5 * ((x - 127) / 20) ** 2)
noisy = signal + rng.normal(0, 1.0, 256)

result = fft_cnr(noisy)

print(f"CNR:       {result.cnr:.1f}")
print(f"CNR 95%CI: ({result.cnr_ci95[0]:.1f}, {result.cnr_ci95[1]:.1f})")
print(f"Amplitude: {result.amplitude:.2f}")
print(f"Noise RMS: {result.noise_rms:.3f}")
```

## How it works

1. The input profile is demeaned and tapered (Tukey window by default).
2. A Welch periodogram estimates the power spectral density, with
   degree-of-freedom tracking for downstream confidence intervals.
3. A two-segment least-squares fit in log-log space selects the spectral
   knee that separates signal power from the noise floor, using AIC to
   balance goodness of fit against model complexity.
4. Noise RMS is computed from the inverse FFT of the above-knee frequencies,
   corrected for window taper and the fraction of retained bins.
5. Signal amplitude is estimated by one of three methods (see below), and
   CNR = amplitude / noise RMS.
6. A 95% confidence interval on CNR is computed via the delta method,
   propagating uncertainty from both the amplitude estimate and the
   chi-squared noise interval.

## Amplitude estimation

By default, `fft_cnr` uses a non-parametric **peak** method that applies a
spectral low-pass filter (zeroing frequencies above the knee) and reads the
peak of the smoothed signal. This is robust across arbitrary profile shapes
and requires no assumptions about the functional form.

Two alternatives are available:

- **Matched filter** (`template` parameter): when a noise-free template of the
  expected signal shape is available, a whitened matched filter provides the
  most precise amplitude estimate and standard error.

- **Generalized Gaussian fit** (`fit_model="generalized_gaussian"`): fits a
  5-parameter model with a shape exponent that accommodates profiles ranging
  from heavy-tailed to flat-topped. Useful when fitted parameters (center,
  width, shape) are needed in addition to CNR.

```python
# With a known template
result = fft_cnr(noisy, template=signal)

# With a generalized Gaussian fit
result = fft_cnr(noisy, fit_model="generalized_gaussian")
print(result.diagnostics["gaussian_fit_params"])
```

## Return value

`fft_cnr` returns a `CNREstimate` dataclass:

| Field | Type | Description |
| -------------- | ---------------------- | --------------------------------------------------- |
| `cnr` | `float` | Estimated contrast-to-noise ratio |
| `cnr_ci95` | `tuple[float, float]` | 95% confidence interval on CNR |
| `amplitude` | `float` | Signal amplitude estimate |
| `amplitude_se` | `float` | Standard error of the amplitude estimate |
| `noise_rms` | `float` | RMS noise from the high-frequency spectral region |
| `noise_ci95` | `tuple[float, float]` | 95% confidence interval on noise RMS |
| `cutoff_index` | `int` | Spectral index of the signal/noise boundary |
| `diagnostics` | `dict` | Welch parameters, DOF, amplitude method, fit params |

## Parameters

| Parameter | Default | Description |
| ---------------------- | --------------- | --------------------------------------------------------- |
| `x` | (required) | 1-D signal array (length >= 16) |
| `template` | `None` | Noise-free template for matched-filter estimation |
| `fit_model` | `None` | `"peak"` (default behavior) or `"generalized_gaussian"` |
| `window` | `"tukey"` | Taper window: `"tukey"`, `"hann"`, or `"none"` |
| `tukey_alpha` | `0.25` | Tukey window shape parameter |
| `welch_nperseg` | `None` | Welch segment length (defaults to `max(16, N//8)`) |
| `welch_noverlap` | `None` | Welch overlap (defaults to `nperseg // 2`) |
| `cutoff_guard` | `(0.05, 0.5)` | Fractional bounds for AIC knee search |
| `fallback_cut_frac` | `0.25` | Fallback knee position if AIC selection fails |
| `return_bandpassed_noise` | `False` | Include the bandpassed noise array in diagnostics |

## License

MIT
