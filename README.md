# fft-cnr

FFT-based contrast-to-noise ratio estimation from a single frame.

Measure the contrast-to-noise ratio (CNR) of a 1-D signal profile from a
single acquisition---no repeat frames or separate background region needed.

`fft-cnr` decomposes the signal into low-frequency (signal) and high-frequency
(noise) components using a unitary FFT, locates the spectral boundary between
the two via an information-theoretic criterion (AIC), and returns a CNR
estimate with a 95% confidence interval.

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

```text
CNR:       9.8
CNR 95%CI: (6.9, 12.7)
Amplitude: 9.74
Noise RMS: 0.991
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

By default (`fit_model=None` or `"peak"`), `fft_cnr` applies a spectral
low-pass filter (zeroing frequencies above the knee) and reads the peak of the
smoothed signal. This is robust across arbitrary profile shapes and requires no
assumptions about the functional form.

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
| `fit_model` | `None` | Amplitude method: `None`/`"peak"` for spectral low-pass, `"generalized_gaussian"` for parametric fit |
| `window` | `"tukey"` | Taper window: `"tukey"`, `"hann"`, or `"none"` |
| `tukey_alpha` | `0.25` | Tukey window shape parameter |
| `welch_nperseg` | `None` | Welch segment length (defaults to `max(16, N//8)`) |
| `welch_noverlap` | `None` | Welch overlap (defaults to `nperseg // 2`) |
| `cutoff_guard` | `(0.05, 0.5)` | Fractional bounds for AIC knee search |
| `fallback_cut_frac` | `0.25` | Fallback knee position if AIC selection fails |
| `return_bandpassed_noise` | `False` | Include the bandpassed noise array in diagnostics |

## Accuracy

Monte Carlo validation (200 trials per condition, `scripts/validate_accuracy.py`)
characterizes bias, precision, and confidence interval coverage across five
signal shapes: Gaussian, heavy-tailed (generalized Gaussian, p=1.5),
flat-topped (p=4), Gaussian mixture, and Lorentzian.

### Bias

Median estimated-to-true CNR ratio at N=512, by amplitude method:

| Shape | Peak | Gen. Gaussian | Matched filter |
| ----------------------- | ----------- | ------------- | -------------- |
| Gaussian | 1.00--1.03 | 1.00--1.01 | 1.00--1.01 |
| Heavy-tailed (p=1.5) | 0.97--1.00 | 0.99--1.01 | 0.98--1.00 |
| Flat-topped (p=4) | 0.93--1.10 | 0.93--1.01 | 0.92--1.00 |
| Gaussian mixture | 0.99--1.01 | 1.12--1.16 | 0.99--1.00 |
| Lorentzian | 0.98--1.01 | 1.06--1.09 | 0.99--1.00 |

The peak method stays within 3% of the true CNR for most shapes from CNR=5
upward. The main exception is flat-topped profiles, where the spectral
low-pass filter rounds off the flat peak (up to 7% positive bias at CNR=3,
7% negative at CNR=100). The generalized Gaussian fit is accurate when the
signal is in the model family but introduces 6--16% positive bias on shapes
outside it (Gaussian mixture, Lorentzian). The matched filter tracks the true
CNR to within 2% when the template matches the signal shape.

### Profile length

Shorter profiles reduce accuracy slightly. At N=128 (peak method, Gaussian
signal), the estimator shows 6--8% negative bias at high CNR because the
Welch PSD has fewer segments for knee detection.

| N | CNR=5 | CNR=20 | CNR=100 |
| ------- | --------- | --------- | --------- |
| 128 | 1.03 | 0.94 | 0.93 |
| 256 | 1.03 | 1.00 | 1.00 |
| 512 | 1.01 | 1.01 | 1.00 |

### Precision

Trial-to-trial scatter (relative standard deviation) scales as roughly
1/sqrt(N): approximately 11% at N=128, 7% at N=256, and 5% at N=512 (peak
method, CNR=5). Providing a matched template reduces scatter at low CNR.

### Confidence intervals

The 95% confidence intervals contain the true value in 99--100% of trials
across all tested conditions. The intervals are conservative---wider than
the nominal 95%---because the chi-squared noise model overestimates
uncertainty. This means the intervals are reliable but not tight.

## Background

The spectral decomposition approach used here originated in a study of noise
effects on diffusion coefficient estimation in chemical transport imaging:

> J. Thiebes, J. Chem. Phys. **160**, 124201 (2024).
> [doi:10.1063/5.0190515](https://doi.org/10.1063/5.0190515)

The implementation in this package has evolved from the method described in
that paper---the PSD estimation, knee detection, and confidence interval
machinery differ from the original.

Support for non-Gaussian profiles was motivated by work on excess kurtosis
in exciton transport:

> E. Arévalo Rodríguez, M. Meléndez, J. Cuadra, F. Prins, J. Phys. Chem. Lett.
> **17**, 2479--2484 (2026). [doi:10.1021/acs.jpclett.5c03961](https://doi.org/10.1021/acs.jpclett.5c03961)

## Acknowledgments

Thanks to Ferry Prins and Enrique Arévalo Rodríguez for discussions on
non-Gaussian transport profiles that informed the design of this package.

## License

MIT
