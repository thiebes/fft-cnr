# fft-cnr

FFT-based contrast-to-noise ratio estimation from a single frame.

Measure the contrast-to-noise ratio (CNR) of a 1-D signal profile from a
single acquisition—no repeat frames or separate background region needed.

`fft-cnr` uses the Fourier transform to separate slowly-varying signal features
from rapid point-to-point noise fluctuations. An automatic model-selection
criterion (AIC) identifies the frequency boundary between the two, and the
package returns a CNR estimate with a 95% confidence interval.

## Installation

Requires Python 3.10 or later. If you don't have Python installed,
[download it from python.org](https://www.python.org/downloads/) and follow
the installer instructions (on Windows, check "Add Python to PATH" when
prompted).

Then run this command in a terminal (Command Prompt or PowerShell on Windows,
Terminal on macOS/Linux):

```bash
pip install fft-cnr
```

## Quick start

```python
import numpy as np
from fft_cnr import fft_cnr

# Simulate a 1-D intensity profile (e.g., from a microscopy line scan)
# with additive detector noise
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

1. The input profile is mean-subtracted and tapered with a window function
   (Tukey by default) to suppress edge artifacts in the Fourier transform.
2. The power spectrum is estimated by averaging FFTs of overlapping segments,
   which produces a smoother estimate than a single FFT (Welch's method). More
   segments yield tighter confidence intervals.
3. The power spectrum typically shows high power at low frequencies (signal)
   that levels off to a flat noise floor at higher frequencies. The algorithm
   finds this transition by fitting two line segments in log-log space and
   using the Akaike information criterion (AIC, a standard metric that
   penalizes overfitting) to select the boundary that best balances fit
   quality against model complexity.
4. Noise RMS is computed from the inverse FFT of the frequencies above the
   signal/noise boundary.
5. Signal amplitude is estimated by one of three methods (see below), and
   CNR = amplitude / noise RMS.
6. A 95% confidence interval on CNR is computed by error propagation (delta
   method), combining uncertainty from the amplitude estimate and the
   chi-squared noise interval.

The method works best when the signal occupies low frequencies and the noise
is broadband (approximately white). Profiles shorter than about 64 points
provide too few spectral bins for reliable knee detection, and signals with
significant high-frequency content that overlaps the noise band will bias the
CNR estimate.

## Amplitude estimation

By default (`fit_model=None` or `"peak"`), `fft_cnr` removes the high-frequency
noise components (by zeroing frequencies above the signal/noise boundary) and
reads the peak of the resulting smoothed profile. This is robust across
arbitrary profile shapes and requires no assumptions about the functional form.

Two alternatives are available:

- **Matched filter** (`template` parameter): when a noise-free template of the
  expected signal shape is available, a matched filter weighted by the noise
  spectrum provides the most precise amplitude estimate and standard error.

- **Generalized Gaussian fit** (`fit_model="generalized_gaussian"`): fits a
  5-parameter model with a shape exponent that accommodates profiles ranging
  from heavy-tailed to flat-topped. Useful when fitted parameters (center,
  width, shape) are needed in addition to CNR.

The `result.amplitude_snr` property reports the amplitude divided by its
standard error for whichever estimator ran, a measure of how detectable the
amplitude is. With a template it is exactly the matched-filter SNR; with the
generalized-Gaussian fit it is the fit's amplitude SNR; with the peak method it
uses a proxy standard error whose calibration is not characterized. It is NaN
when no standard error is available.

```python
# With a known template
result = fft_cnr(noisy, template=signal)

# With a generalized Gaussian fit
result = fft_cnr(noisy, fit_model="generalized_gaussian")
print(result.diagnostics["gaussian_fit_params"])
```

## Noise model detection

The CNR above treats the noise as one number, which is correct when the noise
is the same at every point (for example, detector read noise). Photon-counting
measurements violate this: shot noise grows with the local signal, so the
noise under the peak is larger than the average noise, and `cnr` (peak over
average noise) overestimates the peak signal-to-noise ratio by a
profile-dependent factor.

Setting `estimate_noise_model=True` makes `fft_cnr` test for this. It fits
the photon-transfer relation (variance = gain x signal + read^2) to its own
residual and attaches a `NoiseModel` to the result:

```python
result = fft_cnr(noisy, estimate_noise_model=True)
model = result.noise_model

if model.signal_dependent:
    # Noise grows with the signal; result.cnr overestimates the peak SNR.
    # The fitted noise model gives the corrected value:
    print(f"Peak SNR: {model.peak_snr(result.amplitude):.1f}")
```

`model.signal_dependent` is `True` when the fitted gain is statistically
significant, `False` when the noise is consistent with a constant level, and
`None` when the profile's signal range is too small to test (the reason
appears in `diagnostics["noise_model_skipped"]`). Significance is calibrated
by simulation through the same estimation pipeline, so the test accounts for
the estimator's own artifacts; the simulation adds about 200 internal
re-runs, so the option costs roughly half a second at N=512.

The detection is deterministic by default: the same input always produces the
same result. Pass your own random generator (`rng=np.random.default_rng()`)
to draw an independent simulation instead.

### Correlated (1/f) noise is not corrected

Signal-dependence is one way the constant-noise assumption fails; spatially
correlated, 1/f-type noise is the orthogonal one. It arises when temporal drift
maps onto a spatial axis (galvo raster scanning, streak cameras), placing noise
power at low spatial frequency where `fft_cnr` reads signal, which biases `cnr`
high. The `NoiseModel` fields `spectral_exponent`, `white_floor`, and
`correlated` name this axis, but they are reserved and stay NaN/None: single-frame
quantitative correction of correlated noise is not supported. An estimated signal
shape leaves low-frequency model error that one frame cannot distinguish from 1/f
noise, so the exponent and floor cannot be recovered without bias. Characterize
and correct correlated noise with multiple frames, interleaved acquisition, or a
reference channel.

## Return value

`fft_cnr` returns a `CNREstimate` dataclass. The first five fields are the
primary outputs; `cutoff_index`, `diagnostics`, and `noise_model` are for
advanced inspection.

| Field | Type | Description |
| -------------- | ---------------------- | --------------------------------------------------- |
| `cnr` | `float` | Estimated contrast-to-noise ratio |
| `cnr_ci95` | `tuple[float, float]` | 95% confidence interval on CNR |
| `amplitude` | `float` | Signal amplitude estimate |
| `amplitude_se` | `float` | Standard error of the amplitude estimate |
| `amplitude_snr` | `float` (property) | Amplitude over its standard error; equals the matched-filter SNR with a template; NaN when no standard error is available |
| `noise_rms` | `float` | RMS noise from the high-frequency spectral region |
| `noise_ci95` | `tuple[float, float]` | 95% confidence interval on noise RMS |
| `cutoff_index` | `int` | Spectral index of the signal/noise boundary |
| `diagnostics` | `dict` | Welch parameters, DOF, amplitude method, fit params |
| `noise_model` | `NoiseModel \| None` | Fitted noise structure (`None` unless requested) |

## Parameters

Most users will only need `x` (and optionally `template` or `fit_model`). The
remaining parameters control internal details of the spectral estimation and
rarely need adjustment.

| Parameter | Default | Description |
| ---------------------- | --------------- | --------------------------------------------------------- |
| `x` | (required) | 1-D signal array (length >= 16) |
| `template` | `None` | Noise-free template for matched-filter estimation |
| `fit_model` | `None` | Amplitude method: `None`/`"peak"` for spectral low-pass, `"generalized_gaussian"` for parametric fit |
| `window` | `"tukey"` | Taper window: `"tukey"`, `"hann"`, or `"none"` |
| `tukey_alpha` | `0.25` | Tukey window shape parameter |
| `welch_nperseg` | `None` | Welch segment length (defaults to `max(16, N//8)`) |
| `welch_noverlap` | `None` | Welch overlap (defaults to `nperseg // 2`) |
| `cutoff_guard` | `(0.05, 0.5)` | Fractional frequency bounds for signal/noise boundary search |
| `fallback_cut_frac` | `0.25` | Fallback signal/noise boundary if AIC selection fails |
| `return_bandpassed_noise` | `False` | Include the bandpassed noise array in diagnostics |
| `estimate_noise_model` | `False` | Fit and test the noise model (see Noise model detection) |
| `rng` | `None` | Generator for the noise-model test; `None` is deterministic |

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
across all tested conditions. The intervals are conservative—wider than
the nominal 95%—because the chi-squared noise model overestimates
uncertainty. This means the intervals are reliable but not tight.

### Noise model detection

Monte Carlo validation of the signal-dependence test (100 trials per
condition, `scripts/validate_noise_model.py`) covers white noise, where the
test should stay quiet, and Poisson counts, where it should fire. Conditions
span Gaussian and flat-topped profiles, N in {256, 512}, and CNR 5--20; no
trial was skipped for insufficient signal range.

| Regime | Flag rate | Median gain | Median read |
| --------------------------------------- | ------------------ | ---------------- | ----------- |
| White noise (truth: gain 0, read 1) | 2--7% (nominal 5%) | -0.004 to +0.007 | 0.99--1.00 |
| Poisson counts (truth: gain 1, read 0) | 100% | 0.95--1.02 | -- |

On white noise the false-positive rate is consistent with the 5% test level
and the fitted parameters recover the truth. On Poisson data the test fires
in every trial, the fitted gain is within 5% of the true value, and the
derived peak SNR tracks the true peak signal-to-noise ratio (ratio
0.99--1.03 across N and CNR).

## Background

The spectral decomposition approach used here originated in a study of noise
effects on diffusion coefficient estimation in chemical transport imaging:

> J. J. Thiebes, E. M. Grumstrup, J. Chem. Phys. **160**, 124201 (2024).
> [doi:10.1063/5.0190347](https://doi.org/10.1063/5.0190347)

The implementation in this package has evolved from the method described in
that paper—the PSD estimation, knee detection, and confidence interval
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
