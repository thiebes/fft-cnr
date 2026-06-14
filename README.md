# fft-cnr

FFT-based contrast-to-noise ratio estimation from a single frame.

Measure the contrast-to-noise ratio (CNR) of a 1-D signal profile from a
single acquisition. You do not need repeat frames or a separate background
region.

Here CNR means the peak signal amplitude above the baseline, divided by the
RMS of the noise. This is a peak-amplitude signal-to-noise ratio, not a
two-region (difference-of-means) contrast measure.

`fft-cnr` uses the Fourier transform to separate the slowly varying signal from
the rapid, point-to-point noise. The package automatically finds the frequency
boundary between the two (using a model-selection score, the Akaike information
criterion or AIC) and returns a CNR estimate with a 95% confidence interval.

## Installation

Requires Python 3.10 or later (tested on 3.10 through 3.13). If you do not have
Python installed, [download it from python.org](https://www.python.org/downloads/)
and follow the installer instructions (on Windows, check "Add Python to PATH"
when prompted).

Install into a virtual environment so the package and its dependencies stay
isolated from your other projects:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install fft-cnr
```

The only runtime dependencies are numpy (>=1.24) and scipy (>=1.10); pip
installs them automatically.

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

The output is deterministic given the seed, so this snippet reproduces the same
numbers on each run. The package exposes three names: the `fft_cnr` function and
its result types `CNREstimate` and `NoiseModel`.

## How it works

A smooth signal spread over many points concentrates its energy in a few
low-frequency Fourier components. Uncorrelated, point-to-point noise spreads its
energy across all frequencies. So in the power spectrum the signal sits at low
frequency and the noise dominates the high-frequency end, which is where the two
can be separated.

1. The input profile is mean-subtracted and tapered with a window function (a
   tapered cosine, or Tukey, window by default) to suppress edge artifacts in
   the Fourier transform.
2. The power spectrum is estimated by averaging the transforms of overlapping
   segments, which is smoother than a single transform (the averaged-segment, or
   Welch, method). More segments give the noise estimate more degrees of
   freedom, which tightens its confidence interval; this does not by itself
   improve the CNR point estimate.
3. The power spectrum typically shows high power at low frequencies, where the
   signal lives, that levels off to a flat noise floor at higher frequencies.
   The algorithm finds this transition by fitting two line segments in log-log
   space. It then uses the Akaike information criterion (AIC, a standard score
   that penalizes overfitting) to pick the boundary that best balances fit
   quality against model complexity.
4. The spectral bins at and below the boundary are set to zero, the remaining
   high-frequency bins are transformed back to real space, and the RMS of that
   reconstructed noise is taken. This RMS is then scaled to undo the window
   energy and the fraction of bins kept, so it estimates the full-band noise
   level.
5. Signal amplitude is estimated by one of three methods (see below), and
   CNR = amplitude / noise RMS.
6. A 95% confidence interval on CNR is computed by first-order error propagation
   (the delta method), combining the amplitude error and the chi-squared noise
   interval.

The method works best when the signal occupies low frequencies and the noise is
broadband (approximately white). The hard minimum length is 16 points (shorter
inputs raise `ValueError`); reliable knee detection needs more, with measurable
bias below roughly 128 points (see Accuracy). Signals with strong high-frequency
content that overlaps the noise band will also bias the CNR estimate.

## Amplitude estimation

By default (`fit_model=None` or `"peak"`), `fft_cnr` removes the high-frequency
noise components (by zeroing frequencies above the signal/noise boundary) and
reads the peak of the resulting smoothed profile, measured above a baseline
estimated from the outer quarter of the profile on each side. This works for any
profile shape and assumes nothing about its functional form. If the signal
extends into those outer regions, the baseline is contaminated and the amplitude
is underestimated.

Two alternatives are available:

- **Matched filter** (`template` parameter): when a noise-free template of the
  expected signal shape is available, a matched filter weighted by the noise
  spectrum gives the most precise amplitude estimate and standard error. This
  holds only when the template matches the true shape; a mismatched template
  biases the amplitude.

- **Generalized Gaussian fit** (`fit_model="generalized_gaussian"`): fits a
  5-parameter model with a shape exponent that covers profiles from heavy-tailed
  to flat-topped. Useful when fitted parameters (center, width, shape) are
  needed in addition to CNR.

The `result.amplitude_snr` property reports the amplitude divided by its
standard error for whichever estimator ran. A larger value means the amplitude
stands out more clearly above the noise. With a template it is exactly the
matched-filter SNR; with the generalized-Gaussian fit it is the fit's amplitude
SNR; with the peak method it uses a proxy standard error whose calibration is
not characterized. It is NaN when no standard error is available.

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
measurements break this assumption. Shot noise grows with the local signal, so
the noise under the peak is larger than the average noise. Because `cnr` divides
the peak by the average noise, it overestimates the true peak signal-to-noise
ratio, and the size of the error depends on the profile shape.

Setting `estimate_noise_model=True` makes `fft_cnr` test for this. It fits the
photon-transfer relation (variance = gain * signal + read^2) to its own
residual, a single-frame surrogate for a photon-transfer curve that is
conventionally measured from many exposures, and attaches a `NoiseModel` to the
result:

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
appears in `diagnostics["noise_model_skipped"]`). To judge significance, the
package runs the whole estimation pipeline on about 200 simulated data sets, so
the test accounts for the estimator's own artifacts. These extra runs cost
roughly half a second at N=512.

The detection is deterministic by default: the same input always produces the
same result. Pass your own random generator (`rng=np.random.default_rng()`)
to draw an independent simulation instead.

### Correlated (1/f) noise is not corrected

Signal-dependence is one way the constant-noise assumption fails. Spatially
correlated, 1/f-type noise is a separate one. It arises when temporal drift maps
onto a spatial direction, as in galvo raster scanning or streak cameras. The
drift puts noise power at low spatial frequency, where `fft_cnr` reads signal,
so it biases `cnr` high.

The `NoiseModel` has fields for this case (`spectral_exponent`, `white_floor`,
and `correlated`), but they are reserved and not filled in: the two floats stay
NaN and the flag stays None. Single-frame quantitative correction of correlated
noise is not supported, and the reason is an identifiability limit. The signal
and the 1/f noise both live in the low frequencies, and a single profile gives
one measurement, so there is no way to tell how much of the low-frequency power
is signal and how much is drift; any split fits the data equally well.
Separating them needs a second, independent view in which the signal repeats but
the noise does not: multiple frames, an interleaved scan, or a reference
channel. Those are also the way to correct correlated noise once it is detected.

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

The minimum accepted length is 16 points. Reliable results need more; see How it
works for guidance on length.

## Accuracy

Monte Carlo validation (200 trials per condition, `scripts/validate_accuracy.py`)
characterizes bias, precision, and confidence-interval coverage across five
signal shapes: Gaussian, heavy-tailed (generalized Gaussian, p=1.5),
flat-topped (p=4), Gaussian mixture, and Lorentzian. All conditions use additive
white Gaussian noise, the regime the method assumes, so the headline accuracy
numbers inherit that assumption.

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

The 95% confidence intervals contain the true value in 99 to 100% of trials
across all tested conditions. The intervals are conservative, that is, wider
than a true 95% interval, because the degrees of freedom used for the noise
interval are set conservatively, so the chi-squared interval comes out wider
than nominal. The intervals are therefore reliable but not tight.

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
and the fitted parameters recover the truth. On Poisson data the test fired in
all 100 trials at these conditions, the fitted gain is within 5% of the true
value, and the derived peak SNR tracks the true peak signal-to-noise ratio
(ratio 0.99 to 1.03 across N and CNR).

## Background

The spectral decomposition approach used here originated in a study of noise
effects on diffusion coefficient estimation in chemical transport imaging:

> J. J. Thiebes, E. M. Grumstrup, J. Chem. Phys. **160**, 124201 (2024).
> [doi:10.1063/5.0190347](https://doi.org/10.1063/5.0190347)

This package has evolved from the method described in that paper. The
power-spectrum estimation, knee detection, and confidence-interval machinery all
differ from the original.

Support for non-Gaussian profiles was motivated by work on excess kurtosis
in exciton transport:

> E. Arévalo Rodríguez, M. Meléndez, J. Cuadra, F. Prins, J. Phys. Chem. Lett.
> **17**, 2479--2484 (2026). [doi:10.1021/acs.jpclett.5c03961](https://doi.org/10.1021/acs.jpclett.5c03961)

## Acknowledgments

Thanks to Ferry Prins and Enrique Arévalo Rodríguez for discussions on
non-Gaussian transport profiles that informed the design of this package.

## License

Released under the MIT License; see the [LICENSE](LICENSE) file for the full
text.
