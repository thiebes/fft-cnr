# fft-cnr

FFT-based contrast-to-noise ratio estimation from a single frame.

`fft-cnr` measures the contrast-to-noise ratio (CNR) of a 1-D signal profile
from a single acquisition, with no repeat frames and no separate background
region. Here CNR means the peak signal amplitude above the baseline divided by
the RMS of the noise. The contrast is the peak's height above the baseline, the
part that stands out from the background, so dividing it by the noise gives the
ratio that governs whether the feature can be detected and measured. This is a
single-peak contrast, distinct from the two-region (difference-of-means) CNR
measured with two hand-drawn regions in MRI or CT.

The input is a one-dimensional profile: a single line scan, a spectrum, or one
row or column of an image, not a two-dimensional image.

The package uses the Fourier transform to separate the slowly varying signal
from the rapid, point-to-point noise. It finds the frequency boundary between
the two automatically (using the Akaike information criterion) and returns a CNR
estimate with a 95% confidence interval.

## Why fft-cnr

Measuring a contrast-to-noise ratio usually needs either repeat acquisitions, to
estimate the noise from frame-to-frame variation, or a separate background region
known to contain no signal. Single-frame profile measurements in microscopy and
spectroscopy often have neither: there is one line scan or one spectrum, and the
signal may fill the field of view. Common practice is then to report a
signal-to-noise number from a hand-chosen region, with no stated uncertainty.

`fft-cnr` estimates the noise from the same single profile that carries the
signal. A smooth signal concentrates its power at low frequency, while
uncorrelated point-to-point noise spreads across all frequencies, so the
high-frequency end of the power spectrum measures the noise directly. The package
finds the signal/noise frequency boundary automatically and returns a CNR with a
95% confidence interval, so the result carries a stated uncertainty rather than a
single ad hoc number.

Typical uses are transport and diffusion imaging (line scans of a spreading
population) and broadband or pump-probe spectroscopy (a peak on a noisy
spectrum), where the contrast of one profile must be quantified from a single
acquisition.

## When it applies

`fft-cnr` assumes the noise is broadband and approximately white. It is most
accurate when the signal occupies low frequencies, the noise is uncorrelated
point-to-point, and the profile has at least about 256 points (the hard minimum
is 16, and accuracy degrades below roughly 128).

It is biased when those conditions do not hold:

- Strong correlated or 1/f noise puts noise power at low frequency, where the
  method reads signal, and biases the CNR high. The package can detect this but
  does not correct it from a single frame.
- Broadband structured background, such as the residual speckle left after
  interferometric scattering microscopy (iSCAT) background subtraction, is
  correlated noise and out of scope for the single-frame `cnr`. In that regime,
  supply the expected signal shape as a `template` and read `amplitude` and
  `amplitude_snr` instead.

The [README](https://github.com/thiebes/fft-cnr#accuracy) reports the quantified
bias, precision, and confidence-interval coverage from the Monte Carlo
validation.

The public surface is small: the {func}`~fft_cnr.fft_cnr` function and its result
types {class}`~fft_cnr.CNREstimate` and {class}`~fft_cnr.NoiseModel`.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
api
```

## Project links

- [Source repository](https://github.com/thiebes/fft-cnr)
- [Issue tracker](https://github.com/thiebes/fft-cnr/issues)
- [Changelog](https://github.com/thiebes/fft-cnr/blob/main/CHANGELOG.md)
- [Contributing guide](https://github.com/thiebes/fft-cnr/blob/main/CONTRIBUTING.md)
