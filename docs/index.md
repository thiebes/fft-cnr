# fft-cnr

FFT-based contrast-to-noise ratio estimation from a single frame.

`fft-cnr` measures the contrast-to-noise ratio (CNR) of a 1-D signal profile
from a single acquisition, with no repeat frames and no separate background
region. Here CNR means the peak signal amplitude above the baseline divided by
the RMS of the noise: a peak-amplitude signal-to-noise ratio, not a two-region
contrast measure.

The package uses the Fourier transform to separate the slowly varying signal
from the rapid, point-to-point noise. It finds the frequency boundary between
the two automatically (using the Akaike information criterion) and returns a CNR
estimate with a 95% confidence interval.

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
