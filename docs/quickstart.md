# Quick start

Estimate the CNR of a noisy 1-D profile in a few lines:

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

The output is deterministic given the seed, so this snippet reproduces the same
numbers on each run:

```text
CNR:       9.8
CNR 95%CI: (6.9, 12.7)
Amplitude: 9.74
Noise RMS: 0.991
```

## Amplitude estimation

By default `fft_cnr` reads the peak of the smoothed profile above an
edge-estimated baseline, which works for any profile shape. Two alternatives are
available through {func}`~fft_cnr.fft_cnr`:

```python
# With a known noise-free template (matched filter, most precise)
result = fft_cnr(noisy, template=signal)

# With a generalized Gaussian fit (when shape parameters are also wanted)
result = fft_cnr(noisy, fit_model="generalized_gaussian")
print(result.diagnostics["gaussian_fit_params"])
```

## Noise model detection

Setting `estimate_noise_model=True` tests whether the noise grows with the
signal (shot noise) and attaches a {class}`~fft_cnr.NoiseModel` to the result:

```python
result = fft_cnr(noisy, estimate_noise_model=True)
model = result.noise_model

if model.signal_dependent:
    # Noise grows with the signal; result.cnr overestimates the peak SNR.
    print(f"Peak SNR: {model.peak_snr(result.amplitude):.1f}")
```

For the full set of parameters, return fields, accuracy characterization, and the
treatment of structured background, see the
[README](https://github.com/thiebes/fft-cnr#readme) and the {doc}`api`.
