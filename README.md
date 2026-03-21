# fft-cnr: FFT-based Contrast-to-Noise Ratio Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small, focused Python library for estimating Contrast-to-Noise Ratio (CNR) from signal profiles using Fast Fourier Transform (FFT) analysis.

## What is FFT-based CNR Estimation?

Contrast-to-Noise Ratio (CNR) is a measure of signal quality that quantifies the ratio between the meaningful signal content and the background noise. This package uses FFT analysis to separate signal from noise in the frequency domain:

1. **Transform to frequency domain**: Apply FFT to convert the signal from time/space domain to frequency domain
2. **Identify signal components**: Locate the dominant frequency peaks that represent the actual signal
3. **Isolate noise components**: Identify high-frequency components beyond the signal peaks as noise
4. **Calculate CNR**: Compute the ratio between signal strength and noise level

This approach is particularly useful for:
- Quality assessment of periodic signals
- Noise characterization in measurement data
- Signal preprocessing and filtering parameter selection
- Comparative analysis of different signal acquisition methods

## Installation

### From PyPI (once published)

```bash
pip install fft-cnr
```

### From source

```bash
git clone https://github.com/yourusername/fft-cnr.git
cd fft-cnr
pip install -e .
```

### Development installation

```bash
git clone https://github.com/yourusername/fft-cnr.git
cd fft-cnr
pip install -e ".[dev]"
```

Or using the requirements files:

```bash
pip install -r requirements-dev.txt
```

## Quick Start

```python
import numpy as np
from fft_cnr import estimate_cnr

# Create a noisy sinusoidal signal
t = np.linspace(0, 1, 100)
clean_signal = np.sin(2 * np.pi * 5 * t)
noise = 0.1 * np.random.randn(100)
noisy_signal = clean_signal + noise

# Estimate CNR
cnr = estimate_cnr(noisy_signal)
print(f"CNR: {cnr}")
```

## Usage Examples

### Example 1: Comparing Signal Quality

```python
import numpy as np
from fft_cnr import estimate_cnr

# Generate signals with different noise levels
t = np.linspace(0, 1, 200)
signal = np.sin(2 * np.pi * 5 * t)

# Low noise
low_noise = signal + 0.05 * np.random.randn(len(t))
cnr_low = estimate_cnr(low_noise)

# High noise
high_noise = signal + 0.3 * np.random.randn(len(t))
cnr_high = estimate_cnr(high_noise)

print(f"Low noise CNR: {cnr_low}")   # Higher value
print(f"High noise CNR: {cnr_high}") # Lower value
```

### Example 2: Multi-frequency Signal

```python
import numpy as np
from fft_cnr import estimate_cnr

# Create a complex signal with multiple frequency components
t = np.linspace(0, 2, 300)
signal = (np.sin(2 * np.pi * 3 * t) +      # Fundamental
          0.5 * np.sin(2 * np.pi * 6 * t) + # First harmonic
          0.25 * np.sin(2 * np.pi * 9 * t)) # Second harmonic

# Add noise
noisy_signal = signal + 0.1 * np.random.randn(len(t))

# Estimate CNR
cnr = estimate_cnr(noisy_signal)
print(f"CNR of composite signal: {cnr}")
```

### Example 3: Handling Edge Cases

```python
import numpy as np
from fft_cnr import estimate_cnr

# The function handles various edge cases:

# 1. Too short - raises ValueError
try:
    estimate_cnr(np.array([1, 2]))
except ValueError as e:
    print(f"Error: {e}")

# 2. Constant signal - raises ValueError
try:
    estimate_cnr(np.ones(100))
except ValueError as e:
    print(f"Error: {e}")

# 3. Very clean signal - returns high CNR
clean = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
print(f"Clean signal CNR: {estimate_cnr(clean)}")

# 4. List input (automatically converted)
signal_list = [np.sin(2 * np.pi * 5 * i/100) for i in range(100)]
print(f"From list CNR: {estimate_cnr(signal_list)}")
```

## API Reference

### `estimate_cnr(noisy_profile: np.ndarray) -> float`

Estimate the Contrast-to-Noise Ratio from a signal profile using FFT analysis.

**Parameters:**
- `noisy_profile` (np.ndarray): 1D array containing the signal profile to analyze. Can also accept Python lists, which will be converted to numpy arrays.

**Returns:**
- `float`: The estimated CNR value, rounded to 2 decimal places. Returns `float('inf')` if noise level is zero.

**Raises:**
- `ValueError`: If the profile has fewer than 3 points
- `ValueError`: If the profile is constant (all values are the same)

**Algorithm Details:**
1. Normalize the profile to unit amplitude (divide by max absolute value)
2. Compute orthogonally normalized single-sided FFT using `np.fft.rfft(norm='ortho')`
3. Find peaks and minima in the FFT modulus using `scipy.signal.find_peaks`
4. Identify the first local minimum after the dominant signal peak
5. Calculate noise as RMS of high-frequency components from that minimum onward
6. Return CNR = 1 / noise_level, rounded to 2 decimal places

**Edge Case Handling:**
- Profiles with < 3 points: Raises `ValueError`
- Constant profiles: Raises `ValueError`
- No clear peak found: Uses half the spectrum as noise
- No minimum after peak: Uses point immediately after peak
- Empty noise regime: Uses last quarter of spectrum
- Zero noise estimate: Returns `float('inf')`

## Running Tests

The package includes comprehensive unit tests covering normal operation and edge cases.

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=fft_cnr --cov-report=html

# Run specific test file
pytest tests/test_estimator.py

# Run with verbose output
pytest -v
```

## Validation and Characterization

The package includes a comprehensive validation framework to assess the accuracy and precision of the CNR estimator across different signal-to-noise levels.

### Quick Characterization Example

```python
from fft_cnr.characterization import run_characterization_study, plot_validation_summary

# Run a characterization study across CNR levels from 1 to 100
results = run_characterization_study(
    cnr_range=(1.0, 100.0),
    num_cnr_levels=20,
    num_trials_per_level=100,
    verbose=True
)

# View results
print(results.head())

# Create validation plots
fig = plot_validation_summary(results, output_path='cnr_validation.png')
```

### Understanding Validation Results

The characterization framework provides:

- **Mean ratio**: Average of CNR_est / CNR_nom (should be ~1.0 for unbiased estimator)
- **Median ratio**: Median of CNR_est / CNR_nom (robust to outliers)
- **Mode ratio**: Most common ratio value
- **Std ratio**: Standard deviation (measures precision)
- **Bias**: Mean(CNR_est - CNR_nom) (systematic error)
- **RMSE**: Root mean squared error (overall accuracy)
- **Percentiles**: 5th and 95th percentiles (confidence intervals)

### Advanced Characterization

```python
from fft_cnr.characterization import (
    generate_gaussian_profile,
    add_calibrated_noise,
    characterize_cnr_level,
    plot_bias_and_rmse
)

# Generate a custom signal
signal = generate_gaussian_profile(
    center=0.5,
    sigma=0.1,
    amplitude=1.0,
    num_points=200
)

# Add calibrated noise at a specific CNR level
noisy_signal = add_calibrated_noise(signal, cnr_nominal=15.0, rng_seed=42)

# Characterize estimator performance at one CNR level
level_results = characterize_cnr_level(
    cnr_nominal=15.0,
    num_trials=100,
    num_points=200,
    rng_seed=42
)

print(f"Mean ratio: {level_results['mean_ratio']:.3f}")
print(f"Std ratio: {level_results['std_ratio']:.3f}")
print(f"Bias: {level_results['bias']:.3f}")

# Create bias and RMSE plots
fig = plot_bias_and_rmse(results, output_path='bias_rmse.png')
```

### Saving Results for Further Analysis

```python
# Run study
results = run_characterization_study(
    cnr_range=(5.0, 50.0),
    num_cnr_levels=15,
    num_trials_per_level=200
)

# Save results to CSV
results.to_csv('cnr_characterization_results.csv', index=False)

# Load and analyze later
import pandas as pd
results = pd.read_csv('cnr_characterization_results.csv')
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0

Development requirements:
- pytest >= 7.0.0

## Project Structure

```
fft-cnr/
├── fft_cnr/                    # Main package
│   ├── __init__.py             # Package interface
│   ├── version.py              # Version information
│   ├── estimator.py            # Core CNR estimation algorithm
│   └── characterization.py     # Validation and characterization tools
├── tests/                      # Unit tests
│   ├── test_estimator.py       # Core estimator tests
│   └── test_characterization.py # Characterization module tests
├── pyproject.toml              # Modern Python packaging config
├── setup.py                    # Package setup
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore patterns
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this package in your research, please cite:

```bibtex
@software{fft_cnr,
  author = {Thiebes, Joseph J.},
  title = {fft-cnr: FFT-based Contrast-to-Noise Ratio Estimation},
  year = {2025},
  url = {https://github.com/yourusername/fft-cnr},
  note = {Developed as part of the DICE project}
}
```

This software was developed as part of the **DICE (Domain-Informed Contrast Enhancement)** project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Developed as part of the DICE project
- Built with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)

## Contact

Joseph J. Thiebes - your.email@example.com

Project Link: [https://github.com/yourusername/fft-cnr](https://github.com/yourusername/fft-cnr)
