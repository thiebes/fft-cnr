# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `roi` parameter to `fft_cnr`: restrict the estimate to a region of interest,
  as explicit `(start, stop)` bounds or `"auto"` (a window sized to the largest
  feature, peak or dip). Windowing removes off-center low-frequency baseline
  structure that would otherwise be counted as signal (issue #7). The resolved
  bounds are reported in `diagnostics["roi"]`.
- Low-frequency baseline guard on the peak and generalized-Gaussian methods:
  `diagnostics["lowfreq_dominated"]` (with `lowfreq_offpeak_ratio`) flags
  profiles where smooth baseline structure away from the peak dominates, so the
  reported `cnr` may reflect baseline power rather than the peak.
- `scripts/validate_iscat_baseline.py`: Monte Carlo validation of the baseline
  guard against an interferometric scattering microscopy (iSCAT) surrogate with
  structured residual background. It documents that broadband structured
  background is correlated noise, out of scope for the single-frame `cnr`, and
  that the matched-filter path (read through `amplitude` / `amplitude_snr`) is
  the recommended estimator in that regime.
- `scripts/validate_amplitude_se.py` and `scripts/validate_cnr_ci_decomposition.py`:
  Monte Carlo sweeps that calibrate the amplitude standard error per path and
  trace it through to the CNR confidence interval.
- Optional `assets` dependency group (matplotlib, curved-text) and
  `assets/hero.py`, which renders the README hero image from a real `fft_cnr`
  run.

### Changed

- Matched-filter standard error is now the exact closed-form error of the
  white-noise projection (windowed data onto the windowed template, noise-only
  weighting), replacing the `Pxx_full`-weighted form whose signal-contaminated
  spectrum overstated the error and widened `cnr_ci95` on the template path.
- `amplitude_snr` is now defined only on the matched-filter (template) path and
  returns NaN on the peak and generalized-Gaussian paths, where the standard
  error is a different quantity that is not comparable to it.
- The peak amplitude read and `roi="auto"` now locate the largest-magnitude
  feature, so a downward (absorption / dark-contrast) feature is found and
  `amplitude` carries its sign. `cnr` is unchanged: it uses the amplitude
  magnitude, and `NoiseModel.peak_snr` does the same.

### Documentation

- README section "Low-frequency baseline and structured background" describing
  the guard, `roi`, the iSCAT scope limit, and the matched-filter recommendation.

## [0.2.1] - 2026-06-14

### Added

- Zenodo DOI badge and a Citation section with BibTeX in the README; the concept
  DOI and the method-foundation references in CITATION.cff.
- Prior-art acknowledgment (Foi et al., 2008) and a "Methods and background"
  paragraph citing the implemented method foundations.

This release carries the citation metadata into the published artifact. Version
0.2.0 was tagged and archived on Zenodo but not published to PyPI.

## [0.2.0] - 2026-06-14

### Added

- Real-space noise-model detector (`estimate_noise_model=True`) that fits the
  photon-transfer relation (`var = gain * signal + read**2`) to its own residual
  and flags signal-dependent (shot) noise. Results are returned in a new
  `NoiseModel` dataclass with a `peak_snr(amplitude)` method; significance is
  calibrated against a pipeline-matched parametric-bootstrap null.
- `amplitude_snr` derived property on `CNREstimate` (amplitude over its standard
  error; equals the matched-filter SNR when a template is supplied).
- Monte Carlo validation sweep for the noise-model detector
  (`scripts/validate_noise_model.py`), measuring false-positive rate and power.
- Weekly CI jobs testing the latest and lowest supported dependency versions.

### Changed

- Documented that single-frame quantitative 1/f (spatially correlated) noise
  correction is unsupported. The spectral `NoiseModel` fields
  (`spectral_exponent`, `white_floor`, `correlated`) are reserved sentinels that
  stay NaN/None; correlated noise is mitigated at acquisition (multiple frames,
  interleaved acquisition, a reference channel).
- Pinned GitHub Actions to commit SHAs and normalized line endings to LF.

## [0.1.1] - 2026-03-22

### Changed

- Set the development status classifier to Beta.

## [0.1.0] - 2026-03-22

### Added

- Initial release: FFT-based CNR estimation from single 1-D profiles, with an
  AIC-selected signal/noise frequency boundary and a 95% confidence interval.

[Unreleased]: https://github.com/thiebes/fft-cnr/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/thiebes/fft-cnr/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/thiebes/fft-cnr/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/thiebes/fft-cnr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/thiebes/fft-cnr/releases/tag/v0.1.0
