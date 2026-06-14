# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.1]: https://github.com/thiebes/fft-cnr/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/thiebes/fft-cnr/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/thiebes/fft-cnr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/thiebes/fft-cnr/releases/tag/v0.1.0
