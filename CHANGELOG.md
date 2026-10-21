# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CI job that builds the package on pull requests and checks it with the same
  twine and packaging versions as the PyPI publish action.
- Dependabot weekly updates for the SHA-pinned GitHub Actions, with a seven-day
  cooldown and the publish action proposed on its own.

### Changed

- `diagnostics["gaussian_fit_rejected"]` is renamed
  `diagnostics["gaussian_fit_rejection_reason"]`. The value is a string naming
  the reason, and the old name read like the boolean flags beside it
  (`lowfreq_dominated`, `amplitude_sign_ambiguous`). Code that reads the old
  key must use the new name.
- The source distribution lists its contents explicitly: the package, tests,
  docs, validation scripts, `README.md`, `LICENSE`, `CHANGELOG.md`, and
  `CITATION.cff`, plus `pyproject.toml` and `.gitignore`, which the build
  backend always includes. Repository-only files are no longer included (`assets/`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `uv.lock`, `.github/`,
  `.readthedocs.yaml`, `.gitattributes`), nor are files that are ignored only
  locally. The wheel is unchanged.
- The test, docs, and compatibility workflow tokens are restricted to
  read-only contents.

### Fixed

- The generalized-Gaussian path also refuses a fit whose center is pinned at
  its bound, half a profile length outside the data
  (`gaussian_fit_rejection_reason == "center_at_bound"`). Such a fit describes
  only a broad wing reaching into the profile, usually with the wrong sign, and
  on noise-dominated profiles it could read a CNR near 10. A center just
  outside the profile is still accepted, because a genuine feature can sit at
  the edge of the data. In validation (Gaussian, heavy-tailed, flat-topped,
  Gaussian-mixture, and Lorentzian profiles; N=241 and 512; CNR 0.1 to 20;
  400 profiles per condition), the rule refused no fit within 50% of the true
  amplitude.

## [0.3.3] - 2026-10-20

### Changed

- Pinned the build backend (`hatchling==1.32.4`) and updated the PyPI publish
  action to v1.14.2, whose Twine accepts the metadata version 2.5 that current
  hatchling writes.

This release carries the 0.3.2 changes to PyPI. Version 0.3.2 was tagged and
archived on Zenodo but not published to PyPI, because the publish action
rejected its metadata. The package code is unchanged from 0.3.2.

## [0.3.2] - 2026-10-20

### Added

- `diagnostics["gaussian_fit_rejected"]` on the generalized-Gaussian path:
  `"amplitude_exceeds_data_range"` or `"sigma_at_upper_bound"` when a
  degenerate fit is refused, absent otherwise.

### Fixed

- The generalized-Gaussian path refuses two kinds of degenerate fit instead of
  reporting them. On noise-dominated profiles the fit could run away, for
  example to a flat-topped model as wide as the profile whose amplitude and
  baseline cancel, and report a CNR of tens to thousands. A fit whose amplitude
  exceeds twice the data's peak-to-peak range, or whose width is pinned at the
  profile length, now falls back to the peak method
  (`amplitude_method == "generalized_gaussian_fit_fallback"`), and
  `diagnostics["gaussian_fit_rejected"]` names the reason. As with a fit that
  fails to converge, `diagnostics["gaussian_fit_params"]` is absent on
  fallback, so code that indexes it directly now raises `KeyError` on these
  profiles; read it with `.get()` or check `amplitude_method` first. In
  validation, no fit near the true amplitude was refused. Refusals are not
  limited to low CNR: on features wide relative to the profile, or on short
  profiles, typically 1--5% of fits at CNR 1 to 3 were refused (up to about 10%
  when the feature spans most of the profile), each of them degenerate. A
  runaway fit that meets neither condition is still reported: at CNR 0.3 to 1,
  about 1% of profiles read a CNR of 5 to 12 from fits centered outside the
  profile or nearly as wide as it.

### Documentation

- README: the peak method's baseline margin is quantified (profile length in
  full widths at half maximum against amplitude deficit), and the N=128 bias in
  "Profile length" is attributed to that margin rather than to the Welch
  segment count.
- README: the negative CNR bias on flat-topped profiles at high CNR is
  attributed to signal leakage above the signal/noise boundary, which inflates
  the noise level for every amplitude method, rather than to the low-pass
  filter rounding the peak.

## [0.3.1] - 2026-06-29

### Added

- Documentation site built with Sphinx and the pydata theme, published on Read
  the Docs (`docs/`, `.readthedocs.yaml`): installation, quick start, and an API
  reference rendered from the docstrings.
- `docs` optional-dependency group for building the documentation.
- `[project.optional-dependencies]` for the `dev` and `validation` groups, so
  `pip install -e ".[dev]"` works without uv.
- Expanded `[project.urls]` (Homepage, Documentation, Issues, Changelog).

## [0.3.0] - 2026-06-25

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
- `diagnostics["amplitude_sign_ambiguous"]` on the peak and generalized-Gaussian
  methods: true when an opposite-sign excursion at least half the chosen
  feature's magnitude rivals it, so the largest-magnitude read could switch
  features and flip the amplitude sign. It covers the localized competitor that
  `lowfreq_dominated` does not, since a narrow rival is not low-frequency
  baseline structure.
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
- Matched-filter amplitude is now the same flat windowed projection
  (`sum(xw * tw) / sum(tw**2)`), replacing the `Pxx_full`-weighted projection.
  On an isolated feature the two agree closely, so template-path amplitudes are
  essentially unchanged there; when a second feature lies in the window, or the
  template shape does not match the signal, the amplitude can differ from the
  previous release.
- `amplitude_snr` is now defined only on the matched-filter (template) path and
  returns NaN on the peak and generalized-Gaussian paths, where the standard
  error is a different quantity that is not comparable to it.
- The peak amplitude read and `roi="auto"` now locate the largest-magnitude
  feature, so a downward (absorption / dark-contrast) feature is found and
  `amplitude` carries its sign. `cnr` is unchanged: it uses the amplitude
  magnitude, and `NoiseModel.peak_snr` does the same.
- A `template` that is constant after mean-subtraction now raises `ValueError`
  rather than returning a silent all-NaN result: the matched-filter projection
  is undefined when the template has no variation.

### Documentation

- README section "Low-frequency baseline and structured background" describing
  the guard, `roi`, the iSCAT scope limit, and the matched-filter recommendation.
- README "References", "Contributing and community", and "Development disclosure"
  sections, and a Tests CI badge.

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

[Unreleased]: https://github.com/thiebes/fft-cnr/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/thiebes/fft-cnr/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/thiebes/fft-cnr/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/thiebes/fft-cnr/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/thiebes/fft-cnr/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/thiebes/fft-cnr/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/thiebes/fft-cnr/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/thiebes/fft-cnr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/thiebes/fft-cnr/releases/tag/v0.1.0
