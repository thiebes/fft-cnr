# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `scripts/validate_record_width.py` prints the fit-path median to three
  decimals.

### Documentation

- README "Profile length": the record-width figures for the generalized
  Gaussian fit come from 6000 profiles per condition
  (`scripts/validate_record_width.py --trials 2000`). The median is within
  about 3% of the simulated CNR down to +/-2.5 SD from CNR 8, and 0.96 at CNR 13
  and +/-2 SD. The paragraph gives the share of fits that fall back to the peak
  method at +/-2.5 and +/-3 SD.

## [0.4.2] - 2026-11-20

### Added

- `scripts/validate_record_width.py`: wrong-sign share, median CNR, and
  fallback share of the generalized-Gaussian path, and the peak path's median
  beside an approximation of its noiseless read, for a centered Gaussian in
  records of +/-1.5 to +/-4 SD.

### Changed

- A generalized-Gaussian result changes only where the second start (see
  Fixed) has a lower residual than the first; elsewhere it is bit-identical to
  0.4.1. The peak path, the matched filter, `noise_rms`, the intervals of those
  paths, and the noise model are bit-identical to 0.4.1. With
  `scripts/compare_release_outputs.py --ref v0.4.1 --allow-fit-changes`, 752
  peak-path and 724 matched-filter runs are identical. Of 744
  generalized-Gaussian runs, 470 are identical and 274 change, only in the
  outputs that follow the fit. In each of the 274, the second start's fit has
  a lower residual than 0.4.1's fit. In 229 it reaches a lower minimum, and in
  8 it converges where 0.4.1's fit failed. In 45 of these the fit is refused,
  so the result is the peak read; 17 of those replace a fit of the wrong sign
  that 0.4.1 kept. In the remaining 37, both starts reach the same minimum
  (residuals within a relative 1e-6). They change because the second fit must
  lower the residual by a relative 1e-9 to replace the first, and the
  optimizer stops at a relative tolerance of 1e-8, so two fits of the same
  minimum can differ by more than the margin. Their residuals differ by 1e-9 to
  8e-9 relative, and the amplitude moves by a median of 2e-5 relative, at most
  5e-4.
- A first fit that fails to converge loses to a second fit that converges. So
  where 0.4.1 fell back to the peak method because its fit failed, the second
  fit can now be kept or refused. Where the second start changes whether the
  fit is kept or falls back, `amplitude_method`,
  `diagnostics["amplitude_sign_ambiguous"]` (computed on the peak read, and
  False on a kept fit), `diagnostics["gaussian_fit_params"]`, and
  `diagnostics["gaussian_fit_rejection_reason"]` change with it.
- The changes concentrate on records narrower than about +/-3 SD and at low
  CNR. In the paired simulation reported in #41, which runs the same profiles
  on 0.4.1 and on this release, the README interval cells (centered Gaussian,
  N = 256 and 512, and the +/-4 SD records of 28 to 52 samples) change as
  follows: from CNR 10 no profile's CNR changes by more than 1%; at CNR 5, up
  to 4.3% of profiles change by more than 1%; at CNR 2, 13.5% to 37% do. On
  flat-topped features (p = 8 and 12), at most 2 of 200 profiles per condition
  change from CNR 3, and 36% to 44% change at CNR 1, where the interval
  coverage rises from about 0.70 to between 0.93 and 0.95.
- In the README tables, rerun on this release: the generalized-Gaussian CNR
  interval at CNR 2 covers 0.92 to 0.93 instead of 0.79 to 0.82 (2000 trials),
  and the generalized-Gaussian range for the Gaussian mixture moves from
  1.12--1.16 to 1.12--1.17 (1.164 to 1.170 at CNR 3). No other entry of the
  interval, bias, or short-record tables changes.
- In the README kinetic-traces simulation, rerun with the same seeds: with the
  generalized Gaussian fit, the single-exponential lifetime error goes from
  3.2% / 0.8% short to within 0.2% at an initial CNR of 10 / 20, and the
  long-lived lifetime error from 46% / 35% short to 3.4% short / 1.3% long. On
  profiles where the band has decayed, the fit falls back to the peak method in
  about 3% of cases instead of 5% to 7%. The matched-filter and peak-method
  figures are unchanged.
- Two costs, from the paired simulation in #41. The generalized-Gaussian path
  runs a second fit, so its run time roughly doubles on long records. And on
  records of 32 and 52 samples with no feature, more fits are kept, mostly fits
  that did not converge from the first start and converge from the second. The
  median CNR read on pure noise rises by 0.15 to 0.2, and the share reading a
  CNR above 3 by less than 0.01; the share above 5 stays at or below 0.4% on
  each record length.
- `scripts/compare_release_outputs.py` compares against `v0.4.1` by default,
  runs every profile on the generalized-Gaussian path too, and adds a seeded
  batch of record widths and long records. With `--allow-fit-changes` it
  accepts a generalized-Gaussian difference only when the replayed fits
  reproduce the parameters each result reports, the working tree's fit lowers
  the residual by more than its margin, and only the outputs that follow the
  fit differ. It prints the residual gains of each outcome, the amplitude
  change of the same-minimum runs, and the moves between a kept fit and the
  fallback.
- The test of a real profile refused by the center rule uses a different seed:
  on the earlier frame the second start finds a lower-residual fit inside the
  record.

### Fixed

- The generalized-Gaussian fit (`fit_model="generalized_gaussian"`) returned a
  fit of the wrong sign, which read about half the CNR, when the feature filled
  most of the record (#41). The fit started only from the sample farthest from
  the profile's median. When the feature fills the record the median lies on
  the feature and that sample is an edge, so the fit could converge to a dip of
  the wrong sign at the edge, a local minimum, and the degenerate-fit screen
  kept it. For a centered Gaussian in a record of +/-2 standard deviations
  (SD), 19% to 35% of profiles at CNR 5 to 34 returned such a fit. The fit now
  also starts from the peak read (the largest excursion of the low-pass
  reconstruction from the edge baseline, at its location, with the baseline at
  the edge level) and keeps the fit with the lower sum of squared residuals.
  The screen and the fallback to the peak method are unchanged.

  With the fix, in simulation (`scripts/validate_record_width.py`), no kept fit
  has the wrong sign on records of +/-2 to +/-4 SD at CNR 5 to 34, nor at
  +/-1.5 SD from CNR 8 (0.7% at CNR 5). From CNR 8 the median CNR stays within
  3% of the simulated value down to +/-2.5 SD, and from CNR 21 down to +/-2 SD
  (0.97 at CNR 13). On 0.4.1 the median stays within 3% down to +/-3 SD from
  CNR 8 and down to +/-2.5 SD from CNR 13, but at +/-2.5 SD up to 7% of fits
  have the wrong sign, which the median hides. At CNR 5 the median reads 0.97
  to 1.06 of the simulated value from +/-2.5 to +/-4 SD; 0.4.1 reads 1.04 and
  1.05 at +/-3.5 and +/-4 SD. Below +/-2 SD the fit cannot separate amplitude,
  baseline, and shape: more fits run away into a flat top as wide as the
  record, are refused, and fall back to the peak method, which reads low there
  (#37). A record of +/-1.5 SD stays unusable on both paths.

### Documentation

- README "Amplitude estimation": the generalized Gaussian fit is described with
  its two starting points, and it is no longer said to be unaffected by a short
  record. "Profile length" gives the record width each method needs.
- README "Confidence intervals", "Bias", and "Kinetic traces" carry the
  figures listed under Changed.

## [0.4.1] - 2026-11-18

### Added

- `NoiseModel.baseline`: the signal level of the profile's baseline, the edge
  level the peak read subtracts (the mean of the outer quarter of the samples
  at each end).
- `NoiseModel.baseline_variance`: the fitted noise variance at that level, from
  the photon-transfer fit with its intercept unclipped. For a model built by
  hand it is derived as `gain * baseline + read**2`. Both fields follow the
  existing ones and have defaults, so construction by position still works,
  and both are NaN when the noise-model fit is skipped. `baseline_variance` can
  be zero or negative when the baseline is near zero counts and the read floor
  is small; treat that as a floor variance of zero.
- `scripts/compare_release_outputs.py`: runs a reference release's `core.py`,
  loaded from a git ref, beside the working tree on a fixed set of profiles and
  checks that every output is bit-identical.

### Changed

- `NoiseModel.peak_snr(amplitude)` computes the variance at the feature's
  extremum as `baseline_variance + gain * amplitude`. It evaluated the fitted
  relation at the signal level `|amplitude|`, which leaves out the baseline: on
  a photon background it read high. For a 100-photon Poisson peak it read 14%
  above the true peak SNR on a 25-photon background and 34% above on 100
  photons. Anchoring the variance at the baseline also keeps a detector offset,
  which carries no shot noise, out of the noise at the peak.
- Behavior change for negative features: `peak_snr` evaluates a dip at its
  bottom, so `peak_snr(-A)` no longer equals `peak_snr(A)` on a nonzero
  baseline. The test that pinned the symmetry is replaced by one for a dip on a
  background.
- `peak_snr` treats a negative fitted gain as zero. This changes its value on
  white-noise profiles whose fitted gain is negative, where it read high. On a
  zero baseline, `baseline_variance` uses the signed intercept where `read` is
  clipped at zero, so `peak_snr` values there can move slightly.

### Fixed

- `peak_snr` returned NaN with a RuntimeWarning when a negative fitted gain made
  the variance negative. It returns NaN, without a warning, only when the model
  predicts no positive variance at the extremum.

### Documentation

- The README and quick-start matched-filter examples passed the noise-free
  signal as the template, which made the amplitude and the CNR read about 1
  instead of 10: the matched-filter amplitude is in units of the template's
  peak. The examples normalize the template to a peak of 1, and the README
  states the convention.

No other output changes: `cnr`, the intervals, the diagnostics, and the fitted
`gain`, `read`, and `var_signal_p` are bit-identical to 0.4.0, as checked by
`scripts/compare_release_outputs.py`.

Known limit, unchanged from 0.4.0: when `estimate_noise_model` finds
signal-dependent noise on data with a detector offset, the intervals use the
fitted variance with the read floor clipped at zero, which overstates it. The
intervals are then conservative (`amplitude_se` 1.8 times the observed scatter
at a 100-count offset, 4.3 times at 1000 counts). `cnr` is unaffected.

## [0.4.0] - 2026-11-17

### Added

- CI job that builds the package on pull requests and checks it with the same
  twine and packaging versions as the PyPI publish action.
- Dependabot weekly updates for the SHA-pinned GitHub Actions, with a seven-day
  cooldown and the publish action proposed on its own.
- A manually triggered run of the publish workflow tests, builds, and passes
  the distributions between jobs without publishing.
- `diagnostics["interval_noise"]`: `"white"`, or `"signal_dependent"` when the
  intervals use the variance fitted by `estimate_noise_model`.
- `scripts/validate_interval_coverage.py`: Monte Carlo coverage, width, and
  standard-error calibration of `noise_ci95` and `cnr_ci95` per amplitude path,
  under white and shot noise.

### Changed

- `noise_ci95`, and with it `cnr_ci95`, takes its degrees of freedom from the
  noise estimate itself: the tapered, band-limited power of the full record,
  whose effective degrees of freedom are computed exactly for white noise
  (`tr(WPW)**2 / tr((WPW)**2)`, with `W` the taper and `P` the kept band). They
  came from the Welch segment count, which only locates the signal/noise
  boundary, and were 4 to 30 times too few for records of 128 to 1024 samples
  (more so the longer the record). The noise interval was 2 to 6 times too
  wide, the CNR interval at CNR 20 about 3 to 4 times (N = 256 and 512), and
  both covered 0.99 to 1.00 against a nominal 0.95. On white noise the noise
  interval now covers 0.93 to 0.95, and at CNR 20 the CNR interval spans about
  0.14 to 0.21 of the CNR
  (N = 512 and 256) instead of about 0.58. `diagnostics["dof"]` is a float.
  This change alone leaves `cnr`, `noise_rms`, and `amplitude` unchanged.
- The peak path's `amplitude_se` is the exact standard error of the read given
  its location, `noise_rms * ||L e_i - b||` (the low-pass reconstruction at the
  peak minus the edge baseline), replacing the proxy `noise_rms /
  sqrt(cutoff_index)`, which understated the scatter by up to a factor of two.
  The oversized noise term had hidden this; with the noise interval corrected,
  the peak-path CNR interval covers 0.94 to 0.95 from CNR 5 and 0.91 to 0.93 at
  CNR 2, where the upward bias of the maximum read remains. These figures are
  for a feature centered in the profile. A feature that reaches the outer
  quarters of the profile, where the peak path reads its baseline, is read low,
  and the narrower interval no longer hides that bias: for a Gaussian of
  standard deviation 30 samples centered on the first sample of 512, the CNR
  reads 0.84 of the true value and the interval covers 0.04, where 0.3.3
  covered 0.95 only because its interval was about four times too wide. The
  matched filter covers 0.94 to 0.95 there.
- With `estimate_noise_model=True`, when the detector finds signal-dependent
  noise, the fitted photon-transfer variance replaces the white-noise forms in
  the noise degrees of freedom and in `amplitude_se` on the peak and
  matched-filter paths. Under shot noise (background 10 counts, CNR 20) the
  CNR interval then covers 0.93 to 0.95; the default, white-noise intervals
  cover 0.80 to 0.84 there.
- `diagnostics["gaussian_fit_rejected"]` is renamed
  `diagnostics["gaussian_fit_rejection_reason"]`. The value is a string naming
  the reason, and the old name read like the boolean flags beside it
  (`lowfreq_dominated`, `amplitude_sign_ambiguous`). Code that reads the old
  key must use the new name.
- The source distribution lists its contents explicitly: the package, tests,
  docs, validation scripts, `README.md`, `LICENSE`, `CHANGELOG.md`, and
  `CITATION.cff`, plus `pyproject.toml` and `.gitignore`, which the build
  backend always includes. Repository-only files are no longer included
  (`assets/`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `uv.lock`, `.github/`,
  `.readthedocs.yaml`, `.gitattributes`), nor are files that are ignored only
  locally. The wheel is unchanged.
- The test, docs, and compatibility workflow tokens are restricted to
  read-only contents.
- The PyPI publish workflow builds in a separate job without the publishing
  token and hands the distributions to the publish job as an artifact. The
  build backend and its dependencies are pinned with hashes
  (`.github/build-constraints.txt`), the build runs on a pinned Python, the uv
  cache is off throughout the workflow, checkout does not keep credentials, and
  the workflow token is read-only except for the publish job's `id-token`.
- The `validation` extra and the `validation` and `assets` dependency groups
  install only on Python 3.11 and later. Their matplotlib dependency pulls in
  Pillow, whose security fixes (13.0.0 and later) no longer support Python
  3.10. The package itself still supports Python 3.10.
- CI and the publish workflow install a pinned uv (0.13.0) that setup-uv
  verifies against its known checksums; setup-uv v10 also disables its cache
  on release runs.

### Fixed

- The matched filter's `amplitude_se` is now the exact standard error of the
  projection. The data are demeaned before the projection, so its weights in
  the raw profile have their mean removed; the error was computed with the
  mean left in, which overstated it by about 0.3% for a narrow centered
  feature, 1 to 2% for a centered feature that fills the record, about 4% at
  the profile edge, and over 30% for a template centered outside the data. The
  amplitude is unchanged; `amplitude_se`, `amplitude_snr`, and `cnr_ci95` on
  the template path are slightly smaller.
- On a short record, the noise estimate no longer counts the taper's edge ramps
  as noise. The Tukey taper was applied to the mean-subtracted profile. On a
  record spanning +/-4 standard deviations of a Gaussian feature the mean sits
  about 0.31 of the peak above the edges, and the taper ramps that offset to
  zero over N/8 samples, which puts power above the signal/noise boundary.
  That leaked power is fixed while the noise falls, so `noise_rms` read high
  and `cnr` low as the CNR rose: at N = 28, 40, and 52 a CNR of 100 read as
  about 23, 44, and 78, and the CNR interval covered 0.00, 0.12, and 0.50. The
  noise path now tapers the profile relative to its edge level, the mean of the
  samples under the taper ramps. The amplitude, the Welch spectrum, the cutoff,
  and the low-pass reconstruction are unchanged. On those records the median
  noise level now reads within 2% of its expected value, except 4% high at
  N = 28 and CNR 100, and for a centered feature the CNR interval covers 0.92
  to 0.97 from CNR 5 on every amplitude path. The change applies at every
  length, so `noise_rms` and `cnr` are no longer identical to 0.3.3 from
  N = 128 either: on Gaussian, generalized-Gaussian, Voigt, and narrow-peak
  profiles at N = 256 and 512 and on pure noise at N = 128 to 1024,
  `noise_rms` moved by at most 0.24% on any
  profile, with a median change of about 1e-5. With `roi="auto"`, whose window
  is itself a short record, the same leak was present and the correction is
  larger: for a Gaussian of standard deviation 10 samples at N = 512 and CNR
  100 the median noise level fell from 1.10 to 1.00 times the true value. A
  feature narrow enough to put its own power above the boundary keeps that
  separate leak, and it can read slightly higher, because the pedestal ramp no
  longer partly cancels it: for a Gaussian of standard deviation 3.1 samples
  (full width at half maximum about 7.4) at N = 64 and CNR 100, the noise level
  reads 1.44 instead of 1.41 times the true value. The Hann and untapered
  windows are unaffected, because a Hann ramp spans the whole record and an
  untapered offset is pure DC. With `estimate_noise_model=True` the
  significance null draws its noise at `noise_rms`, so
  `diagnostics["var_signal_p"]` can also shift slightly.
- The generalized-Gaussian path also refuses a fit whose center is pinned at
  its bound, half a profile length outside the data
  (`gaussian_fit_rejection_reason == "center_at_bound"`), and falls back to the
  peak method. Some fits kept in 0.3.3 therefore fall back in 0.4.0: on those
  profiles `diagnostics["gaussian_fit_params"]` is absent and `cnr` changes. A
  center outside the profile but short of its bound, up to almost half a
  profile length outside, is still accepted, because a genuine feature can sit
  at the edge of the data. A runaway fit that meets none of the three
  conditions is still reported. In validation, at most 1% of profiles at CNR
  0.3 read a CNR above 5, and the runaway fits that were kept reached a fit
  CNR of up to 12.5 at a true CNR of 1.5 or below.
- Validation of the center rule, for a feature inside the profile. It used
  five shapes with the feature centered (profile lengths 241 and 512, CNR 0.1
  to 20, 400 profiles per condition), and Gaussian, flat-topped, and
  Lorentzian features centered or a tenth of the profile length from the edge
  (CNR 0.3 to 30, 200 profiles per condition). For a centered feature, the
  rule acted only at CNR 3 and below, on about 1% of fits or fewer. The
  refused fits whose sign was examined all had the wrong sign: a negative wing
  under a positive peak, which could read a CNR of up to about 13. For a
  feature a tenth of the profile length from the edge, it refused up to 2.5%
  of fits, only at CNR 1 and below, with either sign.
- Validation of the center rule, for a feature at the profile edge or outside
  the data. It used the same three shapes and two profile lengths, with the
  peak at the first sample, a quarter of the profile length before it, or a
  fifth of the profile length after the end (CNR 0.3 to 30, 200 profiles per
  condition). With the peak at the profile edge, the rule refused up to 18% of
  fits at CNR 1 to 10 (up to 3.5% for a flat-topped feature), mostly fits of
  the correct sign. At CNR 3 to 10 the refused fits were off by a median of 30%
  or more, and the peak-method fallback by 10 to 25%. At CNR 1 both reads were
  poor. With the peak outside the data, it refused at most 1.5% of fits for
  Gaussian and flat-topped features, and up to 12.5% for a Lorentzian at CNR
  10 to 30. In those Lorentzian cases both the fit and the fallback were off by
  about 75 to 115%, because the peak is not in the data. At lower CNR the few
  refused fits varied widely.

### Documentation

- README: the citation guidance asks for the concept DOI together with the
  version used, instead of a per-release DOI.
- README and quick start: a note on kinetic traces under "Amplitude
  estimation". To follow a feature's amplitude across a series of profiles,
  use the matched filter and read the signed `amplitude`. The note gives the
  errors in fitted lifetimes for each amplitude method in simulation, and the
  false floor of about 1.6 noise standard deviations in a trace built from
  `cnr`.

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

[Unreleased]: https://github.com/thiebes/fft-cnr/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/thiebes/fft-cnr/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/thiebes/fft-cnr/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/thiebes/fft-cnr/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/thiebes/fft-cnr/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/thiebes/fft-cnr/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/thiebes/fft-cnr/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/thiebes/fft-cnr/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/thiebes/fft-cnr/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/thiebes/fft-cnr/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/thiebes/fft-cnr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/thiebes/fft-cnr/releases/tag/v0.1.0
