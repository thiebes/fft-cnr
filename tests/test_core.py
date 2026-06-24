"""Integration tests for FFT-based CNR estimation."""

from unittest.mock import patch

import numpy as np
import pytest

from fft_cnr import CNREstimate, NoiseModel, fft_cnr


class TestFFTCNR:
    """Tests for fft_cnr with known synthetic signals."""

    def test_returns_cnr_estimate(self):
        rng = np.random.default_rng(42)
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 256))
        noise = rng.normal(0, 0.1, 256)
        result = fft_cnr(signal + noise)
        assert isinstance(result, CNREstimate)

    def test_required_fields_present(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 256)
        result = fft_cnr(x)
        assert isinstance(result.cnr, float)
        assert isinstance(result.cnr_ci95, tuple)
        assert len(result.cnr_ci95) == 2
        assert isinstance(result.amplitude, float)
        assert isinstance(result.noise_rms, float)
        assert isinstance(result.noise_ci95, tuple)
        assert len(result.noise_ci95) == 2
        assert isinstance(result.cutoff_index, (int, np.integer))
        assert isinstance(result.diagnostics, dict)

    def test_diagnostics_fields(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 256)
        result = fft_cnr(x)
        expected_keys = {
            "N", "window_rms", "dof",
            "welch_nperseg", "welch_noverlap", "kept_bins",
        }
        assert expected_keys.issubset(result.diagnostics.keys())
        assert result.diagnostics["N"] == 256

    def test_rejects_short_input(self):
        with pytest.raises(ValueError, match="too short"):
            fft_cnr(np.array([1.0, 2.0, 3.0]))

    def test_noise_rms_positive(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 512)
        result = fft_cnr(x)
        assert result.noise_rms > 0

    def test_noise_ci_brackets_estimate(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 512)
        result = fft_cnr(x)
        lo, hi = result.noise_ci95
        assert lo <= result.noise_rms <= hi

    def test_high_cnr_signal(self):
        """A strong signal with weak noise should yield a large CNR."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 512, endpoint=False)
        signal = 10.0 * np.sin(2 * np.pi * 3 * t)
        noise = rng.normal(0, 0.01, 512)
        result = fft_cnr(signal + noise)
        assert result.cnr > 10

    def test_pure_noise_low_cnr(self):
        """Pure noise (no signal) should yield a low CNR."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 512)
        result = fft_cnr(x)
        assert result.cnr < 10

    def test_default_uses_peak_method(self):
        """Default call (no template, no fit_model) should use peak method."""
        rng = np.random.default_rng(42)
        N = 256
        x = np.arange(N, dtype=float)
        center = (N - 1) / 2.0
        signal = 10.0 * np.exp(-0.5 * ((x - center) / 20.0) ** 2)
        noise = rng.normal(0, 1.0, N)
        result = fft_cnr(signal + noise)
        assert result.diagnostics["amplitude_method"] == "peak"
        assert np.isfinite(result.amplitude_se)
        lo, hi = result.cnr_ci95
        assert np.isfinite(lo) and np.isfinite(hi)

    def test_template_matched_filter(self):
        """Matched filter with correct template should recover amplitude."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 256, endpoint=False)
        template = np.sin(2 * np.pi * 5 * t)
        signal = 3.0 * template
        noise = rng.normal(0, 0.5, 256)
        result = fft_cnr(signal + noise, template=template)
        assert not np.isnan(result.amplitude_se)
        assert result.amplitude > 0

    def test_template_cnr_has_ci(self):
        """With a template, CNR confidence intervals should be finite."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 256, endpoint=False)
        template = np.sin(2 * np.pi * 5 * t)
        signal = 3.0 * template + rng.normal(0, 0.5, 256)
        result = fft_cnr(signal, template=template)
        lo, hi = result.cnr_ci95
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < hi

    def test_window_options(self):
        """All window options should produce valid results."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 256)
        for win in ("tukey", "hann", "none"):
            result = fft_cnr(x, window=win)
            assert np.isfinite(result.noise_rms)

    def test_return_bandpassed_noise(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 256)
        result = fft_cnr(x, return_bandpassed_noise=True)
        assert "x_bp" in result.diagnostics
        assert len(result.diagnostics["x_bp"]) == 256

    def test_cutoff_within_bounds(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 512)
        result = fft_cnr(x)
        assert 1 <= result.cutoff_index <= 512 // 2 + 1

    def test_invalid_fit_model(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 256)
        with pytest.raises(ValueError, match="Unsupported fit_model"):
            fft_cnr(x, fit_model="invalid")


class TestGeneralizedGaussianFit:
    """Tests for the generalized Gaussian fit amplitude estimation path."""

    @staticmethod
    def _make_gen_gaussian_signal(
        N=256, amplitude=10.0, sigma=20.0, noise_std=1.0, p=3.0, seed=42
    ):
        rng = np.random.default_rng(seed)
        x = np.arange(N, dtype=float)
        center = (N - 1) / 2.0
        z = np.abs((x - center) / sigma)
        clean = amplitude * np.exp(-0.5 * z ** p)
        noise = rng.normal(0, noise_std, N)
        return clean + noise, clean, amplitude, noise_std

    def test_recovers_amplitude_gaussian_signal(self):
        """On a standard Gaussian (p=2), should recover amplitude within 20%."""
        signal, _, true_amp, _ = self._make_gen_gaussian_signal(p=2.0)
        result = fft_cnr(signal, fit_model="generalized_gaussian")
        assert abs(result.amplitude - true_amp) / true_amp < 0.2

    def test_recovers_amplitude_flat_top(self):
        """On a flat-topped profile (p=4), should still recover amplitude within 20%."""
        signal, _, true_amp, _ = self._make_gen_gaussian_signal(p=4.0)
        result = fft_cnr(signal, fit_model="generalized_gaussian")
        assert abs(result.amplitude - true_amp) / true_amp < 0.2

    def test_has_se(self):
        signal, _, _, _ = self._make_gen_gaussian_signal()
        result = fft_cnr(signal, fit_model="generalized_gaussian")
        assert np.isfinite(result.amplitude_se)
        assert result.amplitude_se > 0

    def test_cnr_has_ci(self):
        signal, _, _, _ = self._make_gen_gaussian_signal()
        result = fft_cnr(signal, fit_model="generalized_gaussian")
        lo, hi = result.cnr_ci95
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < hi

    def test_diagnostics(self):
        signal, _, _, _ = self._make_gen_gaussian_signal()
        result = fft_cnr(signal, fit_model="generalized_gaussian")
        assert result.diagnostics["amplitude_method"] == "generalized_gaussian_fit"
        gp = result.diagnostics["gaussian_fit_params"]
        assert {"amplitude", "center", "sigma", "baseline", "shape"} == set(gp.keys())

    def test_shape_parameter_near_p(self):
        """Fitted shape should be close to the true p for a clean signal."""
        signal, _, _, _ = self._make_gen_gaussian_signal(
            amplitude=50.0, noise_std=0.5, p=3.0
        )
        result = fft_cnr(signal, fit_model="generalized_gaussian")
        fitted_p = result.diagnostics["gaussian_fit_params"]["shape"]
        assert abs(fitted_p - 3.0) < 1.0


class TestPeakMethod:
    """Tests for the non-parametric peak amplitude estimation path."""

    @staticmethod
    def _make_gaussian_signal(
        N=512, amplitude=20.0, sigma=20.0, noise_std=1.0, seed=42
    ):
        rng = np.random.default_rng(seed)
        x = np.arange(N, dtype=float)
        center = (N - 1) / 2.0
        clean = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        noise = rng.normal(0, noise_std, N)
        return clean + noise, clean, amplitude, noise_std

    @staticmethod
    def _make_flat_top_signal(
        N=512, amplitude=20.0, sigma=20.0, noise_std=1.0, p=4.0, seed=42
    ):
        rng = np.random.default_rng(seed)
        x = np.arange(N, dtype=float)
        center = (N - 1) / 2.0
        z = np.abs((x - center) / sigma)
        clean = amplitude * np.exp(-0.5 * z ** p)
        noise = rng.normal(0, noise_std, N)
        return clean + noise, clean, amplitude, noise_std

    def test_recovers_gaussian_amplitude(self):
        """Peak method should recover Gaussian amplitude within 20%."""
        signal, _, true_amp, _ = self._make_gaussian_signal()
        result = fft_cnr(signal, fit_model="peak")
        assert abs(result.amplitude - true_amp) / true_amp < 0.2

    def test_recovers_flat_top_amplitude(self):
        """Peak method should recover flat-top amplitude within 20%."""
        signal, _, true_amp, _ = self._make_flat_top_signal(p=4.0)
        result = fft_cnr(signal, fit_model="peak")
        assert abs(result.amplitude - true_amp) / true_amp < 0.2

    def test_has_finite_se(self):
        signal, _, _, _ = self._make_gaussian_signal()
        result = fft_cnr(signal, fit_model="peak")
        assert np.isfinite(result.amplitude_se)
        assert result.amplitude_se > 0

    def test_cnr_has_ci(self):
        signal, _, _, _ = self._make_gaussian_signal()
        result = fft_cnr(signal, fit_model="peak")
        lo, hi = result.cnr_ci95
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < hi

    def test_diagnostics(self):
        signal, _, _, _ = self._make_gaussian_signal()
        result = fft_cnr(signal, fit_model="peak")
        assert result.diagnostics["amplitude_method"] == "peak"

    def test_pure_noise(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 512)
        result = fft_cnr(x, fit_model="peak")
        assert isinstance(result, CNREstimate)
        assert np.isfinite(result.cnr)


class TestEdgeCases:
    """Tests for edge cases and untested code paths."""

    def test_template_truncation(self):
        """Template longer than signal should be truncated."""
        rng = np.random.default_rng(42)
        N = 256
        t = np.linspace(0, 1, N, endpoint=False)
        signal = 5.0 * np.sin(2 * np.pi * 3 * t) + rng.normal(0, 0.5, N)
        template_long = np.sin(2 * np.pi * 3 * np.linspace(0, 1, N + 100))
        result = fft_cnr(signal, template=template_long)
        assert result.diagnostics["amplitude_method"] == "matched_filter"
        assert np.isfinite(result.cnr)

    def test_template_padding(self):
        """Template shorter than signal should be zero-padded."""
        rng = np.random.default_rng(42)
        N = 256
        t = np.linspace(0, 1, N, endpoint=False)
        signal = 5.0 * np.sin(2 * np.pi * 3 * t) + rng.normal(0, 0.5, N)
        template_short = np.sin(2 * np.pi * 3 * np.linspace(0, 1, N - 50))
        result = fft_cnr(signal, template=template_short)
        assert result.diagnostics["amplitude_method"] == "matched_filter"
        assert np.isfinite(result.cnr)

    def test_full_length_template_aligned_to_roi(self):
        """A full-profile template combined with roi must be sliced to the same
        window as the data, not truncated from its start. Otherwise the matched
        filter projects onto the wrong samples and the amplitude explodes. The
        full-length template must give the same result as a pre-sliced one."""
        N = 200
        x = np.arange(N, dtype=float)
        clean = np.exp(-0.5 * ((x - 100.0) / 10.0) ** 2)
        rng = np.random.default_rng(0)
        y = clean + rng.normal(0, 0.05, N)
        full = fft_cnr(y, template=clean, roi=(70, 130))
        presliced = fft_cnr(y, template=clean[70:130], roi=(70, 130))
        assert full.amplitude == pytest.approx(presliced.amplitude, rel=1e-9)
        assert full.amplitude == pytest.approx(1.0, rel=0.1)

    def test_generalized_gaussian_fallback(self):
        """When curve_fit fails, should fall back to peak method."""
        rng = np.random.default_rng(42)
        N = 256
        x = np.arange(N, dtype=float)
        center = (N - 1) / 2.0
        signal = 10.0 * np.exp(-0.5 * ((x - center) / 20.0) ** 2)
        noise = rng.normal(0, 1.0, N)
        with patch("fft_cnr.core.curve_fit", side_effect=RuntimeError("mock fail")):
            result = fft_cnr(signal + noise, fit_model="generalized_gaussian")
        assert result.diagnostics["amplitude_method"] == "generalized_gaussian_fit_fallback"
        assert np.isfinite(result.cnr)

    def test_custom_welch_params(self):
        """Custom Welch parameters should be used and reported."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 512)
        result = fft_cnr(x, welch_nperseg=64, welch_noverlap=32)
        assert result.diagnostics["welch_nperseg"] == 64
        assert result.diagnostics["welch_noverlap"] == 32
        assert np.isfinite(result.cnr)

    def test_minimum_valid_length(self):
        """Exactly 16 points (the minimum) should produce a valid result."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 16)
        result = fft_cnr(x)
        assert isinstance(result, CNREstimate)
        assert np.isfinite(result.noise_rms)

    def test_fallback_cut_frac(self):
        """When AIC knee detection returns out-of-bounds, fallback should apply."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 256)
        with patch("fft_cnr.core._break_knee_loglog", return_value=0):
            result = fft_cnr(x, fallback_cut_frac=0.3)
        assert np.isfinite(result.cnr)
        assert result.cutoff_index >= 1


class TestGridConsistencyRegression:
    """Pin numerical outputs on a fixed signal to catch silent regressions.

    The estimation pipeline bridges two frequency grids: knee detection and
    the noise CI run on the coarse Welch PSD grid, while noise extraction and
    amplitude run on the full rFFT grid. The Welch PSD is interpolated to the
    full grid and the knee index is separately scaled across grids. A mismatch
    in those conversions would shift the signal/noise boundary and bias the
    result without changing any output's shape or type, so the structural
    tests above would not catch it. These values are deterministic (fixed seed,
    no randomness in the algorithm); a change here flags that the estimation
    math moved and must be re-validated with scripts/compare_cnr_accuracy.py.
    """

    @staticmethod
    def _fixed_signal():
        rng = np.random.default_rng(20240611)
        N = 512
        x = np.arange(N, dtype=float)
        center = (N - 1) / 2.0
        clean = 20.0 * np.exp(-0.5 * ((x - center) / 25.0) ** 2)
        return clean + rng.normal(0, 1.0, N), clean

    def test_peak_method_values(self):
        noisy, _ = self._fixed_signal()
        result = fft_cnr(noisy)
        assert result.cutoff_index == 24
        assert result.cnr == pytest.approx(19.9876, rel=1e-4)
        assert result.noise_rms == pytest.approx(1.0120, rel=1e-4)
        assert result.amplitude == pytest.approx(20.2278, rel=1e-4)

    def test_matched_filter_cutoff_matches_peak(self):
        """Knee detection is independent of the amplitude method, so the
        cross-grid cutoff index must be identical for both."""
        noisy, clean = self._fixed_signal()
        result = fft_cnr(noisy, template=clean)
        assert result.cutoff_index == 24

    def test_matched_filter_values(self):
        noisy, clean = self._fixed_signal()
        result = fft_cnr(noisy, template=clean)
        assert result.cnr == pytest.approx(0.97641, rel=1e-4)
        assert result.amplitude == pytest.approx(0.98814, rel=1e-4)
        assert result.noise_rms == pytest.approx(1.0120, rel=1e-4)
        # Pin the noise-only-whitened standard error: a regression here would
        # signal the signal-contaminated whitening has crept back in.
        assert result.amplitude_se == pytest.approx(0.0084777, rel=1e-3)

    def test_generalized_gaussian_values(self):
        noisy, _ = self._fixed_signal()
        result = fft_cnr(noisy, fit_model="generalized_gaussian")
        assert result.diagnostics["amplitude_method"] == "generalized_gaussian_fit"
        assert result.cutoff_index == 24
        assert result.cnr == pytest.approx(19.7954, rel=1e-4)
        assert result.amplitude == pytest.approx(20.0333, rel=1e-4)
        assert result.noise_rms == pytest.approx(1.0120, rel=1e-4)

    def test_noise_rms_independent_of_amplitude_method(self):
        """The noise path never branches on the amplitude method, so all
        three methods must report the exact same noise RMS."""
        noisy, clean = self._fixed_signal()
        rms_values = {
            fft_cnr(noisy).noise_rms,
            fft_cnr(noisy, template=clean).noise_rms,
            fft_cnr(noisy, fit_model="generalized_gaussian").noise_rms,
        }
        assert len(rms_values) == 1


class TestAmplitudeSNR:
    """Tests for the amplitude_snr derived property on CNREstimate."""

    def test_template_path_equals_amplitude_over_se(self):
        """On the template path the property is exactly amplitude / SE, which
        is the matched-filter SNR (the efficient, full-covariance estimator)."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 256, endpoint=False)
        template = np.sin(2 * np.pi * 5 * t)
        signal = 3.0 * template + rng.normal(0, 0.5, 256)
        result = fft_cnr(signal, template=template)
        assert result.diagnostics["amplitude_method"] == "matched_filter"
        assert np.isfinite(result.amplitude_se)
        assert result.amplitude_snr == pytest.approx(
            result.amplitude / result.amplitude_se
        )

    def test_nan_when_se_absent(self):
        """With no finite standard error the property is NaN, not fabricated."""
        result = CNREstimate(
            cnr=5.0,
            cnr_ci95=(4.0, 6.0),
            amplitude=10.0,
            amplitude_se=float("nan"),
            noise_rms=2.0,
            noise_ci95=(1.8, 2.2),
            cutoff_index=24,
            diagnostics={"amplitude_method": "matched_filter"},
        )
        assert np.isnan(result.amplitude_snr)

    def test_nan_on_peak_path(self):
        """The peak proxy SE is uncharacterized, so it is not exposed."""
        rng = np.random.default_rng(42)
        N = 256
        x = np.arange(N, dtype=float)
        signal = 10.0 * np.exp(-0.5 * ((x - (N - 1) / 2) / 20.0) ** 2)
        result = fft_cnr(signal + rng.normal(0, 1.0, N))
        assert result.diagnostics["amplitude_method"] == "peak"
        assert np.isfinite(result.amplitude_se)
        assert np.isnan(result.amplitude_snr)

    def test_nan_on_generalized_gaussian_path(self):
        """The fit-residual SE is a different object, so it is not exposed."""
        rng = np.random.default_rng(42)
        N = 256
        x = np.arange(N, dtype=float)
        signal = 10.0 * np.exp(-0.5 * ((x - (N - 1) / 2) / 20.0) ** 2)
        result = fft_cnr(signal + rng.normal(0, 1.0, N),
                         fit_model="generalized_gaussian")
        assert result.diagnostics["amplitude_method"] == "generalized_gaussian_fit"
        assert np.isfinite(result.amplitude_se)
        assert np.isnan(result.amplitude_snr)


class TestNoiseModel:
    """Tests for the NoiseModel dataclass and its derived quantities."""

    def test_noise_model_none_by_default(self):
        rng = np.random.default_rng(42)
        result = fft_cnr(rng.normal(0, 1.0, 256))
        assert result.noise_model is None

    def test_peak_snr_pure_shot_noise(self):
        """With no read floor, peak SNR reduces to sqrt(amplitude / gain)."""
        model = NoiseModel(
            read=0.0,
            gain=0.01,
            spectral_exponent=float("nan"),
            white_floor=float("nan"),
            signal_dependent=True,
            correlated=None,
        )
        assert model.peak_snr(100.0) == pytest.approx(100.0)

    def test_peak_snr_pure_read_noise(self):
        """With zero gain, peak SNR reduces to amplitude / read."""
        model = NoiseModel(
            read=2.0,
            gain=0.0,
            spectral_exponent=float("nan"),
            white_floor=float("nan"),
            signal_dependent=False,
            correlated=None,
        )
        assert model.peak_snr(10.0) == pytest.approx(5.0)

    def test_peak_snr_negative_amplitude_uses_magnitude(self):
        """A negative (dip / dark-contrast) amplitude must give the same finite,
        positive SNR as a peak of equal depth, not a NaN from a negative
        radicand. The peak amplitude read can return a signed value, and
        peak_snr is documented as the recommended read on that path."""
        model = NoiseModel(
            read=0.0,
            gain=0.01,
            spectral_exponent=float("nan"),
            white_floor=float("nan"),
            signal_dependent=True,
            correlated=None,
        )
        assert model.peak_snr(-100.0) == pytest.approx(model.peak_snr(100.0))
        assert np.isfinite(model.peak_snr(-100.0))


class TestNoiseModelDetection:
    """Tests for the real-space (signal-dependence) noise-model detector."""

    @staticmethod
    def _white_noise_signal():
        rng = np.random.default_rng(20240611)
        N = 512
        x = np.arange(N, dtype=float)
        center = (N - 1) / 2.0
        clean = 20.0 * np.exp(-0.5 * ((x - center) / 25.0) ** 2)
        return clean + rng.normal(0, 1.0, N), clean

    @staticmethod
    def _shot_noise_signal(seed=7):
        """Poisson profile whose peak holds 100 photons, so in count units
        the true photon-transfer gain is 1.0 with no read floor and the
        true peak SNR is 10 (the issue #1 construction)."""
        N = 512
        x = np.arange(N, dtype=float)
        center = (N - 1) / 2.0
        lam = 100.0 * np.exp(-0.5 * ((x - center) / 25.0) ** 2) + 1e-9
        return np.random.default_rng(seed).poisson(lam).astype(float)

    def test_white_noise_not_flagged(self):
        noisy, _ = self._white_noise_signal()
        result = fft_cnr(noisy, estimate_noise_model=True)
        model = result.noise_model
        assert model.signal_dependent is False
        assert model.correlated is None
        assert np.isnan(model.spectral_exponent)
        assert np.isnan(model.white_floor)

    def test_white_noise_read_matches_noise_rms(self):
        """Under white noise the photon-transfer intercept and the spectral
        noise estimate measure the same quantity; the shared frac_kept
        attenuation convention is what keeps them consistent."""
        noisy, _ = self._white_noise_signal()
        result = fft_cnr(noisy, estimate_noise_model=True)
        assert result.noise_model.read == pytest.approx(
            result.noise_rms, rel=0.05
        )

    def test_white_noise_pinned_values(self):
        noisy, _ = self._white_noise_signal()
        result = fft_cnr(noisy, estimate_noise_model=True)
        assert result.noise_model.gain == pytest.approx(-0.0227155, rel=1e-4)
        assert result.noise_model.read == pytest.approx(1.0201, rel=1e-4)
        assert result.diagnostics["var_signal_p"] == pytest.approx(0.98)

    def test_shot_noise_flagged_and_gain_recovered(self):
        x = self._shot_noise_signal()
        result = fft_cnr(x, estimate_noise_model=True)
        model = result.noise_model
        assert model.signal_dependent is True
        assert 0.5 < model.gain < 1.5
        assert model.read < 0.5
        assert model.correlated is None
        assert np.isnan(model.spectral_exponent)
        assert np.isnan(model.white_floor)

    def test_shot_noise_peak_snr(self):
        """peak_snr from the fitted model should land near the true peak SNR
        of 10, while the spectral cnr overestimates it severalfold."""
        x = self._shot_noise_signal()
        result = fft_cnr(x, estimate_noise_model=True)
        snr = result.noise_model.peak_snr(result.amplitude)
        assert 7.0 < snr < 14.0
        assert result.cnr > 2.0 * snr

    def test_deterministic_by_default(self):
        noisy, _ = self._white_noise_signal()
        a = fft_cnr(noisy, estimate_noise_model=True)
        b = fft_cnr(noisy, estimate_noise_model=True)
        assert a.noise_model.gain == b.noise_model.gain
        assert a.noise_model.read == b.noise_model.read
        assert a.diagnostics["var_signal_p"] == b.diagnostics["var_signal_p"]

    def test_weak_signal_skipped(self):
        """A profile whose signal range cannot constrain the slope is
        reported as not tested (None/NaN), not as tested-negative."""
        rng = np.random.default_rng(3)
        N = 512
        x = np.arange(N, dtype=float)
        weak = 1.5 * np.exp(-0.5 * ((x - (N - 1) / 2.0) / 25.0) ** 2)
        result = fft_cnr(
            weak + rng.normal(0, 1.0, N), estimate_noise_model=True
        )
        model = result.noise_model
        assert model.signal_dependent is None
        assert np.isnan(model.gain)
        assert np.isnan(model.read)
        assert model.correlated is None
        assert np.isnan(model.spectral_exponent)
        assert np.isnan(model.white_floor)
        assert "noise_model_skipped" in result.diagnostics

    def test_works_with_template_path(self):
        noisy, clean = self._white_noise_signal()
        result = fft_cnr(noisy, template=clean, estimate_noise_model=True)
        assert result.noise_model is not None
        assert result.noise_model.signal_dependent is False

    def test_default_path_unaffected(self):
        noisy, _ = self._white_noise_signal()
        result = fft_cnr(noisy)
        assert result.noise_model is None
        assert "var_signal_p" not in result.diagnostics


class TestLowFreqBaseline:
    """Low-frequency-baseline guard and region-of-interest windowing (issue #7).

    A smooth baseline lives entirely below the spectral knee, so it is
    reconstructed as signal and inflates the CNR even when no peak is present.
    The ``lowfreq_dominated`` diagnostic flags that case, and ``roi`` restricts
    the estimate to a window where the baseline is locally negligible.
    """

    N = 200

    @staticmethod
    def _peak(amp, center=100.0, sigma=10.0):
        x = np.arange(TestLowFreqBaseline.N, dtype=float)
        return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    @staticmethod
    def _baseline():
        x = np.arange(TestLowFreqBaseline.N, dtype=float)
        return 6.0 * np.cos(2 * np.pi * x / 140.0 + 0.6)

    def test_flag_fires_on_peakless_baseline(self):
        """A baseline with no peak must be flagged: CNR should be ~0 but the
        estimator reports it high, so the guard is the only signal of trouble."""
        rng = np.random.default_rng(0)
        y = self._baseline() + rng.normal(0, 1.0, self.N)
        result = fft_cnr(y)
        assert result.diagnostics["lowfreq_dominated"] is True
        assert result.diagnostics["lowfreq_offpeak_ratio"] > 2.5

    def test_flag_silent_on_clean_peak(self):
        """A localized peak on a flat baseline must not be flagged, including
        at marginal CNR where the ratio is independent of peak height."""
        rng = np.random.default_rng(1)
        for amp in (20.0, 2.0):
            y = self._peak(amp) + rng.normal(0, 1.0, self.N)
            result = fft_cnr(y)
            assert result.diagnostics["lowfreq_dominated"] is False

    def test_ratio_nan_on_template_path(self):
        """The off-peak statistic assumes a localized peak; on the matched
        filter the template defines the signal, so the ratio is NaN and the
        flag is never set."""
        rng = np.random.default_rng(2)
        clean = self._peak(20.0)
        result = fft_cnr(clean + rng.normal(0, 1.0, self.N), template=clean)
        assert np.isnan(result.diagnostics["lowfreq_offpeak_ratio"])
        assert result.diagnostics["lowfreq_dominated"] is False

    def test_explicit_roi_records_bounds_and_restricts(self):
        rng = np.random.default_rng(3)
        y = self._peak(20.0) + rng.normal(0, 1.0, self.N)
        result = fft_cnr(y, roi=(70, 130))
        assert result.diagnostics["roi"] == (70, 130)
        assert result.diagnostics["N"] == 60

    def test_roi_clears_flag_on_localized_baseline(self):
        """Windowing to the peak removes off-center baseline structure, so the
        flag set on the full profile clears on the restricted estimate."""
        rng = np.random.default_rng(4)
        # Baseline bump well away from the peak at index 100.
        x = np.arange(self.N, dtype=float)
        bump = 8.0 * np.exp(-0.5 * ((x - 30) / 12.0) ** 2)
        y = self._peak(20.0) + bump + rng.normal(0, 1.0, self.N)
        full = fft_cnr(y)
        windowed = fft_cnr(y, roi=(70, 130))
        assert full.diagnostics["lowfreq_dominated"] is True
        assert windowed.diagnostics["lowfreq_dominated"] is False

    def test_auto_roi_does_not_false_flag_clean_peak(self):
        """The off-peak exclusion scales to the feature width, so windowing a
        clean peak with roi="auto" must not raise lowfreq_dominated: the tight
        window leaves no off-peak region, and the flag stays False (the contract
        that the guard does not false-fire on a clean peak, now under auto-roi)."""
        x = np.arange(self.N, dtype=float)
        for seed in range(8):
            rng = np.random.default_rng(seed)
            y = self._peak(20.0) + rng.normal(0, 1.0, self.N)
            result = fft_cnr(y, roi="auto")
            assert result.diagnostics["lowfreq_dominated"] is False

    def test_auto_roi_tracks_dominant_peak(self):
        rng = np.random.default_rng(5)
        y = self._peak(20.0) + rng.normal(0, 1.0, self.N)
        result = fft_cnr(y, roi="auto")
        start, stop = result.diagnostics["roi"]
        assert start < 100 < stop  # window brackets the true peak center
        assert result.cnr == pytest.approx(20.0, rel=0.2)

    def test_auto_roi_tracks_dominant_dip(self):
        """A downward (absorption / dark-contrast) feature must be located and
        read end to end: the window brackets the dip, the CNR matches its depth,
        and the amplitude carries the negative sign."""
        rng = np.random.default_rng(8)
        y = self._peak(-20.0) + rng.normal(0, 1.0, self.N)
        result = fft_cnr(y, roi="auto")
        start, stop = result.diagnostics["roi"]
        assert start < 100 < stop  # window brackets the true dip center
        assert result.cnr == pytest.approx(20.0, rel=0.2)
        assert result.amplitude < 0

    def test_roi_too_short_raises(self):
        rng = np.random.default_rng(6)
        y = self._peak(20.0) + rng.normal(0, 1.0, self.N)
        with pytest.raises(ValueError, match="fewer than 16"):
            fft_cnr(y, roi=(100, 110))

    def test_invalid_roi_string_raises(self):
        rng = np.random.default_rng(7)
        y = self._peak(20.0) + rng.normal(0, 1.0, self.N)
        with pytest.raises(ValueError, match="Unsupported roi"):
            fft_cnr(y, roi="peak")

    def test_reversed_roi_bounds_raise(self):
        rng = np.random.default_rng(10)
        y = self._peak(20.0) + rng.normal(0, 1.0, self.N)
        with pytest.raises(ValueError, match="must be increasing"):
            fft_cnr(y, roi=(130, 70))

    def test_wrong_length_roi_raises(self):
        rng = np.random.default_rng(11)
        y = self._peak(20.0) + rng.normal(0, 1.0, self.N)
        with pytest.raises(ValueError, match="must be 'auto' or a"):
            fft_cnr(y, roi=(70, 100, 130))

    def test_auto_roi_misses_peak_when_offcenter_baseline_is_larger(self):
        """Documented limitation: ``"auto"`` locates the largest feature, so an
        off-center baseline that exceeds the peak of interest captures the
        window. Explicit bounds are the remedy and recover the peak's CNR."""
        rng = np.random.default_rng(12)
        x = np.arange(self.N, dtype=float)
        # Bump away from the peak at index 100, larger than the peak itself.
        bump = 30.0 * np.exp(-0.5 * ((x - 30) / 12.0) ** 2)
        y = self._peak(20.0) + bump + rng.normal(0, 1.0, self.N)
        auto = fft_cnr(y, roi="auto")
        a_start, a_stop = auto.diagnostics["roi"]
        assert not (a_start < 100 < a_stop)  # auto misses the intended peak
        explicit = fft_cnr(y, roi=(70, 130))
        assert explicit.cnr == pytest.approx(20.0, rel=0.25)

    def test_offpeak_ratio_pinned_values(self):
        """Pin the off-peak ratio so a change to the off-peak window fraction or
        the baseline subtraction cannot shift the dominance calibration
        silently. Deterministic (fixed seed); a change here means the guard was
        recalibrated and must be re-checked against scripts/validate_iscat_baseline.py."""
        rng = np.random.default_rng(0)
        baseline = self._baseline() + rng.normal(0, 1.0, self.N)
        assert fft_cnr(baseline).diagnostics[
            "lowfreq_offpeak_ratio"
        ] == pytest.approx(3.6869, rel=1e-3)
        rng = np.random.default_rng(1)
        clean = self._peak(20.0) + rng.normal(0, 1.0, self.N)
        assert fft_cnr(clean).diagnostics[
            "lowfreq_offpeak_ratio"
        ] == pytest.approx(0.3967, rel=1e-2)
