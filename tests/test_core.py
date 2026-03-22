"""Integration tests for FFT-based CNR estimation."""

from unittest.mock import patch

import numpy as np
import pytest

from fft_cnr import CNREstimate, fft_cnr


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
