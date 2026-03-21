"""Unit tests for the characterization module."""

import numpy as np
import pandas as pd
import pytest
from fft_cnr.characterization import (
    generate_gaussian_profile,
    add_calibrated_noise,
    characterize_cnr_level,
    run_characterization_study,
    plot_validation_summary,
    plot_bias_and_rmse,
)


class TestGenerateGaussianProfile:
    """Tests for Gaussian profile generation."""

    def test_basic_generation(self):
        """Test basic Gaussian profile generation."""
        profile = generate_gaussian_profile(num_points=100)

        assert len(profile) == 100
        assert isinstance(profile, np.ndarray)
        assert profile.max() > 0

    def test_peak_amplitude(self):
        """Test that peak amplitude is correct."""
        amplitude = 5.0
        profile = generate_gaussian_profile(amplitude=amplitude, num_points=100)

        # Peak should be close to specified amplitude
        assert abs(profile.max() - amplitude) < 0.01

    def test_peak_location(self):
        """Test that peak is at the correct center position."""
        center = 0.7
        profile = generate_gaussian_profile(center=center, num_points=100)

        # Find peak location
        peak_idx = np.argmax(profile)
        x = np.linspace(0.0, 1.0, 100)
        peak_position = x[peak_idx]

        # Should be close to specified center
        assert abs(peak_position - center) < 0.02

    def test_different_sigma(self):
        """Test that sigma affects profile width."""
        narrow = generate_gaussian_profile(sigma=0.05, num_points=100)
        wide = generate_gaussian_profile(sigma=0.2, num_points=100)

        # Wider Gaussian should have more points above half-max
        half_max_narrow = len(narrow[narrow > 0.5 * narrow.max()])
        half_max_wide = len(wide[wide > 0.5 * wide.max()])

        assert half_max_wide > half_max_narrow

    def test_custom_x_range(self):
        """Test with custom x-axis range."""
        profile = generate_gaussian_profile(
            center=5.0,
            x_min=0.0,
            x_max=10.0,
            num_points=100
        )

        assert len(profile) == 100
        assert profile.max() > 0


class TestAddCalibratedNoise:
    """Tests for calibrated noise addition."""

    def test_noise_addition(self):
        """Test that noise is added to signal."""
        signal = np.ones(100)
        noisy = add_calibrated_noise(signal, cnr_nominal=10.0, rng_seed=42)

        # Noisy signal should differ from original
        assert not np.allclose(signal, noisy)
        assert len(noisy) == len(signal)

    def test_cnr_nominal_validation(self):
        """Test that non-positive CNR raises error."""
        signal = np.ones(100)

        with pytest.raises(ValueError, match="CNR nominal must be positive"):
            add_calibrated_noise(signal, cnr_nominal=0)

        with pytest.raises(ValueError, match="CNR nominal must be positive"):
            add_calibrated_noise(signal, cnr_nominal=-5.0)

    def test_zero_amplitude_signal_raises_error(self):
        """Test that zero amplitude signal raises error."""
        signal = np.zeros(100)

        with pytest.raises(ValueError, match="zero amplitude"):
            add_calibrated_noise(signal, cnr_nominal=10.0)

    def test_higher_cnr_less_noise(self):
        """Test that higher CNR results in less noise."""
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))

        # Add noise at different CNR levels
        noisy_low_cnr = add_calibrated_noise(signal, cnr_nominal=5.0, rng_seed=42)
        noisy_high_cnr = add_calibrated_noise(signal, cnr_nominal=50.0, rng_seed=42)

        # Calculate actual noise added
        noise_low = noisy_low_cnr - signal
        noise_high = noisy_high_cnr - signal

        # Higher CNR should have smaller noise
        assert np.std(noise_high) < np.std(noise_low)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same noise."""
        signal = np.ones(100)

        noisy1 = add_calibrated_noise(signal, cnr_nominal=10.0, rng_seed=123)
        noisy2 = add_calibrated_noise(signal, cnr_nominal=10.0, rng_seed=123)

        assert np.allclose(noisy1, noisy2)

    def test_different_seeds_different_noise(self):
        """Test that different seeds produce different noise."""
        signal = np.ones(100)

        noisy1 = add_calibrated_noise(signal, cnr_nominal=10.0, rng_seed=123)
        noisy2 = add_calibrated_noise(signal, cnr_nominal=10.0, rng_seed=456)

        assert not np.allclose(noisy1, noisy2)


class TestCharacterizeCNRLevel:
    """Tests for single CNR level characterization."""

    def test_basic_characterization(self):
        """Test basic characterization at one CNR level."""
        results = characterize_cnr_level(
            cnr_nominal=10.0,
            num_trials=10,
            num_points=100,
            rng_seed=42
        )

        # Check that all expected keys are present
        expected_keys = [
            'cnr_nominal', 'cnr_estimates', 'cnr_ratios',
            'mean_ratio', 'median_ratio', 'mode_ratio', 'std_ratio',
            'bias', 'rmse', 'percentile_5', 'percentile_95'
        ]
        for key in expected_keys:
            assert key in results

    def test_cnr_estimates_array(self):
        """Test that CNR estimates are returned as array."""
        results = characterize_cnr_level(
            cnr_nominal=10.0,
            num_trials=20,
            rng_seed=42
        )

        assert len(results['cnr_estimates']) == 20
        assert isinstance(results['cnr_estimates'], np.ndarray)
        assert all(results['cnr_estimates'] > 0)

    def test_statistics_are_scalars(self):
        """Test that statistics are scalar values."""
        results = characterize_cnr_level(
            cnr_nominal=10.0,
            num_trials=10,
            rng_seed=42
        )

        assert isinstance(results['mean_ratio'], (float, np.floating))
        assert isinstance(results['median_ratio'], (float, np.floating))
        assert isinstance(results['std_ratio'], (float, np.floating))

    def test_ratios_calculation(self):
        """Test that ratios are correctly calculated."""
        results = characterize_cnr_level(
            cnr_nominal=10.0,
            num_trials=10,
            rng_seed=42
        )

        # Manually calculate ratios
        expected_ratios = results['cnr_estimates'] / results['cnr_nominal']

        assert np.allclose(results['cnr_ratios'], expected_ratios)

    def test_percentiles_in_range(self):
        """Test that percentiles are in reasonable range."""
        results = characterize_cnr_level(
            cnr_nominal=10.0,
            num_trials=50,
            rng_seed=42
        )

        # 5th percentile should be less than median
        assert results['percentile_5'] < results['median_ratio']

        # 95th percentile should be greater than median
        assert results['percentile_95'] > results['median_ratio']

    def test_custom_signal_params(self):
        """Test with custom signal parameters."""
        signal_params = {
            'center': 0.3,
            'sigma': 0.15,
            'amplitude': 2.0
        }

        results = characterize_cnr_level(
            cnr_nominal=15.0,
            num_trials=10,
            signal_params=signal_params,
            rng_seed=42
        )

        assert results['cnr_nominal'] == 15.0
        assert len(results['cnr_estimates']) == 10


class TestRunCharacterizationStudy:
    """Tests for full characterization study."""

    def test_basic_study(self):
        """Test basic characterization study."""
        results = run_characterization_study(
            cnr_range=(5.0, 20.0),
            num_cnr_levels=5,
            num_trials_per_level=10,
            rng_seed=42,
            verbose=False
        )

        # Should return a DataFrame
        assert isinstance(results, pd.DataFrame)

        # Should have 5 rows (one per CNR level)
        assert len(results) == 5

    def test_dataframe_columns(self):
        """Test that DataFrame has expected columns."""
        results = run_characterization_study(
            cnr_range=(5.0, 20.0),
            num_cnr_levels=3,
            num_trials_per_level=5,
            verbose=False,
            rng_seed=42
        )

        expected_columns = [
            'cnr_nominal', 'mean_ratio', 'median_ratio', 'mode_ratio',
            'std_ratio', 'bias', 'rmse', 'percentile_5', 'percentile_95'
        ]

        for col in expected_columns:
            assert col in results.columns

    def test_log_spacing(self):
        """Test logarithmic spacing of CNR levels."""
        results = run_characterization_study(
            cnr_range=(1.0, 100.0),
            num_cnr_levels=10,
            num_trials_per_level=5,
            use_log_spacing=True,
            verbose=False,
            rng_seed=42
        )

        cnr_values = results['cnr_nominal'].values

        # Check that spacing is roughly logarithmic
        log_cnr = np.log10(cnr_values)
        log_diffs = np.diff(log_cnr)

        # Differences in log space should be approximately constant
        assert np.std(log_diffs) < 0.1

    def test_linear_spacing(self):
        """Test linear spacing of CNR levels."""
        results = run_characterization_study(
            cnr_range=(10.0, 50.0),
            num_cnr_levels=5,
            num_trials_per_level=5,
            use_log_spacing=False,
            verbose=False,
            rng_seed=42
        )

        cnr_values = results['cnr_nominal'].values

        # Check that spacing is roughly linear
        diffs = np.diff(cnr_values)

        # Differences should be approximately constant
        assert np.std(diffs) < 1.0

    def test_cnr_range_coverage(self):
        """Test that CNR range is properly covered."""
        cnr_min, cnr_max = 5.0, 50.0

        results = run_characterization_study(
            cnr_range=(cnr_min, cnr_max),
            num_cnr_levels=10,
            num_trials_per_level=5,
            verbose=False,
            rng_seed=42
        )

        cnr_values = results['cnr_nominal'].values

        # Should include values close to min and max
        assert cnr_values.min() >= cnr_min * 0.99
        assert cnr_values.max() <= cnr_max * 1.01


class TestVisualizationFunctions:
    """Tests for visualization functions."""

    def test_plot_validation_summary_creates_figure(self):
        """Test that validation summary plot is created."""
        # Create minimal test data
        results = run_characterization_study(
            cnr_range=(5.0, 20.0),
            num_cnr_levels=3,
            num_trials_per_level=5,
            verbose=False,
            rng_seed=42
        )

        fig = plot_validation_summary(results)

        # Should return a Figure object
        assert fig is not None
        assert hasattr(fig, 'axes')

        # Should have 4 subplots (2x2 grid)
        assert len(fig.axes) == 4

    def test_plot_bias_and_rmse_creates_figure(self):
        """Test that bias/RMSE plot is created."""
        # Create minimal test data
        results = run_characterization_study(
            cnr_range=(5.0, 20.0),
            num_cnr_levels=3,
            num_trials_per_level=5,
            verbose=False,
            rng_seed=42
        )

        fig = plot_bias_and_rmse(results)

        # Should return a Figure object
        assert fig is not None
        assert hasattr(fig, 'axes')

        # Should have 2 subplots (1x2 grid)
        assert len(fig.axes) == 2

    def test_plot_with_custom_figsize(self):
        """Test plotting with custom figure size."""
        results = run_characterization_study(
            cnr_range=(5.0, 20.0),
            num_cnr_levels=3,
            num_trials_per_level=5,
            verbose=False,
            rng_seed=42
        )

        figsize = (10, 8)
        fig = plot_validation_summary(results, figsize=figsize)

        # Check figure size (approximately, due to DPI)
        assert abs(fig.get_figwidth() - figsize[0]) < 0.1
        assert abs(fig.get_figheight() - figsize[1]) < 0.1


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self):
        """Test complete workflow from generation to analysis."""
        # Generate a signal
        signal = generate_gaussian_profile(num_points=100)
        assert len(signal) == 100

        # Add calibrated noise
        noisy = add_calibrated_noise(signal, cnr_nominal=10.0, rng_seed=42)
        assert len(noisy) == 100

        # Run characterization at one level
        level_results = characterize_cnr_level(
            cnr_nominal=10.0,
            num_trials=20,
            rng_seed=42
        )
        assert 'mean_ratio' in level_results

        # Run full study
        study_results = run_characterization_study(
            cnr_range=(5.0, 20.0),
            num_cnr_levels=5,
            num_trials_per_level=10,
            verbose=False,
            rng_seed=42
        )
        assert len(study_results) == 5

        # Create plots (just verify they don't crash)
        fig1 = plot_validation_summary(study_results)
        assert fig1 is not None

        fig2 = plot_bias_and_rmse(study_results)
        assert fig2 is not None
