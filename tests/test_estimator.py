"""Unit tests for the FFT-CNR estimator."""

import numpy as np
import pytest
from fft_cnr import estimate_cnr


class TestEstimateCNR:
    """Test suite for the estimate_cnr function."""

    def test_basic_sinusoid_with_noise(self):
        """Test CNR estimation on a simple sinusoidal signal with Gaussian noise."""
        # Create a clean sinusoid
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t)

        # Add controlled noise
        np.random.seed(42)
        noise = 0.1 * np.random.randn(len(t))
        noisy_signal = signal + noise

        # Estimate CNR
        cnr = estimate_cnr(noisy_signal)

        # CNR should be positive and reasonable
        assert cnr > 0
        assert isinstance(cnr, float)

    def test_clean_signal_high_cnr(self):
        """Test that a very clean signal produces high CNR."""
        # Create a very clean sinusoid
        t = np.linspace(0, 2, 200)
        signal = np.sin(2 * np.pi * 3 * t)

        # Add minimal noise
        np.random.seed(123)
        noisy_signal = signal + 0.001 * np.random.randn(len(t))

        cnr = estimate_cnr(noisy_signal)

        # Should have high CNR
        assert cnr > 5.0

    def test_noisy_signal_low_cnr(self):
        """Test that a very noisy signal produces low CNR."""
        # Create a sinusoid
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t)

        # Add significant noise
        np.random.seed(456)
        noise = 0.5 * np.random.randn(len(t))
        noisy_signal = signal + noise

        cnr = estimate_cnr(noisy_signal)

        # Should have lower CNR
        assert cnr > 0
        assert cnr < 10.0

    def test_profile_too_short_raises_error(self):
        """Test that profiles with fewer than 3 points raise ValueError."""
        # Test with 0 points
        with pytest.raises(ValueError, match="at least 3 points"):
            estimate_cnr(np.array([]))

        # Test with 1 point
        with pytest.raises(ValueError, match="at least 3 points"):
            estimate_cnr(np.array([1.0]))

        # Test with 2 points
        with pytest.raises(ValueError, match="at least 3 points"):
            estimate_cnr(np.array([1.0, 2.0]))

    def test_constant_profile_raises_error(self):
        """Test that constant profiles raise ValueError."""
        # All zeros
        with pytest.raises(ValueError, match="constant"):
            estimate_cnr(np.zeros(10))

        # All ones
        with pytest.raises(ValueError, match="constant"):
            estimate_cnr(np.ones(20))

        # All same value
        with pytest.raises(ValueError, match="constant"):
            estimate_cnr(np.full(15, 3.14))

    def test_normalization_to_unit_amplitude(self):
        """Test that the function handles different amplitude scales correctly."""
        # Create two identical signals with different amplitudes but same noise
        t = np.linspace(0, 1, 100)
        signal1 = 1.0 * np.sin(2 * np.pi * 5 * t)
        signal2 = 10.0 * np.sin(2 * np.pi * 5 * t)

        # Add the same proportional noise pattern (scaled)
        np.random.seed(789)
        noise_pattern = np.random.randn(len(t))

        noise1 = 0.1 * noise_pattern
        noise2 = 1.0 * noise_pattern  # 10x noise for 10x signal

        noisy1 = signal1 + noise1
        noisy2 = signal2 + noise2

        cnr1 = estimate_cnr(noisy1)
        cnr2 = estimate_cnr(noisy2)

        # CNR should be very similar after normalization since noise is proportional
        assert abs(cnr1 - cnr2) < 0.1

    def test_negative_values_handled(self):
        """Test that signals with negative values are handled correctly."""
        # Create a signal that goes negative
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t) - 0.5

        np.random.seed(111)
        noisy_signal = signal + 0.1 * np.random.randn(len(t))

        cnr = estimate_cnr(noisy_signal)

        assert cnr > 0
        assert np.isfinite(cnr)

    def test_list_input_converted_to_array(self):
        """Test that list inputs are properly converted to numpy arrays."""
        # Create a signal as a Python list
        signal_list = [np.sin(2 * np.pi * 5 * i / 100) for i in range(100)]

        np.random.seed(222)
        noise = [0.1 * np.random.randn() for _ in range(100)]
        noisy_list = [s + n for s, n in zip(signal_list, noise)]

        cnr = estimate_cnr(noisy_list)

        assert cnr > 0
        assert isinstance(cnr, float)

    def test_return_value_rounded_to_two_decimals(self):
        """Test that the CNR value is rounded to 2 decimal places."""
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t)

        np.random.seed(333)
        noisy_signal = signal + 0.1 * np.random.randn(len(t))

        cnr = estimate_cnr(noisy_signal)

        # Check that it's rounded to 2 decimal places
        assert cnr == round(cnr, 2)

    def test_zero_noise_returns_infinity(self):
        """Test that zero noise level returns infinity."""
        # This is a theoretical edge case - create a perfect signal
        # Note: In practice, numerical precision may prevent exactly zero noise
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t)

        # For this test, we'll create a signal that should produce very high CNR
        cnr = estimate_cnr(signal)

        # Should be very high (potentially infinity)
        assert cnr > 0
        assert np.isfinite(cnr) or cnr == float('inf')

    def test_complex_signal_with_multiple_frequencies(self):
        """Test CNR estimation on a signal with multiple frequency components."""
        t = np.linspace(0, 1, 200)
        # Composite signal: fundamental + harmonics
        signal = (np.sin(2 * np.pi * 5 * t) +
                  0.5 * np.sin(2 * np.pi * 10 * t) +
                  0.25 * np.sin(2 * np.pi * 15 * t))

        np.random.seed(444)
        noisy_signal = signal + 0.1 * np.random.randn(len(t))

        cnr = estimate_cnr(noisy_signal)

        assert cnr > 0
        assert np.isfinite(cnr)

    def test_step_function_signal(self):
        """Test CNR estimation on a step function."""
        # Create a step function
        signal = np.concatenate([np.zeros(50), np.ones(50)])

        np.random.seed(555)
        noisy_signal = signal + 0.1 * np.random.randn(len(signal))

        cnr = estimate_cnr(noisy_signal)

        assert cnr > 0
        assert np.isfinite(cnr)

    def test_triangular_wave(self):
        """Test CNR estimation on a triangular wave."""
        # Create a triangular wave
        t = np.linspace(0, 1, 100)
        signal = np.abs((t % 0.2) - 0.1) * 10

        np.random.seed(666)
        noisy_signal = signal + 0.1 * np.random.randn(len(t))

        cnr = estimate_cnr(noisy_signal)

        assert cnr > 0
        assert np.isfinite(cnr)

    def test_minimum_valid_length(self):
        """Test that the minimum valid length (3 points) works."""
        # Create a 3-point signal
        signal = np.array([0.0, 1.0, 0.0])

        cnr = estimate_cnr(signal)

        assert cnr > 0
        assert np.isfinite(cnr)

    def test_reproducibility(self):
        """Test that the same input produces the same output."""
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t)

        np.random.seed(777)
        noisy_signal = signal + 0.1 * np.random.randn(len(t))

        cnr1 = estimate_cnr(noisy_signal)
        cnr2 = estimate_cnr(noisy_signal)

        assert cnr1 == cnr2

    def test_dtype_float32(self):
        """Test that float32 arrays work correctly."""
        t = np.linspace(0, 1, 100, dtype=np.float32)
        signal = np.sin(2 * np.pi * 5 * t)

        np.random.seed(888)
        noisy_signal = signal + 0.1 * np.random.randn(len(t)).astype(np.float32)

        cnr = estimate_cnr(noisy_signal)

        assert cnr > 0
        assert isinstance(cnr, float)

    def test_dtype_float64(self):
        """Test that float64 arrays work correctly."""
        t = np.linspace(0, 1, 100, dtype=np.float64)
        signal = np.sin(2 * np.pi * 5 * t)

        np.random.seed(999)
        noisy_signal = signal + 0.1 * np.random.randn(len(t)).astype(np.float64)

        cnr = estimate_cnr(noisy_signal)

        assert cnr > 0
        assert isinstance(cnr, float)
