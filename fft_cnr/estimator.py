"""FFT-based Contrast-to-Noise Ratio (CNR) estimation for 1D signal profiles."""

import numpy as np
from scipy.signal import find_peaks


def estimate_cnr(noisy_profile: np.ndarray) -> float:
    """
    Estimate the Contrast-to-Noise Ratio (CNR) from a 1D signal profile using FFT analysis.

    This function analyzes the frequency content of a signal to separate signal from noise
    components. It uses the Fast Fourier Transform (FFT) to identify the dominant signal
    frequencies and estimates the noise level from high-frequency components.

    Algorithm:
    1. Normalize the profile to unit amplitude (divide by max)
    2. Compute orthogonally normalized single-sided FFT
    3. Find peaks and minima in the FFT modulus
    4. Identify the first local minimum after the signal peak
    5. Calculate noise as RMS of high-frequency components from that minimum onward
    6. Return CNR = 1 / noise_level, rounded to 2 decimal places

    Parameters
    ----------
    noisy_profile : np.ndarray
        1D array containing the signal profile to analyze.

    Returns
    -------
    float
        The estimated CNR value, rounded to 2 decimal places.
        Returns float('inf') if noise level is zero.

    Raises
    ------
    ValueError
        If the profile has fewer than 3 points.
    ValueError
        If the profile is constant (all values are the same).

    Examples
    --------
    >>> import numpy as np
    >>> from fft_cnr import estimate_cnr
    >>>
    >>> # Create a clean sinusoidal signal
    >>> t = np.linspace(0, 1, 100)
    >>> signal = np.sin(2 * np.pi * 5 * t)
    >>>
    >>> # Add some noise
    >>> noisy_signal = signal + 0.1 * np.random.randn(len(t))
    >>>
    >>> # Estimate CNR
    >>> cnr = estimate_cnr(noisy_signal)
    >>> print(f"CNR: {cnr}")
    """
    # Convert to numpy array if needed
    profile = np.asarray(noisy_profile)

    # Edge case: Check for minimum length
    if len(profile) < 3:
        raise ValueError("Profile must contain at least 3 points for CNR estimation")

    # Edge case: Check for constant profile
    if np.allclose(profile, profile[0]):
        raise ValueError("Profile is constant; cannot estimate CNR")

    # Step 1: Normalize the profile to unit amplitude
    max_val = np.max(np.abs(profile))
    if max_val == 0:
        raise ValueError("Profile has zero amplitude; cannot estimate CNR")
    normalized_profile = profile / max_val

    # Step 2: Compute orthogonally normalized single-sided FFT
    fft_spectrum = np.fft.rfft(normalized_profile, norm='ortho')
    fft_modulus = np.abs(fft_spectrum)

    # Step 3: Find peaks in the FFT modulus
    peaks, _ = find_peaks(fft_modulus)

    # Edge case: No clear peak found - use half the spectrum as noise
    if len(peaks) == 0:
        noise_start_idx = len(fft_modulus) // 2
    else:
        # Step 4: Identify the first local minimum after the signal peak
        # Find the dominant peak (maximum modulus)
        signal_peak_idx = np.argmax(fft_modulus)

        # Find minima by inverting the spectrum and finding peaks
        minima, _ = find_peaks(-fft_modulus)

        # Find the first minimum after the signal peak
        minima_after_peak = minima[minima > signal_peak_idx]

        # Edge case: No minimum after peak - use point after peak
        if len(minima_after_peak) == 0:
            noise_start_idx = signal_peak_idx + 1
        else:
            noise_start_idx = minima_after_peak[0]

    # Step 5: Calculate noise as RMS of high-frequency components
    noise_components = fft_modulus[noise_start_idx:]

    # Edge case: Empty noise regime - use last quarter of spectrum
    if len(noise_components) == 0:
        noise_start_idx = 3 * len(fft_modulus) // 4
        noise_components = fft_modulus[noise_start_idx:]

        # If still empty, use the last value
        if len(noise_components) == 0:
            noise_components = np.array([fft_modulus[-1]])

    # Calculate RMS (root mean square) of noise components
    noise_level = np.sqrt(np.mean(noise_components ** 2))

    # Step 6: Edge case - Zero noise estimate
    if noise_level == 0:
        return float('inf')

    # Return CNR = 1 / noise_level, rounded to 2 decimal places
    cnr = 1.0 / noise_level
    return float(round(cnr, 2))
