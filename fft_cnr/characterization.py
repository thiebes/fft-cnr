"""
Validation and characterization framework for the FFT-based CNR estimator.

This module provides tools to assess the accuracy and precision of the CNR
estimation algorithm across different signal-to-noise levels.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .estimator import estimate_cnr


def generate_gaussian_profile(
    center: float = 0.0,
    sigma: float = 0.424661,
    amplitude: float = 1.0,
    num_points: int = 100,
    x_min: float = -5.0,
    x_max: float = 5.0,
) -> np.ndarray:
    """
    Generate a 1D Gaussian profile.

    Parameters
    ----------
    center : float, optional
        Center position of the Gaussian peak, by default 0.0
    sigma : float, optional
        Standard deviation of the Gaussian (FWHM=1 when sigma=0.424661), by default 0.424661
    amplitude : float, optional
        Peak amplitude of the Gaussian, by default 1.0
    num_points : int, optional
        Number of points in the profile, by default 100
    x_min : float, optional
        Minimum x-coordinate, by default -5.0
    x_max : float, optional
        Maximum x-coordinate, by default 5.0

    Returns
    -------
    np.ndarray
        1D array containing the Gaussian profile

    Examples
    --------
    >>> profile = generate_gaussian_profile(center=0.5, sigma=0.1, num_points=100)
    >>> len(profile)
    100
    >>> profile.max()
    1.0
    """
    x = np.linspace(x_min, x_max, num_points)
    profile = amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return profile


def add_calibrated_noise(
    signal: np.ndarray,
    cnr_nominal: float,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add white Gaussian noise to a signal to achieve a target CNR.

    The nominal CNR is defined as the ratio of the signal's peak amplitude
    to the RMS (root mean square) of the added noise:
    CNR_nominal = peak_amplitude / noise_rms

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    cnr_nominal : float
        Target contrast-to-noise ratio
    rng_seed : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    np.ndarray
        Signal with added noise

    Raises
    ------
    ValueError
        If cnr_nominal is non-positive
    ValueError
        If signal has zero amplitude

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
    >>> noisy = add_calibrated_noise(signal, cnr_nominal=10.0, rng_seed=42)
    >>> len(noisy)
    100
    """
    if cnr_nominal <= 0:
        raise ValueError("CNR nominal must be positive")

    # Get signal amplitude (peak value)
    signal_amplitude = np.max(np.abs(signal))

    if signal_amplitude == 0:
        raise ValueError("Signal has zero amplitude")

    # Calculate required noise RMS to achieve target CNR
    # CNR = signal_amplitude / noise_rms
    # Therefore: noise_rms = signal_amplitude / CNR
    noise_rms = signal_amplitude / cnr_nominal

    # Generate white Gaussian noise with the calculated RMS
    rng = np.random.default_rng(rng_seed)
    noise = noise_rms * rng.standard_normal(len(signal))

    return signal + noise


def characterize_cnr_level(
    cnr_nominal: float,
    num_trials: int = 100,
    num_points: int = 100,
    signal_params: Optional[Dict] = None,
    rng_seed: Optional[int] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Characterize CNR estimator performance at a single nominal CNR level.

    Generates multiple noisy realizations of a signal with the specified
    nominal CNR, estimates the CNR for each, and computes statistics on
    the estimation accuracy.

    Parameters
    ----------
    cnr_nominal : float
        Target CNR level to test
    num_trials : int, optional
        Number of independent trials to run, by default 100
    num_points : int, optional
        Number of points in the signal profile, by default 100
    signal_params : dict, optional
        Parameters for generate_gaussian_profile(), by default None
        (uses default Gaussian parameters)
    rng_seed : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    dict
        Dictionary containing:
        - 'cnr_nominal': The input nominal CNR
        - 'cnr_estimates': Array of all CNR estimates
        - 'cnr_ratios': Array of CNR_est / CNR_nom ratios
        - 'mean_ratio': Mean of the ratios
        - 'median_ratio': Median of the ratios
        - 'mode_ratio': Mode of the ratios (via KDE)
        - 'std_ratio': Standard deviation of the ratios
        - 'bias': Mean(CNR_est - CNR_nom)
        - 'rmse': Root mean squared error
        - 'percentile_5': 5th percentile of ratios
        - 'percentile_95': 95th percentile of ratios

    Examples
    --------
    >>> results = characterize_cnr_level(cnr_nominal=10.0, num_trials=50)
    >>> results['mean_ratio']  # Should be close to 1.0 for unbiased estimator
    """
    # Set default signal parameters if not provided
    if signal_params is None:
        signal_params = {}

    # Generate base signal once
    signal = generate_gaussian_profile(num_points=num_points, **signal_params)

    # Run trials
    cnr_estimates = np.zeros(num_trials)
    rng = np.random.default_rng(rng_seed)

    for i in range(num_trials):
        # Generate noisy signal with a unique seed
        trial_seed = None if rng_seed is None else rng.integers(0, 2**31)
        noisy_signal = add_calibrated_noise(signal, cnr_nominal, rng_seed=trial_seed)

        # Estimate CNR
        cnr_estimates[i] = estimate_cnr(noisy_signal)

    # Calculate ratios
    cnr_ratios = cnr_estimates / cnr_nominal

    # Calculate statistics
    mean_ratio = np.mean(cnr_ratios)
    median_ratio = np.median(cnr_ratios)
    std_ratio = np.std(cnr_ratios, ddof=1)

    # Calculate mode using kernel density estimation
    try:
        kde = stats.gaussian_kde(cnr_ratios)
        x_range = np.linspace(cnr_ratios.min(), cnr_ratios.max(), 1000)
        kde_values = kde(x_range)
        mode_ratio = x_range[np.argmax(kde_values)]
    except (np.linalg.LinAlgError, ValueError):
        # Fallback to median if KDE fails
        mode_ratio = median_ratio

    # Additional metrics
    bias = np.mean(cnr_estimates - cnr_nominal)
    rmse = np.sqrt(np.mean((cnr_estimates - cnr_nominal) ** 2))
    percentile_5 = np.percentile(cnr_ratios, 5)
    percentile_95 = np.percentile(cnr_ratios, 95)

    return {
        'cnr_nominal': cnr_nominal,
        'cnr_estimates': cnr_estimates,
        'cnr_ratios': cnr_ratios,
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'mode_ratio': mode_ratio,
        'std_ratio': std_ratio,
        'bias': bias,
        'rmse': rmse,
        'percentile_5': percentile_5,
        'percentile_95': percentile_95,
    }


def run_characterization_study(
    cnr_range: Tuple[float, float] = (1.0, 100.0),
    num_cnr_levels: int = 20,
    num_trials_per_level: int = 100,
    num_points: int = 100,
    signal_params: Optional[Dict] = None,
    use_log_spacing: bool = True,
    rng_seed: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a comprehensive characterization study across multiple CNR levels.

    Parameters
    ----------
    cnr_range : tuple of float, optional
        (min_cnr, max_cnr) range to test, by default (1.0, 100.0)
    num_cnr_levels : int, optional
        Number of CNR levels to test, by default 20
    num_trials_per_level : int, optional
        Number of trials at each CNR level, by default 100
    num_points : int, optional
        Number of points in signal profiles, by default 100
    signal_params : dict, optional
        Parameters for generate_gaussian_profile(), by default None
    use_log_spacing : bool, optional
        Use logarithmic spacing for CNR levels, by default True
    rng_seed : int, optional
        Random seed for reproducibility, by default None
    verbose : bool, optional
        Print progress messages, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - cnr_nominal: Nominal CNR values tested
        - mean_ratio: Mean of CNR_est / CNR_nom
        - median_ratio: Median of CNR_est / CNR_nom
        - mode_ratio: Mode of CNR_est / CNR_nom
        - std_ratio: Standard deviation of ratios
        - bias: Mean estimation bias
        - rmse: Root mean squared error
        - percentile_5: 5th percentile of ratios
        - percentile_95: 95th percentile of ratios

    Examples
    --------
    >>> results = run_characterization_study(
    ...     cnr_range=(5.0, 50.0),
    ...     num_cnr_levels=10,
    ...     num_trials_per_level=50
    ... )
    >>> print(results.head())
    """
    # Generate CNR levels to test
    if use_log_spacing:
        cnr_levels = np.logspace(
            np.log10(cnr_range[0]),
            np.log10(cnr_range[1]),
            num_cnr_levels
        )
    else:
        cnr_levels = np.linspace(cnr_range[0], cnr_range[1], num_cnr_levels)

    # Initialize results storage
    results = []

    # Run characterization for each CNR level
    for i, cnr_nom in enumerate(cnr_levels):
        if verbose:
            print(f"Testing CNR level {i+1}/{num_cnr_levels}: {cnr_nom:.2f}")

        # Get statistics for this CNR level
        level_results = characterize_cnr_level(
            cnr_nominal=cnr_nom,
            num_trials=num_trials_per_level,
            num_points=num_points,
            signal_params=signal_params,
            rng_seed=rng_seed,
        )

        # Store summary statistics (not raw arrays)
        results.append({
            'cnr_nominal': level_results['cnr_nominal'],
            'mean_ratio': level_results['mean_ratio'],
            'median_ratio': level_results['median_ratio'],
            'mode_ratio': level_results['mode_ratio'],
            'std_ratio': level_results['std_ratio'],
            'bias': level_results['bias'],
            'rmse': level_results['rmse'],
            'percentile_5': level_results['percentile_5'],
            'percentile_95': level_results['percentile_95'],
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if verbose:
        print(f"\nCharacterization study complete!")
        print(f"Tested {num_cnr_levels} CNR levels with {num_trials_per_level} trials each")

    return df


def plot_validation_summary(
    results: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 10),
    dpi: int = 150,
) -> Figure:
    """
    Create a comprehensive validation summary plot.

    Generates a 2x2 grid showing:
    - Mean CNR_est / CNR_nom vs CNR_nom
    - Median CNR_est / CNR_nom vs CNR_nom
    - Mode CNR_est / CNR_nom vs CNR_nom
    - Standard deviation of ratios vs CNR_nom

    Parameters
    ----------
    results : pd.DataFrame
        Results from run_characterization_study()
    output_path : str, optional
        Path to save the figure, by default None (display only)
    figsize : tuple of float, optional
        Figure size (width, height) in inches, by default (12, 10)
    dpi : int, optional
        Resolution in dots per inch, by default 150

    Returns
    -------
    Figure
        Matplotlib Figure object

    Examples
    --------
    >>> results = run_characterization_study()
    >>> fig = plot_validation_summary(results, output_path='validation.png')
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle('CNR Estimator Validation Summary', fontsize=16, fontweight='bold')

    cnr_nom = results['cnr_nominal']

    # Plot 1: Mean ratio
    ax = axes[0, 0]
    ax.plot(cnr_nom, results['mean_ratio'], 'o-', label='Mean ratio', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Ideal (ratio=1)')
    ax.fill_between(
        cnr_nom,
        results['mean_ratio'] - results['std_ratio'],
        results['mean_ratio'] + results['std_ratio'],
        alpha=0.3,
        label='±1 std'
    )
    ax.set_xscale('log')
    ax.set_xlabel('Nominal CNR')
    ax.set_ylabel('Mean(CNR_est / CNR_nom)')
    ax.set_title('Accuracy: Mean Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Median ratio
    ax = axes[0, 1]
    ax.plot(cnr_nom, results['median_ratio'], 's-', label='Median ratio',
            color='green', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Ideal (ratio=1)')
    ax.fill_between(
        cnr_nom,
        results['percentile_5'],
        results['percentile_95'],
        alpha=0.3,
        color='green',
        label='5-95 percentile'
    )
    ax.set_xscale('log')
    ax.set_xlabel('Nominal CNR')
    ax.set_ylabel('Median(CNR_est / CNR_nom)')
    ax.set_title('Accuracy: Median Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Mode ratio
    ax = axes[1, 0]
    ax.plot(cnr_nom, results['mode_ratio'], '^-', label='Mode ratio',
            color='purple', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Ideal (ratio=1)')
    ax.set_xscale('log')
    ax.set_xlabel('Nominal CNR')
    ax.set_ylabel('Mode(CNR_est / CNR_nom)')
    ax.set_title('Accuracy: Mode Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Standard deviation
    ax = axes[1, 1]
    ax.plot(cnr_nom, results['std_ratio'], 'd-', label='Std(ratio)',
            color='orange', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Nominal CNR')
    ax.set_ylabel('Std(CNR_est / CNR_nom)')
    ax.set_title('Precision: Standard Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Validation plot saved to: {output_path}")

    return fig


def plot_bias_and_rmse(
    results: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    dpi: int = 150,
) -> Figure:
    """
    Plot estimation bias and RMSE vs nominal CNR.

    Parameters
    ----------
    results : pd.DataFrame
        Results from run_characterization_study()
    output_path : str, optional
        Path to save the figure, by default None (display only)
    figsize : tuple of float, optional
        Figure size (width, height) in inches, by default (12, 5)
    dpi : int, optional
        Resolution in dots per inch, by default 150

    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.suptitle('CNR Estimator Bias and Error Analysis', fontsize=14, fontweight='bold')

    cnr_nom = results['cnr_nominal']

    # Plot 1: Bias
    ax = axes[0]
    ax.plot(cnr_nom, results['bias'], 'o-', linewidth=2, color='red')
    ax.axhline(y=0, color='k', linestyle='--', label='Zero bias')
    ax.set_xscale('log')
    ax.set_xlabel('Nominal CNR')
    ax.set_ylabel('Bias (Mean[CNR_est - CNR_nom])')
    ax.set_title('Estimation Bias')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: RMSE
    ax = axes[1]
    ax.plot(cnr_nom, results['rmse'], 's-', linewidth=2, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Nominal CNR')
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Squared Error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Bias/RMSE plot saved to: {output_path}")

    return fig
