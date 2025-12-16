"""
Noisy sufficient statistics for differentially private regression.

Implements the NSS approach: compute X'X and X'y, then add calibrated
Gaussian noise for (ε,δ)-differential privacy.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

from .mechanisms import GaussianMechanism
from .sensitivity import (
    compute_xtx_sensitivity,
    compute_xty_sensitivity,
    compute_yty_sensitivity,
    compute_n_sensitivity,
)


def compute_noisy_xtx(
    X: np.ndarray,
    epsilon: float,
    delta: float,
    bounds_X: Optional[Tuple[float, float]] = None,
    clip: bool = True
) -> np.ndarray:
    """
    Compute X'X with Gaussian noise for differential privacy.

    Parameters
    ----------
    X : np.ndarray of shape (n, k)
        Design matrix.
    epsilon : float
        Privacy parameter ε for this statistic.
    delta : float
        Privacy parameter δ.
    bounds_X : tuple of (min, max), optional
        Bounds on feature values. If None, computed from data (privacy leak!).
    clip : bool
        Whether to clip X to bounds before computing.

    Returns
    -------
    np.ndarray of shape (k, k)
        Noisy X'X matrix (symmetric).
    """
    n, k = X.shape

    # Handle bounds
    if bounds_X is None:
        warnings.warn(
            "No bounds_X provided. Computing bounds from data leaks privacy. "
            "Provide bounds_X for proper differential privacy.",
            UserWarning
        )
        bounds_X = (X.min(), X.max())

    # Clip data to bounds
    if clip:
        X = np.clip(X, bounds_X[0], bounds_X[1])

    # Compute true X'X
    xtx = X.T @ X

    # Compute sensitivity
    sensitivity = compute_xtx_sensitivity(bounds_X, k)

    # Create mechanism and add noise
    mechanism = GaussianMechanism(sensitivity, epsilon, delta)

    # Add noise to upper triangle, then symmetrize
    noise = np.random.normal(0, mechanism.sigma, (k, k))
    noise_symmetric = (noise + noise.T) / 2

    noisy_xtx = xtx + noise_symmetric

    return noisy_xtx


def compute_noisy_xty(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    delta: float,
    bounds_X: Optional[Tuple[float, float]] = None,
    bounds_y: Optional[Tuple[float, float]] = None,
    clip: bool = True
) -> np.ndarray:
    """
    Compute X'y with Gaussian noise for differential privacy.

    Parameters
    ----------
    X : np.ndarray of shape (n, k)
        Design matrix.
    y : np.ndarray of shape (n,)
        Response variable.
    epsilon : float
        Privacy parameter ε for this statistic.
    delta : float
        Privacy parameter δ.
    bounds_X : tuple of (min, max), optional
        Bounds on feature values.
    bounds_y : tuple of (min, max), optional
        Bounds on response variable.
    clip : bool
        Whether to clip data to bounds.

    Returns
    -------
    np.ndarray of shape (k,)
        Noisy X'y vector.
    """
    n, k = X.shape

    # Handle bounds
    if bounds_X is None:
        warnings.warn(
            "No bounds_X provided. Computing from data leaks privacy.",
            UserWarning
        )
        bounds_X = (X.min(), X.max())

    if bounds_y is None:
        warnings.warn(
            "No bounds_y provided. Computing from data leaks privacy.",
            UserWarning
        )
        bounds_y = (y.min(), y.max())

    # Clip data
    if clip:
        X = np.clip(X, bounds_X[0], bounds_X[1])
        y = np.clip(y, bounds_y[0], bounds_y[1])

    # Compute true X'y
    xty = X.T @ y

    # Compute sensitivity
    sensitivity = compute_xty_sensitivity(bounds_X, bounds_y, k)

    # Create mechanism and add noise
    mechanism = GaussianMechanism(sensitivity, epsilon, delta)
    noise = np.random.normal(0, mechanism.sigma, k)

    noisy_xty = xty + noise

    return noisy_xty


def compute_noisy_yty(
    y: np.ndarray,
    epsilon: float,
    delta: float,
    bounds_y: Optional[Tuple[float, float]] = None,
    clip: bool = True
) -> float:
    """
    Compute y'y with Gaussian noise for differential privacy.

    Parameters
    ----------
    y : np.ndarray
        Response variable.
    epsilon : float
        Privacy parameter.
    delta : float
        Privacy parameter.
    bounds_y : tuple, optional
        Bounds on y.
    clip : bool
        Whether to clip.

    Returns
    -------
    float
        Noisy y'y.
    """
    if bounds_y is None:
        warnings.warn(
            "No bounds_y provided. Computing from data leaks privacy.",
            UserWarning
        )
        bounds_y = (y.min(), y.max())

    if clip:
        y = np.clip(y, bounds_y[0], bounds_y[1])

    yty = y @ y

    sensitivity = compute_yty_sensitivity(bounds_y)
    mechanism = GaussianMechanism(sensitivity, epsilon, delta)

    noisy_yty = mechanism.add_noise(yty)

    return noisy_yty


def compute_noisy_n(
    n: int,
    epsilon: float,
    delta: float
) -> float:
    """
    Compute noisy count for differential privacy.

    Parameters
    ----------
    n : int
        True count.
    epsilon : float
        Privacy parameter.
    delta : float
        Privacy parameter.

    Returns
    -------
    float
        Noisy count.
    """
    sensitivity = compute_n_sensitivity()
    mechanism = GaussianMechanism(sensitivity, epsilon, delta)

    return mechanism.add_noise(float(n))


def compute_all_noisy_stats(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    delta: float,
    bounds_X: Optional[Tuple[float, float]] = None,
    bounds_y: Optional[Tuple[float, float]] = None,
    epsilon_split: Optional[dict] = None
) -> dict:
    """
    Compute all noisy sufficient statistics for OLS.

    Splits the privacy budget across X'X, X'y, y'y, and n.

    Parameters
    ----------
    X : np.ndarray
        Design matrix.
    y : np.ndarray
        Response.
    epsilon : float
        Total epsilon budget.
    delta : float
        Total delta budget.
    bounds_X : tuple, optional
        Bounds on X.
    bounds_y : tuple, optional
        Bounds on y.
    epsilon_split : dict, optional
        How to split epsilon. Default is equal split.

    Returns
    -------
    dict
        Dictionary with 'xtx', 'xty', 'yty', 'n' (all noisy).
    """
    # Default: equal split across 4 statistics
    if epsilon_split is None:
        epsilon_split = {
            'xtx': 0.4,
            'xty': 0.4,
            'yty': 0.1,
            'n': 0.1
        }

    # Each statistic uses a fraction of total delta too (by composition)
    delta_per_stat = delta / 4

    n = len(y)

    return {
        'xtx': compute_noisy_xtx(
            X, epsilon * epsilon_split['xtx'], delta_per_stat,
            bounds_X
        ),
        'xty': compute_noisy_xty(
            X, y, epsilon * epsilon_split['xty'], delta_per_stat,
            bounds_X, bounds_y
        ),
        'yty': compute_noisy_yty(
            y, epsilon * epsilon_split['yty'], delta_per_stat,
            bounds_y
        ),
        'n': compute_noisy_n(
            n, epsilon * epsilon_split['n'], delta_per_stat
        )
    }
