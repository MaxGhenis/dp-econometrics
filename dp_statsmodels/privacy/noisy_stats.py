"""
Noisy sufficient statistics for differentially private regression.

Implements the NSS approach: compute X'X and X'y, then add calibrated
Gaussian noise for (ε,δ)-differential privacy.
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings

from .mechanisms import GaussianMechanism
from .sensitivity import (
    compute_xtx_sensitivity,
    compute_xty_sensitivity,
    compute_yty_sensitivity,
    compute_n_sensitivity,
)


def _get_rng(
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.random.Generator:
    """Get a numpy random Generator from various inputs."""
    if random_state is None:
        return np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        return random_state
    else:
        return np.random.default_rng(random_state)


def compute_noisy_xtx(
    X: np.ndarray,
    epsilon: float,
    delta: float,
    bounds_X: Optional[Tuple[float, float]] = None,
    clip: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    require_bounds: bool = True,
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
        Bounds on feature values. Required for proper DP guarantees.
    clip : bool
        Whether to clip X to bounds before computing.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.
    require_bounds : bool
        If True (default), raise error when bounds not provided.
        Set to False only for testing/development.

    Returns
    -------
    np.ndarray of shape (k, k)
        Noisy X'X matrix (symmetric).

    Raises
    ------
    ValueError
        If bounds_X is None and require_bounds is True.
    """
    n, k = X.shape
    rng = _get_rng(random_state)

    # Handle bounds
    if bounds_X is None:
        if require_bounds:
            raise ValueError(
                "bounds_X is required for differential privacy guarantees. "
                "Computing bounds from data completely breaks privacy. "
                "Set require_bounds=False only for testing/development."
            )
        else:
            warnings.warn(
                "No bounds_X provided. Computing bounds from data leaks privacy. "
                "This mode should only be used for testing/development.",
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

    # Add noise to symmetric matrix correctly:
    # Generate noise for upper triangle (including diagonal) only,
    # then mirror to lower triangle. This ensures each unique entry
    # gets independent N(0, σ²) noise with the correct variance.
    noise_matrix = np.zeros((k, k))
    # Number of unique entries in upper triangle (including diagonal)
    n_unique = k * (k + 1) // 2
    noise_values = rng.normal(0, mechanism.sigma, n_unique)

    idx = 0
    for i in range(k):
        for j in range(i, k):
            noise_matrix[i, j] = noise_values[idx]
            noise_matrix[j, i] = noise_values[idx]  # Mirror to lower triangle
            idx += 1

    noisy_xtx = xtx + noise_matrix

    return noisy_xtx


def compute_noisy_xty(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    delta: float,
    bounds_X: Optional[Tuple[float, float]] = None,
    bounds_y: Optional[Tuple[float, float]] = None,
    clip: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    require_bounds: bool = True,
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
        Bounds on feature values. Required for proper DP guarantees.
    bounds_y : tuple of (min, max), optional
        Bounds on response variable. Required for proper DP guarantees.
    clip : bool
        Whether to clip data to bounds.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.
    require_bounds : bool
        If True (default), raise error when bounds not provided.

    Returns
    -------
    np.ndarray of shape (k,)
        Noisy X'y vector.

    Raises
    ------
    ValueError
        If bounds_X or bounds_y is None and require_bounds is True.
    """
    n, k = X.shape
    rng = _get_rng(random_state)

    # Handle bounds
    if bounds_X is None:
        if require_bounds:
            raise ValueError(
                "bounds_X is required for differential privacy guarantees."
            )
        else:
            warnings.warn(
                "No bounds_X provided. Computing from data leaks privacy.",
                UserWarning
            )
            bounds_X = (X.min(), X.max())

    if bounds_y is None:
        if require_bounds:
            raise ValueError(
                "bounds_y is required for differential privacy guarantees."
            )
        else:
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
    noise = rng.normal(0, mechanism.sigma, k)

    noisy_xty = xty + noise

    return noisy_xty


def compute_noisy_yty(
    y: np.ndarray,
    epsilon: float,
    delta: float,
    bounds_y: Optional[Tuple[float, float]] = None,
    clip: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    require_bounds: bool = True,
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
        Bounds on y. Required for proper DP guarantees.
    clip : bool
        Whether to clip.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.
    require_bounds : bool
        If True (default), raise error when bounds not provided.

    Returns
    -------
    float
        Noisy y'y.
    """
    rng = _get_rng(random_state)

    if bounds_y is None:
        if require_bounds:
            raise ValueError(
                "bounds_y is required for differential privacy guarantees."
            )
        else:
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

    noisy_yty = float(yty) + rng.normal(0, mechanism.sigma)

    return noisy_yty


def compute_noisy_n(
    n: int,
    epsilon: float,
    delta: float,
    random_state: Optional[Union[int, np.random.Generator]] = None,
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
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    float
        Noisy count.
    """
    rng = _get_rng(random_state)
    sensitivity = compute_n_sensitivity()
    mechanism = GaussianMechanism(sensitivity, epsilon, delta)

    return float(n) + rng.normal(0, mechanism.sigma)


def compute_all_noisy_stats(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    delta: float,
    bounds_X: Optional[Tuple[float, float]] = None,
    bounds_y: Optional[Tuple[float, float]] = None,
    epsilon_split: Optional[dict] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    require_bounds: bool = True,
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
        Bounds on X. Required for proper DP guarantees.
    bounds_y : tuple, optional
        Bounds on y. Required for proper DP guarantees.
    epsilon_split : dict, optional
        How to split epsilon. Default is equal split.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.
    require_bounds : bool
        If True (default), raise error when bounds not provided.

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
    rng = _get_rng(random_state)

    return {
        'xtx': compute_noisy_xtx(
            X, epsilon * epsilon_split['xtx'], delta_per_stat,
            bounds_X, random_state=rng, require_bounds=require_bounds
        ),
        'xty': compute_noisy_xty(
            X, y, epsilon * epsilon_split['xty'], delta_per_stat,
            bounds_X, bounds_y, random_state=rng, require_bounds=require_bounds
        ),
        'yty': compute_noisy_yty(
            y, epsilon * epsilon_split['yty'], delta_per_stat,
            bounds_y, random_state=rng, require_bounds=require_bounds
        ),
        'n': compute_noisy_n(
            n, epsilon * epsilon_split['n'], delta_per_stat,
            random_state=rng
        )
    }
