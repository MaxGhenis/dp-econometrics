"""
Sensitivity computation for differential privacy.

Computes the sensitivity of sufficient statistics for linear regression,
which determines how much noise is needed for privacy.
"""

import numpy as np
from typing import Tuple, Union


def compute_xtx_sensitivity(
    bounds_X: Tuple[float, float],
    n_features: int
) -> float:
    """
    Compute the L2 sensitivity of X'X.

    The sensitivity measures the maximum change in X'X when a single
    row is added or removed. For bounded data, this depends on the
    maximum possible row contribution.

    Parameters
    ----------
    bounds_X : tuple of (min, max)
        Bounds on each feature value.
    n_features : int
        Number of features (columns in X).

    Returns
    -------
    float
        The L2 sensitivity of X'X.

    Notes
    -----
    For a single row x with ||x||_∞ ≤ B, the contribution to X'X is xx'.
    The Frobenius norm of xx' is ||x||₂².
    With each entry bounded by B, ||x||₂² ≤ k × B², so the sensitivity
    is at most k × B² for each entry, and the matrix has k² entries.

    For Gaussian mechanism, we use the L2 sensitivity (Frobenius norm).
    """
    x_min, x_max = bounds_X
    max_abs = max(abs(x_min), abs(x_max))

    # Maximum ||x||₂ for a row
    max_row_norm_sq = n_features * max_abs ** 2

    # Sensitivity of X'X in Frobenius norm
    # ||x x'||_F = ||x||₂²
    # But we're adding noise to each entry, so we need per-entry sensitivity
    # Each entry (X'X)_{ij} = sum_k x_{ki} x_{kj}
    # Adding/removing one row changes it by at most x_i x_j ≤ B²

    # For the whole matrix, L2 sensitivity is ||x x'||_F = ||x||₂²
    sensitivity = max_row_norm_sq

    return sensitivity


def compute_xty_sensitivity(
    bounds_X: Tuple[float, float],
    bounds_y: Tuple[float, float],
    n_features: int
) -> float:
    """
    Compute the L2 sensitivity of X'y.

    Parameters
    ----------
    bounds_X : tuple of (min, max)
        Bounds on each feature value.
    bounds_y : tuple of (min, max)
        Bounds on the response variable.
    n_features : int
        Number of features.

    Returns
    -------
    float
        The L2 sensitivity of X'y.

    Notes
    -----
    For a single observation (x, y), the contribution to X'y is x*y.
    The L2 norm of this is ||x||₂ × |y|.
    """
    x_min, x_max = bounds_X
    y_min, y_max = bounds_y

    max_abs_x = max(abs(x_min), abs(x_max))
    max_abs_y = max(abs(y_min), abs(y_max))

    # Maximum ||x||₂ for a row
    max_row_norm = np.sqrt(n_features) * max_abs_x

    # Sensitivity of X'y
    sensitivity = max_row_norm * max_abs_y

    return sensitivity


def compute_yty_sensitivity(bounds_y: Tuple[float, float]) -> float:
    """
    Compute the sensitivity of y'y (sum of squared y).

    Parameters
    ----------
    bounds_y : tuple of (min, max)
        Bounds on the response variable.

    Returns
    -------
    float
        The sensitivity of y'y.
    """
    y_min, y_max = bounds_y
    max_abs_y = max(abs(y_min), abs(y_max))

    # Adding/removing one observation changes y'y by at most y²
    return max_abs_y ** 2


def compute_n_sensitivity() -> float:
    """
    Compute the sensitivity of the count n.

    The count changes by 1 when adding/removing a single record.

    Returns
    -------
    float
        Always returns 1.
    """
    return 1.0
