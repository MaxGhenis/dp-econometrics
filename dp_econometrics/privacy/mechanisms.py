"""
Privacy mechanisms for differential privacy.

Implements the Gaussian mechanism for (ε,δ)-differential privacy.
"""

import numpy as np
from typing import Union


def compute_gaussian_noise_scale(
    sensitivity: float,
    epsilon: float,
    delta: float
) -> float:
    """
    Compute the noise scale (σ) for the Gaussian mechanism.

    For (ε,δ)-differential privacy, the Gaussian mechanism requires:
        σ ≥ sensitivity × √(2 ln(1.25/δ)) / ε

    Parameters
    ----------
    sensitivity : float
        The L2 sensitivity of the query.
    epsilon : float
        Privacy parameter ε (must be positive).
    delta : float
        Privacy parameter δ (must be in (0, 1)).

    Returns
    -------
    float
        The required noise standard deviation σ.

    Raises
    ------
    ValueError
        If epsilon ≤ 0 or delta not in (0, 1).

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
    differential privacy. Foundations and Trends in Theoretical
    Computer Science, 9(3-4), 211-407.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon


class GaussianMechanism:
    """
    Gaussian mechanism for (ε,δ)-differential privacy.

    The Gaussian mechanism achieves (ε,δ)-differential privacy by adding
    Gaussian noise calibrated to the query's L2 sensitivity.

    Parameters
    ----------
    sensitivity : float
        The L2 sensitivity of the query.
    epsilon : float
        Privacy parameter ε.
    delta : float
        Privacy parameter δ.

    Attributes
    ----------
    sigma : float
        The computed noise standard deviation.

    Examples
    --------
    >>> mechanism = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
    >>> noisy_value = mechanism.add_noise(true_value)
    """

    def __init__(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float
    ):
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = compute_gaussian_noise_scale(sensitivity, epsilon, delta)

    def add_noise(
        self,
        value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Add Gaussian noise to a value for differential privacy.

        Parameters
        ----------
        value : float or np.ndarray
            The true value(s) to privatize.

        Returns
        -------
        float or np.ndarray
            The noisy value(s) with same shape as input.
        """
        if np.isscalar(value):
            noise = np.random.normal(0, self.sigma)
            return value + noise
        else:
            value = np.asarray(value)
            noise = np.random.normal(0, self.sigma, value.shape)
            return value + noise

    def get_privacy_guarantee(self) -> tuple:
        """
        Get the (ε, δ) privacy guarantee.

        Returns
        -------
        tuple
            (epsilon, delta) privacy parameters.
        """
        return (self.epsilon, self.delta)

    def __repr__(self) -> str:
        return (
            f"GaussianMechanism(sensitivity={self.sensitivity}, "
            f"epsilon={self.epsilon}, delta={self.delta}, sigma={self.sigma:.4f})"
        )
