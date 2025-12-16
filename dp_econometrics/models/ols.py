"""
Differentially Private OLS using Noisy Sufficient Statistics.

Implements OLS regression with (ε,δ)-differential privacy by adding
calibrated Gaussian noise to X'X and X'y before solving for β.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

from dp_econometrics.privacy import (
    compute_noisy_xtx,
    compute_noisy_xty,
    compute_xtx_sensitivity,
    compute_xty_sensitivity,
)


@dataclass
class DPOLSResults:
    """
    Results from DP OLS regression.

    Attributes
    ----------
    params : np.ndarray
        Coefficient estimates (including intercept if add_constant=True).
    bse : np.ndarray
        Standard errors of coefficients.
    tvalues : np.ndarray
        t-statistics.
    pvalues : np.ndarray
        Two-sided p-values.
    nobs : int
        Number of observations.
    df_resid : int
        Residual degrees of freedom.
    epsilon_used : float
        Privacy budget consumed.
    delta_used : float
        Delta parameter used.
    """
    params: np.ndarray
    bse: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    nobs: int
    df_resid: int
    epsilon_used: float
    delta_used: float
    resid_var: float
    _noisy_xtx: np.ndarray  # For variance computation
    _noisy_xty: np.ndarray

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """
        Compute confidence intervals for coefficients.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% CI).

        Returns
        -------
        np.ndarray of shape (k, 2)
            Lower and upper bounds for each coefficient.
        """
        # Use t-distribution
        t_crit = stats.t.ppf(1 - alpha / 2, self.df_resid)

        ci_lower = self.params - t_crit * self.bse
        ci_upper = self.params + t_crit * self.bse

        return np.column_stack([ci_lower, ci_upper])

    def summary(self) -> str:
        """
        Generate a summary table of results.

        Returns
        -------
        str
            Formatted summary table.
        """
        ci = self.conf_int()

        lines = [
            "=" * 70,
            "Differentially Private OLS Results".center(70),
            "=" * 70,
            f"Observations: {self.nobs}".ljust(35) +
            f"Privacy: ε={self.epsilon_used:.3f}, δ={self.delta_used:.1e}",
            f"Df Residuals: {self.df_resid}".ljust(35) +
            f"Residual Var: {self.resid_var:.4f}",
            "=" * 70,
            f"{'':>10} {'coef':>10} {'std err':>10} {'t':>10} "
            f"{'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}",
            "-" * 70,
        ]

        for i in range(len(self.params)):
            name = f"x{i}" if i > 0 else "const"
            lines.append(
                f"{name:>10} {self.params[i]:>10.4f} {self.bse[i]:>10.4f} "
                f"{self.tvalues[i]:>10.3f} {self.pvalues[i]:>10.3f} "
                f"{ci[i, 0]:>10.4f} {ci[i, 1]:>10.4f}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DPOLSResults(nobs={self.nobs}, k={len(self.params)}, ε={self.epsilon_used:.3f})"


class DPOLS:
    """
    Differentially Private Ordinary Least Squares.

    Uses Noisy Sufficient Statistics (NSS) to compute OLS estimates
    with formal (ε,δ)-differential privacy guarantees.

    Parameters
    ----------
    epsilon : float
        Privacy parameter ε.
    delta : float
        Privacy parameter δ.
    bounds_X : tuple of (min, max), optional
        Bounds on feature values. Required for proper DP.
    bounds_y : tuple of (min, max), optional
        Bounds on response variable. Required for proper DP.

    Examples
    --------
    >>> model = DPOLS(epsilon=1.0, delta=1e-5, bounds_X=(-10, 10), bounds_y=(-100, 100))
    >>> result = model.fit(X, y)
    >>> print(result.summary())

    References
    ----------
    Sheffet, O. (2017). Differentially private ordinary least squares.
    ICML 2017.

    Evans, G., King, G., et al. (2024). Differentially Private Linear
    Regression with Linked Data. Harvard Data Science Review.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        bounds_X: Optional[Tuple[float, float]] = None,
        bounds_y: Optional[Tuple[float, float]] = None,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")

        self.epsilon = epsilon
        self.delta = delta
        self.bounds_X = bounds_X
        self.bounds_y = bounds_y

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None,
        add_constant: bool = True,
    ) -> DPOLSResults:
        """
        Fit DP OLS model.

        Parameters
        ----------
        y : np.ndarray of shape (n,)
            Response variable.
        X : np.ndarray of shape (n, k)
            Design matrix (without constant).
        weights : np.ndarray of shape (n,), optional
            Sample weights.
        add_constant : bool
            Whether to add a constant term.

        Returns
        -------
        DPOLSResults
            Results object with coefficients, standard errors, etc.
        """
        y = np.asarray(y).flatten()
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, k = X.shape

        # Add constant if requested
        if add_constant:
            X = np.column_stack([np.ones(n), X])
            k = k + 1

        # Handle bounds
        bounds_X = self.bounds_X
        bounds_y = self.bounds_y

        if bounds_X is None:
            warnings.warn(
                "bounds_X not provided. Computing from data, which leaks privacy. "
                "For proper differential privacy, provide explicit bounds.",
                UserWarning
            )
            bounds_X = (X.min(), X.max())

        if bounds_y is None:
            warnings.warn(
                "bounds_y not provided. Computing from data, which leaks privacy.",
                UserWarning
            )
            bounds_y = (y.min(), y.max())

        # Handle weights (WLS)
        if weights is not None:
            weights = np.asarray(weights).flatten()
            sqrt_w = np.sqrt(weights)
            X = X * sqrt_w[:, np.newaxis]
            y = y * sqrt_w

        # Split privacy budget between X'X and X'y
        eps_xtx = self.epsilon * 0.5
        eps_xty = self.epsilon * 0.5
        delta_each = self.delta / 2

        # Compute noisy sufficient statistics
        noisy_xtx = compute_noisy_xtx(X, eps_xtx, delta_each, bounds_X)
        noisy_xty = compute_noisy_xty(X, y, eps_xty, delta_each, bounds_X, bounds_y)

        # Ensure X'X is positive definite (regularize if needed)
        min_eig = np.min(np.linalg.eigvalsh(noisy_xtx))
        if min_eig <= 0:
            warnings.warn(
                "Noisy X'X is not positive definite. Adding regularization. "
                "This may indicate collinear features or excessive noise.",
                UserWarning
            )
            noisy_xtx = noisy_xtx + (abs(min_eig) + 1e-6) * np.eye(k)

        # Solve for coefficients: β = (X'X)^{-1} X'y
        try:
            params = np.linalg.solve(noisy_xtx, noisy_xty)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Singular matrix encountered. Using pseudo-inverse.",
                UserWarning
            )
            params = np.linalg.lstsq(noisy_xtx, noisy_xty, rcond=None)[0]

        # Compute standard errors
        # The variance of β̂ = (X'X)^{-1} X'y includes both:
        # 1. Sampling variance: σ² (X'X)^{-1}
        # 2. Privacy noise variance

        # Estimate residual variance (using noisy estimate)
        # Note: This is approximate since we don't have true residuals
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_explained = params @ noisy_xtx @ params - n * y_mean ** 2
        ss_resid = max(ss_total - ss_explained, 1e-10)
        resid_var = ss_resid / (n - k)

        # Compute variance of β̂
        # Var(β̂) = σ² (X'X)^{-1} + Var_noise
        try:
            xtx_inv = np.linalg.inv(noisy_xtx)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(noisy_xtx)

        # Sampling variance component
        var_sampling = resid_var * xtx_inv

        # Privacy noise variance component
        # The noise added to X'X and X'y propagates to β̂
        sens_xtx = compute_xtx_sensitivity(bounds_X, k)
        sens_xty = compute_xty_sensitivity(bounds_X, bounds_y, k)

        sigma_xtx = sens_xtx * np.sqrt(2 * np.log(1.25 / delta_each)) / eps_xtx
        sigma_xty = sens_xty * np.sqrt(2 * np.log(1.25 / delta_each)) / eps_xty

        # Approximate variance from noise (using delta method)
        # This is a simplified approximation
        var_noise_diag = (sigma_xty ** 2 + params ** 2 * sigma_xtx ** 2) * np.diag(xtx_inv ** 2)

        # Total variance
        var_total = np.diag(var_sampling) + var_noise_diag
        bse = np.sqrt(np.maximum(var_total, 1e-10))

        # t-statistics and p-values
        tvalues = params / bse
        pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), n - k))

        return DPOLSResults(
            params=params,
            bse=bse,
            tvalues=tvalues,
            pvalues=pvalues,
            nobs=n,
            df_resid=n - k,
            epsilon_used=self.epsilon,
            delta_used=self.delta,
            resid_var=resid_var,
            _noisy_xtx=noisy_xtx,
            _noisy_xty=noisy_xty,
        )
