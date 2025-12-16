"""
Differentially Private Logistic Regression.

Implements logistic regression with (ε,δ)-differential privacy using
objective perturbation.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

from dp_econometrics.privacy import GaussianMechanism


@dataclass
class DPLogitResults:
    """
    Results from DP Logistic Regression.

    Attributes
    ----------
    params : np.ndarray
        Coefficient estimates.
    bse : np.ndarray
        Standard errors of coefficients.
    tvalues : np.ndarray
        z-statistics (asymptotic normal).
    pvalues : np.ndarray
        Two-sided p-values.
    nobs : int
        Number of observations.
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
    epsilon_used: float
    delta_used: float
    converged: bool

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """
        Compute confidence intervals using asymptotic normal.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        np.ndarray of shape (k, 2)
            Lower and upper bounds.
        """
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = self.params - z_crit * self.bse
        ci_upper = self.params + z_crit * self.bse
        return np.column_stack([ci_lower, ci_upper])

    def summary(self) -> str:
        """Generate summary table."""
        ci = self.conf_int()
        lines = [
            "=" * 70,
            "Differentially Private Logit Results".center(70),
            "=" * 70,
            f"Observations: {self.nobs}".ljust(35) +
            f"Privacy: ε={self.epsilon_used:.3f}, δ={self.delta_used:.1e}",
            f"Converged: {self.converged}".ljust(35),
            "=" * 70,
            f"{'':>10} {'coef':>10} {'std err':>10} {'z':>10} "
            f"{'P>|z|':>10} {'[0.025':>10} {'0.975]':>10}",
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


class DPLogit:
    """
    Differentially Private Logistic Regression.

    Uses objective perturbation to achieve (ε,δ)-differential privacy
    for logistic regression.

    Parameters
    ----------
    epsilon : float
        Privacy parameter ε.
    delta : float
        Privacy parameter δ.
    bounds_X : tuple of (min, max), optional
        Bounds on feature values.
    regularization : float
        L2 regularization parameter (required for DP).

    References
    ----------
    Chaudhuri, K., Monteleoni, C., & Sarwate, A. D. (2011).
    Differentially private empirical risk minimization.
    JMLR, 12(Mar), 1069-1109.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        bounds_X: Optional[Tuple[float, float]] = None,
        regularization: float = 0.01,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if regularization <= 0:
            raise ValueError("regularization must be positive for DP")

        self.epsilon = epsilon
        self.delta = delta
        self.bounds_X = bounds_X
        self.regularization = regularization

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    def _log_loss(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        noise_vector: np.ndarray
    ) -> float:
        """
        Compute perturbed log loss.

        L(β) = -Σ[y log(σ(Xβ)) + (1-y)log(1-σ(Xβ))] + λ||β||² + noise·β
        """
        n = len(y)
        z = X @ params
        prob = self._sigmoid(z)

        # Clip for numerical stability
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        log_likelihood = y * np.log(prob) + (1 - y) * np.log(1 - prob)

        # Regularized loss + noise perturbation
        loss = (
            -np.sum(log_likelihood) / n +
            0.5 * self.regularization * np.sum(params ** 2) +
            np.dot(noise_vector, params) / n
        )

        return loss

    def _log_loss_grad(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        noise_vector: np.ndarray
    ) -> np.ndarray:
        """Gradient of perturbed log loss."""
        n = len(y)
        z = X @ params
        prob = self._sigmoid(z)

        grad = (
            X.T @ (prob - y) / n +
            self.regularization * params +
            noise_vector / n
        )

        return grad

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        add_constant: bool = True,
    ) -> DPLogitResults:
        """
        Fit DP Logistic Regression using objective perturbation.

        Parameters
        ----------
        y : np.ndarray of shape (n,)
            Binary response (0/1).
        X : np.ndarray of shape (n, k)
            Design matrix.
        add_constant : bool
            Whether to add intercept.

        Returns
        -------
        DPLogitResults
            Results with coefficients and standard errors.
        """
        y = np.asarray(y).flatten()
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, k = X.shape

        if add_constant:
            X = np.column_stack([np.ones(n), X])
            k = k + 1

        # Handle bounds
        bounds_X = self.bounds_X
        if bounds_X is None:
            warnings.warn(
                "bounds_X not provided. Computing from data leaks privacy.",
                UserWarning
            )
            bounds_X = (X.min(), X.max())

        # Clip X
        X = np.clip(X, bounds_X[0], bounds_X[1])

        # Compute sensitivity for objective perturbation
        # For logistic regression with regularization λ:
        # sensitivity = 2 / (n * λ)
        max_x_norm = np.sqrt(k) * max(abs(bounds_X[0]), abs(bounds_X[1]))
        sensitivity = 2 * max_x_norm / (n * self.regularization)

        # Generate noise for objective perturbation
        noise_sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise_vector = np.random.normal(0, noise_sigma, k)

        # Optimize perturbed objective
        x0 = np.zeros(k)
        result = minimize(
            self._log_loss,
            x0,
            args=(X, y, noise_vector),
            jac=self._log_loss_grad,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )

        params = result.x
        converged = result.success

        # Compute standard errors using Fisher information
        # Hessian of log-likelihood at optimum
        z = X @ params
        prob = self._sigmoid(z)
        W = prob * (1 - prob)

        # Fisher information (expected)
        fisher_info = X.T @ np.diag(W) @ X / n + self.regularization * np.eye(k)

        # Approximate variance including noise
        try:
            var_matrix = np.linalg.inv(fisher_info)
            # Add noise variance contribution
            noise_var_contrib = (noise_sigma ** 2 / n ** 2) * np.diag(var_matrix @ var_matrix)
            bse = np.sqrt(np.diag(var_matrix) + noise_var_contrib)
        except np.linalg.LinAlgError:
            bse = np.full(k, np.nan)

        # z-statistics and p-values
        tvalues = params / bse
        pvalues = 2 * (1 - stats.norm.cdf(np.abs(tvalues)))

        return DPLogitResults(
            params=params,
            bse=bse,
            tvalues=tvalues,
            pvalues=pvalues,
            nobs=n,
            epsilon_used=self.epsilon,
            delta_used=self.delta,
            converged=converged,
        )
