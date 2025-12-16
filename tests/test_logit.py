"""
Tests for DP Logistic Regression using Objective Perturbation.

Following TDD: These tests define expected behavior.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dp_econometrics import PrivacySession
from dp_econometrics.models import DPLogit


class TestDPLogitBasic:
    """Basic functionality tests for DP Logit."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        n = 1000
        X = np.random.randn(n, 2)
        true_coef = np.array([1.0, -1.5])
        z = X @ true_coef
        prob = 1 / (1 + np.exp(-z))
        y = (np.random.rand(n) < prob).astype(float)
        return X, y, true_coef

    def test_logit_returns_coefficients(self, binary_data):
        """Logit should return coefficient estimates."""
        X, y, _ = binary_data
        session = PrivacySession(epsilon=10.0, delta=1e-5)
        result = session.logit(y, X)

        assert hasattr(result, "params")
        assert len(result.params) == X.shape[1] + 1  # +1 for intercept

    def test_logit_returns_standard_errors(self, binary_data):
        """Logit should return standard errors."""
        X, y, _ = binary_data
        session = PrivacySession(epsilon=10.0, delta=1e-5)
        result = session.logit(y, X)

        assert hasattr(result, "bse")
        assert len(result.bse) == len(result.params)
        assert all(se > 0 for se in result.bse if not np.isnan(se))

    def test_logit_returns_confidence_intervals(self, binary_data):
        """Logit should return confidence intervals."""
        X, y, _ = binary_data
        session = PrivacySession(epsilon=10.0, delta=1e-5)
        result = session.logit(y, X)

        ci = result.conf_int(alpha=0.05)
        assert ci.shape == (len(result.params), 2)

    def test_logit_returns_z_stats_and_pvalues(self, binary_data):
        """Logit should return z-statistics and p-values."""
        X, y, _ = binary_data
        session = PrivacySession(epsilon=10.0, delta=1e-5)
        result = session.logit(y, X)

        assert hasattr(result, "tvalues")  # z-stats stored as tvalues
        assert hasattr(result, "pvalues")
        assert len(result.tvalues) == len(result.params)


class TestDPLogitAccuracy:
    """Tests for accuracy of DP Logit estimates."""

    @pytest.fixture
    def logit_data(self):
        """Generate logistic regression data."""
        np.random.seed(123)
        n = 2000
        X = np.random.randn(n, 2)
        true_coef = np.array([0.5, -1.0])
        true_intercept = 0.0
        z = true_intercept + X @ true_coef
        prob = 1 / (1 + np.exp(-z))
        y = (np.random.rand(n) < prob).astype(float)
        return X, y, true_intercept, true_coef

    def test_coefficients_reasonable_high_epsilon(self, logit_data):
        """With high epsilon, coefficients should be reasonable."""
        X, y, true_intercept, true_coef = logit_data
        session = PrivacySession(epsilon=50.0, delta=1e-5)
        result = session.logit(y, X)

        # Should be in the right ballpark (within 1.0 of true)
        assert_allclose(result.params[1:], true_coef, atol=1.0)

    def test_different_runs_give_different_results(self, logit_data):
        """DP mechanism should give different results on different runs."""
        X, y, _, _ = logit_data

        np.random.seed(1)
        session1 = PrivacySession(epsilon=1.0, delta=1e-5)
        result1 = session1.logit(y, X)

        np.random.seed(2)
        session2 = PrivacySession(epsilon=1.0, delta=1e-5)
        result2 = session2.logit(y, X)

        # Results should differ due to privacy noise
        assert not np.allclose(result1.params, result2.params)

    def test_convergence_flag(self, logit_data):
        """Result should indicate convergence status."""
        X, y, _, _ = logit_data
        session = PrivacySession(epsilon=5.0, delta=1e-5)
        result = session.logit(y, X)

        assert hasattr(result, "converged")
        assert isinstance(result.converged, bool)


class TestDPLogitPrivacyBudget:
    """Tests for privacy budget tracking with logit."""

    def test_logit_consumes_budget(self):
        """Running logit should consume privacy budget."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = (np.random.rand(n) > 0.5).astype(float)

        session = PrivacySession(epsilon=1.0, delta=1e-5)
        assert session.epsilon_spent == 0.0

        session.logit(y, X)
        assert session.epsilon_spent > 0.0

    def test_multiple_logit_queries_accumulate(self):
        """Multiple logit queries should accumulate privacy cost."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = (np.random.rand(n) > 0.5).astype(float)

        session = PrivacySession(epsilon=2.0, delta=1e-5)

        session.logit(y, X)
        spent_after_1 = session.epsilon_spent

        session.logit(y, X)
        spent_after_2 = session.epsilon_spent

        assert spent_after_2 > spent_after_1


class TestDPLogitEdgeCases:
    """Edge case tests for logit."""

    def test_single_feature(self):
        """Should work with single feature."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 1)
        y = (X.flatten() > 0).astype(float)

        session = PrivacySession(epsilon=5.0, delta=1e-5)
        result = session.logit(y, X)

        assert len(result.params) == 2  # intercept + 1 coef

    def test_no_intercept(self):
        """Should support logit without intercept."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = (np.random.rand(n) > 0.5).astype(float)

        session = PrivacySession(epsilon=5.0, delta=1e-5)
        result = session.logit(y, X, add_constant=False)

        assert len(result.params) == 2  # No intercept

    def test_regularization_required(self):
        """Should require positive regularization for DP."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (np.random.rand(100) > 0.5).astype(float)

        with pytest.raises(ValueError, match="regularization"):
            DPLogit(epsilon=1.0, delta=1e-5, regularization=0)

    def test_custom_regularization(self):
        """Should accept custom regularization parameter."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = (np.random.rand(n) > 0.5).astype(float)

        session = PrivacySession(epsilon=5.0, delta=1e-5)
        result = session.logit(y, X, regularization=0.1)

        assert hasattr(result, "params")


class TestDPLogitSummary:
    """Tests for summary output."""

    def test_summary_string(self):
        """Should produce formatted summary."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = (np.random.rand(n) > 0.5).astype(float)

        session = PrivacySession(epsilon=5.0, delta=1e-5)
        result = session.logit(y, X)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Logit" in summary
        assert "Privacy" in summary
