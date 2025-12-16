"""
Tests for privacy accounting and mechanisms.

Following TDD: These tests are written BEFORE the implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dp_econometrics.privacy import (
    GaussianMechanism,
    PrivacyAccountant,
    compute_gaussian_noise_scale,
)


class TestGaussianMechanism:
    """Tests for Gaussian mechanism."""

    def test_noise_scale_computed_correctly(self):
        """Noise scale should follow analytic Gaussian mechanism formula."""
        # σ >= sensitivity * sqrt(2 * log(1.25 / δ)) / ε
        sensitivity = 1.0
        epsilon = 1.0
        delta = 1e-5

        expected_sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        computed_sigma = compute_gaussian_noise_scale(sensitivity, epsilon, delta)

        assert_allclose(computed_sigma, expected_sigma, rtol=1e-10)

    def test_noise_scale_increases_with_lower_epsilon(self):
        """Lower epsilon should require more noise."""
        sensitivity = 1.0
        delta = 1e-5

        sigma_high_eps = compute_gaussian_noise_scale(sensitivity, 2.0, delta)
        sigma_low_eps = compute_gaussian_noise_scale(sensitivity, 0.5, delta)

        assert sigma_low_eps > sigma_high_eps

    def test_noise_scale_increases_with_higher_sensitivity(self):
        """Higher sensitivity should require more noise."""
        epsilon = 1.0
        delta = 1e-5

        sigma_low_sens = compute_gaussian_noise_scale(0.5, epsilon, delta)
        sigma_high_sens = compute_gaussian_noise_scale(2.0, epsilon, delta)

        assert sigma_high_sens > sigma_low_sens

    def test_mechanism_adds_correct_noise(self):
        """Mechanism should add noise with correct scale."""
        mechanism = GaussianMechanism(
            sensitivity=1.0,
            epsilon=1.0,
            delta=1e-5
        )

        # Add noise to zero many times and check variance
        n_samples = 10000
        noisy_values = [mechanism.add_noise(0.0) for _ in range(n_samples)]

        empirical_std = np.std(noisy_values)
        expected_std = mechanism.sigma

        # Should be close (within ~5% for 10k samples)
        assert_allclose(empirical_std, expected_std, rtol=0.1)

    def test_mechanism_preserves_shape(self):
        """Mechanism should preserve input shape."""
        mechanism = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)

        # Scalar
        result_scalar = mechanism.add_noise(5.0)
        assert np.isscalar(result_scalar) or result_scalar.shape == ()

        # 1D array
        result_1d = mechanism.add_noise(np.array([1, 2, 3]))
        assert result_1d.shape == (3,)

        # 2D array
        result_2d = mechanism.add_noise(np.array([[1, 2], [3, 4]]))
        assert result_2d.shape == (2, 2)


class TestPrivacyAccountant:
    """Tests for privacy budget accounting."""

    def test_initial_state(self):
        """Accountant should start with zero spent."""
        accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-5)

        assert accountant.epsilon_budget == 1.0
        assert accountant.delta_budget == 1e-5
        assert accountant.epsilon_spent == 0.0
        assert accountant.queries == 0

    def test_spend_budget(self):
        """Should track budget spending."""
        accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-5)

        accountant.spend(epsilon=0.3, delta=1e-6)

        assert accountant.epsilon_spent == 0.3
        assert accountant.queries == 1

    def test_remaining_budget(self):
        """Should correctly compute remaining budget."""
        accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-5)

        accountant.spend(epsilon=0.3, delta=1e-6)

        assert accountant.epsilon_remaining == 0.7
        assert_allclose(accountant.epsilon_remaining,
                        accountant.epsilon_budget - accountant.epsilon_spent)

    def test_multiple_queries_compose(self):
        """Multiple queries should compose additively (basic composition)."""
        accountant = PrivacyAccountant(epsilon_budget=2.0, delta_budget=1e-4)

        accountant.spend(epsilon=0.5, delta=1e-6)
        accountant.spend(epsilon=0.3, delta=1e-6)
        accountant.spend(epsilon=0.2, delta=1e-6)

        assert accountant.epsilon_spent == 1.0
        assert accountant.queries == 3

    def test_budget_exceeded_raises(self):
        """Should raise when budget exceeded."""
        accountant = PrivacyAccountant(epsilon_budget=0.5, delta_budget=1e-5)

        with pytest.raises(ValueError, match="[Bb]udget|[Ee]xceeded"):
            accountant.spend(epsilon=0.6, delta=1e-6)

    def test_can_afford_check(self):
        """Should correctly check if query is affordable."""
        accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-5)

        assert accountant.can_afford(epsilon=0.5, delta=1e-6)
        assert accountant.can_afford(epsilon=1.0, delta=1e-5)
        assert not accountant.can_afford(epsilon=1.5, delta=1e-6)

    def test_delta_budget_tracked(self):
        """Should track delta spending."""
        accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-4)

        accountant.spend(epsilon=0.1, delta=1e-5)
        accountant.spend(epsilon=0.1, delta=2e-5)

        assert accountant.delta_spent == 3e-5


class TestPrivacyAccountantAdvanced:
    """Advanced composition tests."""

    def test_rdp_composition_tighter(self):
        """RDP composition should give tighter bounds than basic composition."""
        # With RDP, composing many queries is more efficient
        accountant_rdp = PrivacyAccountant(
            epsilon_budget=10.0,
            delta_budget=1e-5,
            composition="rdp"
        )
        accountant_basic = PrivacyAccountant(
            epsilon_budget=10.0,
            delta_budget=1e-5,
            composition="basic"
        )

        # Run 100 small queries
        for _ in range(100):
            accountant_rdp.spend(epsilon=0.1, delta=0)
            accountant_basic.spend(epsilon=0.1, delta=0)

        # RDP should have spent less (tighter accounting)
        # Note: Basic is just sum, RDP uses moments
        assert accountant_rdp.epsilon_spent <= accountant_basic.epsilon_spent

    def test_query_history_tracked(self):
        """Should maintain history of queries."""
        accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-5)

        accountant.spend(epsilon=0.2, delta=1e-6, query_name="ols_1")
        accountant.spend(epsilon=0.3, delta=1e-6, query_name="ols_2")

        history = accountant.get_history()
        assert len(history) == 2
        assert history[0]["query_name"] == "ols_1"
        assert history[1]["epsilon"] == 0.3


class TestSensitivityComputation:
    """Tests for sensitivity computation for sufficient statistics."""

    def test_xtx_sensitivity_with_bounds(self):
        """XᵀX sensitivity should be computed from bounds."""
        from dp_econometrics.privacy import compute_xtx_sensitivity

        bounds_X = (-1, 1)  # Each feature in [-1, 1]
        n_features = 3

        sensitivity = compute_xtx_sensitivity(bounds_X, n_features)

        # Max single row contribution to XᵀX is bounded by ||x||² * ||x||²
        # For x in [-1,1]^k, ||x||² ≤ k, so contribution ≤ k²
        assert sensitivity > 0
        assert sensitivity <= n_features ** 2

    def test_xty_sensitivity_with_bounds(self):
        """Xᵀy sensitivity should be computed from bounds."""
        from dp_econometrics.privacy import compute_xty_sensitivity

        bounds_X = (-1, 1)
        bounds_y = (-10, 10)
        n_features = 3

        sensitivity = compute_xty_sensitivity(bounds_X, bounds_y, n_features)

        # Max contribution is ||x|| * |y| ≤ sqrt(k) * max_y
        assert sensitivity > 0
        assert sensitivity <= np.sqrt(n_features) * 10


class TestNoisySufficientStats:
    """Tests for noisy sufficient statistics computation."""

    def test_noisy_xtx_shape(self):
        """Noisy XᵀX should have correct shape."""
        from dp_econometrics.privacy import compute_noisy_xtx

        np.random.seed(42)
        n, k = 100, 3
        X = np.random.randn(n, k)

        noisy_xtx = compute_noisy_xtx(
            X,
            epsilon=1.0,
            delta=1e-5,
            bounds_X=(-5, 5)
        )

        assert noisy_xtx.shape == (k, k)

    def test_noisy_xtx_symmetric(self):
        """Noisy XᵀX should be symmetric."""
        from dp_econometrics.privacy import compute_noisy_xtx

        np.random.seed(42)
        n, k = 100, 3
        X = np.random.randn(n, k)

        noisy_xtx = compute_noisy_xtx(
            X,
            epsilon=1.0,
            delta=1e-5,
            bounds_X=(-5, 5)
        )

        assert_allclose(noisy_xtx, noisy_xtx.T)

    def test_noisy_xty_shape(self):
        """Noisy Xᵀy should have correct shape."""
        from dp_econometrics.privacy import compute_noisy_xty

        np.random.seed(42)
        n, k = 100, 3
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        noisy_xty = compute_noisy_xty(
            X, y,
            epsilon=1.0,
            delta=1e-5,
            bounds_X=(-5, 5),
            bounds_y=(-10, 10)
        )

        assert noisy_xty.shape == (k,)

    def test_noise_is_actually_added(self):
        """Should add noise (results differ across runs)."""
        from dp_econometrics.privacy import compute_noisy_xtx

        n, k = 100, 3
        X = np.random.randn(n, k)

        np.random.seed(1)
        noisy_xtx_1 = compute_noisy_xtx(X, epsilon=1.0, delta=1e-5, bounds_X=(-5, 5))

        np.random.seed(2)
        noisy_xtx_2 = compute_noisy_xtx(X, epsilon=1.0, delta=1e-5, bounds_X=(-5, 5))

        assert not np.allclose(noisy_xtx_1, noisy_xtx_2)
