"""
Tests for privacy mechanisms to increase coverage.
"""
import numpy as np
import pytest

from dp_statsmodels.privacy.mechanisms import (
    compute_gaussian_noise_scale,
    GaussianMechanism,
)


class TestGaussianNoiseScale:
    """Tests for compute_gaussian_noise_scale."""

    def test_basic_computation(self):
        """Should compute noise scale."""
        sigma = compute_gaussian_noise_scale(1.0, 1.0, 1e-5)
        assert sigma > 0

    def test_invalid_epsilon(self):
        """Should raise for non-positive epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            compute_gaussian_noise_scale(1.0, 0.0, 1e-5)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            compute_gaussian_noise_scale(1.0, -1.0, 1e-5)

    def test_invalid_delta(self):
        """Should raise for invalid delta."""
        with pytest.raises(ValueError, match="delta must be in"):
            compute_gaussian_noise_scale(1.0, 1.0, 0.0)

        with pytest.raises(ValueError, match="delta must be in"):
            compute_gaussian_noise_scale(1.0, 1.0, 1.0)

        with pytest.raises(ValueError, match="delta must be in"):
            compute_gaussian_noise_scale(1.0, 1.0, -0.1)


class TestGaussianMechanism:
    """Tests for GaussianMechanism class."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert mech.sensitivity == 1.0
        assert mech.epsilon == 1.0
        assert mech.delta == 1e-5
        assert mech.sigma > 0

    def test_add_noise_scalar(self):
        """Should add noise to scalar values."""
        np.random.seed(42)
        mech = GaussianMechanism(sensitivity=1.0, epsilon=10.0, delta=1e-5)
        noisy = mech.add_noise(100.0)
        assert isinstance(noisy, float)
        # With high epsilon, should be close
        assert abs(noisy - 100.0) < 10

    def test_add_noise_array(self):
        """Should add noise to array values."""
        np.random.seed(42)
        mech = GaussianMechanism(sensitivity=1.0, epsilon=10.0, delta=1e-5)
        values = np.array([1.0, 2.0, 3.0])
        noisy = mech.add_noise(values)
        assert noisy.shape == values.shape

    def test_get_privacy_guarantee(self):
        """Should return privacy parameters."""
        mech = GaussianMechanism(sensitivity=1.0, epsilon=2.0, delta=1e-6)
        eps, delta = mech.get_privacy_guarantee()
        assert eps == 2.0
        assert delta == 1e-6

    def test_repr(self):
        """Should have string representation."""
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        repr_str = repr(mech)
        assert "GaussianMechanism" in repr_str
        assert "sensitivity=1.0" in repr_str
        assert "epsilon=1.0" in repr_str
