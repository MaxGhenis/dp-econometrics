"""
Additional tests for model classes to increase coverage.
"""
import numpy as np
import pytest

from dp_statsmodels.models.ols import DPOLS
from dp_statsmodels.models.logit import DPLogit
from dp_statsmodels.models.fixed_effects import DPFixedEffects


class TestDPOLSEdgeCases:
    """Tests for DPOLS edge cases."""

    def test_invalid_epsilon(self):
        """Should raise for non-positive epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DPOLS(epsilon=0.0, delta=1e-5)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            DPOLS(epsilon=-1.0, delta=1e-5)

    def test_invalid_delta(self):
        """Should raise for invalid delta."""
        with pytest.raises(ValueError, match="delta must be in"):
            DPOLS(epsilon=1.0, delta=0.0)

        with pytest.raises(ValueError, match="delta must be in"):
            DPOLS(epsilon=1.0, delta=1.0)

    def test_1d_input(self):
        """Should handle 1D X input."""
        np.random.seed(42)
        model = DPOLS(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))
        X = np.random.randn(100)  # 1D
        y = 2 * X + np.random.randn(100)

        result = model.fit(y, X)
        assert result.params is not None
        assert len(result.params) == 2  # const + 1 coef

    def test_singular_matrix_handling(self):
        """Should handle near-singular matrices."""
        np.random.seed(42)
        model = DPOLS(epsilon=0.5, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))
        # Create collinear data
        X1 = np.random.randn(50)
        X = np.column_stack([X1, X1 * 1.0001])  # Nearly collinear
        y = X1 + np.random.randn(50)

        result = model.fit(y, X)
        assert result.params is not None


class TestDPLogitEdgeCases:
    """Tests for DPLogit edge cases."""

    def test_invalid_epsilon(self):
        """Should raise for non-positive epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DPLogit(epsilon=0.0, delta=1e-5)

    def test_invalid_delta(self):
        """Should raise for invalid delta."""
        with pytest.raises(ValueError, match="delta must be in"):
            DPLogit(epsilon=1.0, delta=0.0)

    def test_1d_input(self):
        """Should handle 1D X input."""
        np.random.seed(42)
        model = DPLogit(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5))
        X = np.random.randn(100)  # 1D
        y = (X > 0).astype(float)

        result = model.fit(y, X)
        assert result.params is not None

    def test_summary(self):
        """Should produce summary."""
        np.random.seed(42)
        model = DPLogit(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5))
        X = np.random.randn(100, 2)
        y = (np.random.rand(100) > 0.5).astype(float)

        result = model.fit(y, X)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Logit" in summary


class TestDPFixedEffectsEdgeCases:
    """Tests for DPFixedEffects edge cases."""

    def test_invalid_epsilon(self):
        """Should raise for non-positive epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DPFixedEffects(epsilon=0.0, delta=1e-5)

    def test_invalid_delta(self):
        """Should raise for invalid delta."""
        with pytest.raises(ValueError, match="delta must be in"):
            DPFixedEffects(epsilon=1.0, delta=0.0)

    def test_insufficient_df(self):
        """Should raise when insufficient degrees of freedom."""
        np.random.seed(42)
        model = DPFixedEffects(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))

        # Too few observations relative to groups and features
        n_entities, n_periods = 5, 2
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 6)  # 6 features + 5 groups > 10 obs
        y = np.random.randn(n)

        with pytest.raises(ValueError, match="[Dd]egrees of freedom"):
            model.fit(y, X, groups)

    def test_1d_input(self):
        """Should handle 1D X input."""
        np.random.seed(42)
        model = DPFixedEffects(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))

        n_entities, n_periods = 20, 5
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n)  # 1D
        y = np.random.randn(n)

        result = model.fit(y, X, groups)
        assert result.params is not None

    def test_repr(self):
        """Should have string representation."""
        np.random.seed(42)
        model = DPFixedEffects(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))

        n_entities, n_periods = 20, 5
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        result = model.fit(y, X, groups)
        repr_str = repr(result)
        assert "FixedEffects" in repr_str or "nobs" in repr_str.lower()


class TestResultsRepr:
    """Tests for Results __repr__ methods."""

    def test_dpols_results_repr(self):
        """DPOLSResults should have repr."""
        np.random.seed(42)
        model = DPOLS(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        result = model.fit(y, X)

        repr_str = repr(result)
        assert "DPOLSResults" in repr_str
        assert "nobs" in repr_str

    def test_dplogit_results_repr(self):
        """DPLogitResults should have repr."""
        np.random.seed(42)
        model = DPLogit(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5))
        X = np.random.randn(100, 2)
        y = (np.random.rand(100) > 0.5).astype(float)
        result = model.fit(y, X)

        repr_str = repr(result)
        assert "Logit" in repr_str or "nobs" in repr_str.lower()

    def test_dpfixedeffects_results_repr(self):
        """DPFixedEffectsResults should have repr."""
        np.random.seed(42)
        model = DPFixedEffects(epsilon=10.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))
        n_entities, n_periods = 20, 5
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)
        result = model.fit(y, X, groups)

        repr_str = repr(result)
        assert "nobs" in repr_str.lower() or "Result" in repr_str


class TestPrivacyAccountingEdgeCases:
    """Tests for privacy accounting edge cases."""

    def test_invalid_epsilon_budget(self):
        """Should raise for invalid epsilon budget."""
        from dp_statsmodels.privacy.accounting import PrivacyAccountant

        with pytest.raises(ValueError):
            PrivacyAccountant(epsilon_budget=0.0, delta_budget=1e-5)

    def test_invalid_delta_budget(self):
        """Should raise for invalid delta budget."""
        from dp_statsmodels.privacy.accounting import PrivacyAccountant

        with pytest.raises(ValueError):
            PrivacyAccountant(epsilon_budget=1.0, delta_budget=0.0)

    def test_cannot_afford_large_query(self):
        """Should return False when query exceeds budget."""
        from dp_statsmodels.privacy.accounting import PrivacyAccountant

        accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-5)
        assert not accountant.can_afford(2.0, 1e-5)


class TestLinAlgErrorHandling:
    """Tests for LinAlgError handling in models."""

    def test_ols_singular_solve(self):
        """OLS should handle singular matrix in solve."""
        from unittest.mock import patch
        np.random.seed(42)
        model = DPOLS(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        # Mock linalg.solve to raise LinAlgError
        with patch('numpy.linalg.solve', side_effect=np.linalg.LinAlgError):
            with pytest.warns(UserWarning, match="[Ss]ingular"):
                result = model.fit(y, X)
                assert result.params is not None

    def test_ols_singular_inv(self):
        """OLS should handle singular matrix in inverse."""
        from unittest.mock import patch
        np.random.seed(42)
        model = DPOLS(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        # Mock linalg.inv to raise LinAlgError (after solve succeeds)
        original_solve = np.linalg.solve
        call_count = [0]

        def mock_inv(a):
            raise np.linalg.LinAlgError("Singular matrix")

        with patch('numpy.linalg.inv', mock_inv):
            result = model.fit(y, X)
            assert result.params is not None

    def test_fixed_effects_singular_solve(self):
        """Fixed effects should handle singular matrix in solve."""
        from unittest.mock import patch
        np.random.seed(42)
        model = DPFixedEffects(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))
        n_entities, n_periods = 20, 5
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        with patch('numpy.linalg.solve', side_effect=np.linalg.LinAlgError):
            with pytest.warns(UserWarning, match="[Ss]ingular"):
                result = model.fit(y, X, groups)
                assert result.params is not None

    def test_fixed_effects_singular_inv(self):
        """Fixed effects should handle singular matrix in inverse."""
        from unittest.mock import patch
        np.random.seed(42)
        model = DPFixedEffects(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-20, 20))
        n_entities, n_periods = 20, 5
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        with patch('numpy.linalg.inv', side_effect=np.linalg.LinAlgError):
            result = model.fit(y, X, groups)
            assert result.params is not None

    def test_logit_singular_inv(self):
        """Logit should handle singular Fisher info matrix."""
        from unittest.mock import patch
        np.random.seed(42)
        model = DPLogit(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5))
        X = np.random.randn(100, 2)
        y = (np.random.rand(100) > 0.5).astype(float)

        with patch('numpy.linalg.inv', side_effect=np.linalg.LinAlgError):
            result = model.fit(y, X)
            # bse should be NaN when Fisher info is singular
            assert np.all(np.isnan(result.bse))
