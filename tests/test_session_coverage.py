"""
Additional tests for Session to increase coverage.
"""
import numpy as np
import pytest

from dp_statsmodels import Session


class TestSessionProperties:
    """Tests for Session properties."""

    def test_delta_spent(self):
        """Should track delta spent."""
        session = Session(epsilon=1.0, delta=1e-5)
        assert session.delta_spent == 0.0

        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        session.OLS(y, X)

        assert session.delta_spent > 0

    def test_queries_count(self):
        """Should count queries."""
        session = Session(epsilon=1.0, delta=1e-5)
        assert session.queries == 0

        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        session.OLS(y, X)
        assert session.queries == 1

        session.OLS(y, X)
        assert session.queries == 2


class TestSessionBudgetAllocation:
    """Tests for budget allocation."""

    def test_allocation_with_fraction(self):
        """Should allocate budget with fraction parameter."""
        session = Session(epsilon=1.0, delta=1e-5)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        # Use specific epsilon
        session.OLS(y, X, epsilon=0.5)
        assert 0.4 < session.epsilon_spent < 0.6


class TestSessionAliases:
    """Tests for backwards compatibility aliases."""

    def test_logit_alias(self):
        """logit() should work like Logit()."""
        np.random.seed(42)
        session = Session(epsilon=2.0, delta=1e-5)
        X = np.random.randn(100, 2)
        y = (np.random.rand(100) > 0.5).astype(float)

        result = session.logit(y, X)
        assert result.params is not None

    def test_fe_alias(self):
        """fe() should work like PanelOLS()."""
        np.random.seed(42)
        session = Session(epsilon=5.0, delta=1e-5)
        n_entities, n_periods = 20, 3
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        result = session.fe(y, X, groups)
        assert result.params is not None

    def test_summary(self):
        """summary() should return string."""
        session = Session(epsilon=1.0, delta=1e-5)
        summary = session.summary()
        assert isinstance(summary, str)
        assert "budget" in summary.lower() or "epsilon" in summary.lower()

    def test_get_history_alias(self):
        """get_history() should return query history."""
        session = Session(epsilon=1.0, delta=1e-5)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        session.OLS(y, X)
        history = session.get_history()
        assert isinstance(history, list)
        assert len(history) == 1

    def test_repr(self):
        """__repr__ should return string representation."""
        session = Session(epsilon=1.0, delta=1e-5)
        repr_str = repr(session)
        assert "Session" in repr_str
        assert "ε=1.0" in repr_str


class TestLogitModel:
    """Tests for Logit model via session."""

    def test_logit_budget_exhausted(self):
        """Should raise when budget exhausted for Logit."""
        np.random.seed(42)
        session = Session(epsilon=0.1, delta=1e-5)
        X = np.random.randn(100, 2)
        y = (np.random.rand(100) > 0.5).astype(float)

        with pytest.raises(ValueError, match="[Bb]udget"):
            session.Logit(y, X, epsilon=0.2)

    def test_logit_with_explicit_bounds(self):
        """Logit should use explicit bounds when provided."""
        np.random.seed(42)
        session = Session(epsilon=2.0, delta=1e-5, bounds_X=(-10, 10))
        X = np.random.randn(100, 2)
        y = (np.random.rand(100) > 0.5).astype(float)

        # Provide explicit bounds (different from session default)
        result = session.Logit(y, X, bounds_X=(-5, 5))
        assert result.params is not None

    def test_logit_uses_session_bounds(self):
        """Logit should use session bounds when not provided."""
        np.random.seed(42)
        session = Session(epsilon=2.0, delta=1e-5, bounds_X=(-5, 5))
        X = np.random.randn(100, 2)
        y = (np.random.rand(100) > 0.5).astype(float)

        # Don't provide bounds - should use session default
        result = session.Logit(y, X)
        assert result.params is not None


class TestPanelOLSEdgeCases:
    """Tests for PanelOLS edge cases."""

    def test_panelols_no_entity_effects(self):
        """PanelOLS without entity_effects should run regular OLS."""
        np.random.seed(42)
        session = Session(epsilon=5.0, delta=1e-5)
        n_entities, n_periods = 20, 3
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        result = session.PanelOLS(y, X, groups=groups, entity_effects=False)
        assert result.params is not None

    def test_panelols_no_groups_error(self):
        """PanelOLS without groups should raise error."""
        np.random.seed(42)
        session = Session(epsilon=5.0, delta=1e-5)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="groups"):
            session.PanelOLS(y, X, groups=None)

    def test_panelols_budget_exhausted(self):
        """Should raise when PanelOLS budget exhausted."""
        np.random.seed(42)
        session = Session(epsilon=0.1, delta=1e-5)
        n_entities, n_periods = 20, 3
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        with pytest.raises(ValueError, match="[Bb]udget"):
            session.PanelOLS(y, X, groups=groups, epsilon=0.2)

    def test_panelols_uses_session_bounds(self):
        """PanelOLS should use session default bounds."""
        np.random.seed(42)
        session = Session(epsilon=5.0, delta=1e-5,
                         bounds_X=(-5, 5), bounds_y=(-20, 20))
        n_entities, n_periods = 20, 3
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        # Should not warn since session has bounds
        result = session.PanelOLS(y, X, groups=groups)
        assert result.params is not None

    def test_panelols_with_explicit_bounds(self):
        """PanelOLS should use explicit bounds when provided."""
        np.random.seed(42)
        session = Session(epsilon=5.0, delta=1e-5,
                         bounds_X=(-10, 10), bounds_y=(-50, 50))
        n_entities, n_periods = 20, 3
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        # Provide explicit bounds (different from session default)
        result = session.PanelOLS(y, X, groups=groups,
                                  bounds_X=(-5, 5), bounds_y=(-20, 20))
        assert result.params is not None


class TestPrivacyAccountantRepr:
    """Tests for PrivacyAccountant __repr__."""

    def test_accountant_repr(self):
        """PrivacyAccountant should have repr."""
        from dp_statsmodels.privacy.accounting import PrivacyAccountant
        accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-5)
        repr_str = repr(accountant)
        assert "PrivacyAccountant" in repr_str
        assert "budget" in repr_str


class TestBudgetAllocationFraction:
    """Tests for session budget allocation with fraction parameter."""

    def test_allocation_with_fraction_internal(self):
        """Should allocate budget using fraction parameter."""
        session = Session(epsilon=1.0, delta=1e-5)
        # Access internal method directly
        eps, delta = session._allocate_budget(fraction=0.5)
        assert 0.4 < eps < 0.6  # Should be ~0.5 of 1.0


class TestRDPComposition:
    """Tests for RDP composition edge cases."""

    def test_rdp_with_orders_leq_1(self):
        """RDP should handle orders <= 1."""
        from dp_statsmodels.privacy.accounting import PrivacyAccountant
        # RDP composition with small queries
        accountant = PrivacyAccountant(
            epsilon_budget=10.0,
            delta_budget=1e-5,
            composition="rdp"
        )
        # Make many small queries to exercise RDP accounting
        for _ in range(10):
            accountant.spend(epsilon=0.1, delta=0)
        # Should compute epsilon_spent using RDP
        assert accountant.epsilon_spent > 0

    def test_rdp_spend_exercises_add_rdp(self):
        """RDP spend should call _add_rdp."""
        from dp_statsmodels.privacy.accounting import PrivacyAccountant
        # Use RDP composition and spend
        accountant = PrivacyAccountant(
            epsilon_budget=10.0,
            delta_budget=1e-5,
            composition="rdp"
        )
        accountant.spend(epsilon=0.5, delta=1e-6)
        # RDP accounting should be active
        assert accountant.epsilon_spent > 0
        assert accountant._rdp_spent is not None
