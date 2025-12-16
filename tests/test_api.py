"""
Tests for the dp_statsmodels API design.

TDD: These tests define the target API before implementation.
"""

import numpy as np
import pytest


class TestImportPatterns:
    """Test that import patterns work like statsmodels."""

    def test_import_api_module(self):
        """Should be able to import api module."""
        import dp_statsmodels.api as sm_dp
        assert hasattr(sm_dp, 'Session')
        assert hasattr(sm_dp, 'OLS')
        assert hasattr(sm_dp, 'Logit')
        assert hasattr(sm_dp, 'PanelOLS')

    def test_import_main_package(self):
        """Should be able to import from main package."""
        from dp_statsmodels import Session, OLS, Logit, PanelOLS
        assert Session is not None
        assert OLS is not None

    def test_version_available(self):
        """Package version should be accessible."""
        import dp_statsmodels
        assert hasattr(dp_statsmodels, '__version__')


class TestSessionAPI:
    """Test Session-based API (primary interface for validation servers)."""

    @pytest.fixture
    def regression_data(self):
        """Simple regression data."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = X @ [1, 2] + np.random.randn(n)
        return X, y

    @pytest.fixture
    def binary_data(self):
        """Binary outcome data."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        z = X @ [1, -1]
        y = (1 / (1 + np.exp(-z)) > np.random.rand(n)).astype(float)
        return X, y

    @pytest.fixture
    def panel_data(self):
        """Panel data with entity effects."""
        np.random.seed(42)
        n_entities, n_periods = 50, 10
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        alpha = np.repeat(np.random.randn(n_entities), n_periods)
        y = alpha + X @ [1, 2] + np.random.randn(n) * 0.5
        return X, y, groups

    def test_session_creation(self):
        """Session should be created with privacy budget."""
        import dp_statsmodels.api as sm_dp

        session = sm_dp.Session(epsilon=1.0, delta=1e-5)
        assert session.epsilon == 1.0
        assert session.delta == 1e-5
        assert session.epsilon_spent == 0.0

    def test_session_with_bounds(self):
        """Session should accept default bounds."""
        import dp_statsmodels.api as sm_dp

        session = sm_dp.Session(
            epsilon=1.0,
            delta=1e-5,
            bounds_X=(-10, 10),
            bounds_y=(-100, 100)
        )
        assert session.bounds_X == (-10, 10)
        assert session.bounds_y == (-100, 100)

    def test_session_ols_capitalized(self, regression_data):
        """session.OLS() should work (capitalized like statsmodels)."""
        import dp_statsmodels.api as sm_dp
        X, y = regression_data

        session = sm_dp.Session(epsilon=5.0, delta=1e-5)
        result = session.OLS(y, X)

        assert hasattr(result, 'params')
        assert hasattr(result, 'bse')
        assert hasattr(result, 'summary')

    def test_session_logit_capitalized(self, binary_data):
        """session.Logit() should work."""
        import dp_statsmodels.api as sm_dp
        X, y = binary_data

        session = sm_dp.Session(epsilon=5.0, delta=1e-5)
        result = session.Logit(y, X)

        assert hasattr(result, 'params')
        assert hasattr(result, 'bse')

    def test_session_panel_ols(self, panel_data):
        """session.PanelOLS() should work for fixed effects."""
        import dp_statsmodels.api as sm_dp
        X, y, groups = panel_data

        session = sm_dp.Session(epsilon=5.0, delta=1e-5)
        result = session.PanelOLS(y, X, groups=groups, entity_effects=True)

        assert hasattr(result, 'params')
        assert hasattr(result, 'n_groups')

    def test_session_tracks_budget(self, regression_data):
        """Session should track privacy budget across queries."""
        import dp_statsmodels.api as sm_dp
        X, y = regression_data

        session = sm_dp.Session(epsilon=1.0, delta=1e-5)

        session.OLS(y, X, epsilon=0.3)
        assert session.epsilon_spent == pytest.approx(0.3, rel=0.01)

        session.OLS(y, X, epsilon=0.2)
        assert session.epsilon_spent == pytest.approx(0.5, rel=0.01)

    def test_session_context_manager(self, regression_data):
        """Session should work as context manager."""
        import dp_statsmodels.api as sm_dp
        X, y = regression_data

        with sm_dp.Session(epsilon=1.0, delta=1e-5) as session:
            result = session.OLS(y, X)
            assert result is not None

    def test_session_history(self, regression_data, binary_data):
        """Session should track query history."""
        import dp_statsmodels.api as sm_dp
        X, y = regression_data
        X_bin, y_bin = binary_data

        session = sm_dp.Session(epsilon=2.0, delta=1e-5)
        session.OLS(y, X)
        session.Logit(y_bin, X_bin)

        history = session.history()
        assert len(history) == 2
        assert history[0]['query_type'] == 'OLS'
        assert history[1]['query_type'] == 'Logit'


class TestStandaloneModelAPI:
    """Test standalone model classes (like statsmodels OLS(y, X).fit())."""

    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = X @ [1, 2] + np.random.randn(n)
        return X, y

    def test_ols_standalone(self, simple_data):
        """OLS should work standalone without session."""
        import dp_statsmodels.api as sm_dp
        X, y = simple_data

        # Current API: OLS(epsilon=...).fit(y, X)
        model = sm_dp.OLS(epsilon=1.0, delta=1e-5)
        result = model.fit(y, X)

        assert hasattr(result, 'params')
        assert hasattr(result, 'bse')
        assert hasattr(result, 'summary')

    def test_ols_with_bounds(self, simple_data):
        """OLS should accept bounds."""
        import dp_statsmodels.api as sm_dp
        X, y = simple_data

        model = sm_dp.OLS(
            epsilon=1.0,
            delta=1e-5,
            bounds_X=(-5, 5),
            bounds_y=(-20, 20)
        )
        result = model.fit(y, X)
        assert result is not None

    def test_logit_standalone(self):
        """Logit should work standalone."""
        import dp_statsmodels.api as sm_dp

        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = (np.random.rand(n) > 0.5).astype(float)

        model = sm_dp.Logit(epsilon=1.0, delta=1e-5)
        result = model.fit(y, X)

        assert hasattr(result, 'params')

    def test_panel_ols_standalone(self):
        """PanelOLS should work standalone."""
        import dp_statsmodels.api as sm_dp

        np.random.seed(42)
        n_entities, n_periods = 30, 5
        n = n_entities * n_periods
        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        model = sm_dp.PanelOLS(
            epsilon=1.0,
            delta=1e-5
        )
        result = model.fit(y, X, groups)

        assert hasattr(result, 'params')
        assert hasattr(result, 'n_groups')


class TestResultsInterface:
    """Test that results match statsmodels interface."""

    @pytest.fixture
    def fitted_result(self):
        import dp_statsmodels.api as sm_dp
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = X @ [1, 2] + np.random.randn(n)

        session = sm_dp.Session(epsilon=5.0, delta=1e-5)
        return session.OLS(y, X)

    def test_result_params(self, fitted_result):
        """Result should have params attribute."""
        assert hasattr(fitted_result, 'params')
        assert isinstance(fitted_result.params, np.ndarray)

    def test_result_bse(self, fitted_result):
        """Result should have bse (standard errors)."""
        assert hasattr(fitted_result, 'bse')
        assert len(fitted_result.bse) == len(fitted_result.params)

    def test_result_tvalues(self, fitted_result):
        """Result should have tvalues."""
        assert hasattr(fitted_result, 'tvalues')

    def test_result_pvalues(self, fitted_result):
        """Result should have pvalues."""
        assert hasattr(fitted_result, 'pvalues')
        assert all(0 <= p <= 1 for p in fitted_result.pvalues)

    def test_result_conf_int(self, fitted_result):
        """Result should have conf_int method."""
        ci = fitted_result.conf_int()
        assert ci.shape == (len(fitted_result.params), 2)
        assert all(ci[:, 0] < ci[:, 1])

    def test_result_summary(self, fitted_result):
        """Result should have summary method returning string."""
        summary = fitted_result.summary()
        assert isinstance(summary, str)
        assert 'coef' in summary.lower() or 'Coef' in summary

    def test_result_nobs(self, fitted_result):
        """Result should have nobs."""
        assert hasattr(fitted_result, 'nobs')
        assert fitted_result.nobs == 500

    def test_result_privacy_info(self, fitted_result):
        """Result should include privacy parameters used."""
        assert hasattr(fitted_result, 'epsilon_used')
        assert hasattr(fitted_result, 'delta_used')


class TestBackwardsCompatibility:
    """Ensure old API still works during transition."""

    def test_old_import_still_works(self):
        """Old dp_statsmodels import should still work."""
        # This allows gradual migration
        try:
            import dp_statsmodels as dpe
            assert hasattr(dpe, 'PrivacySession')
        except ImportError:
            pytest.skip("Old package name not available")

    def test_old_session_methods_still_work(self):
        """Old lowercase methods should still work."""
        import dp_statsmodels.api as sm_dp

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)

        session = sm_dp.Session(epsilon=5.0, delta=1e-5)

        # Old style (lowercase) should still work
        result = session.ols(y, X)
        assert result is not None
