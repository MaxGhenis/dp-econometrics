"""
dp_statsmodels API module.

Import as:
    import dp_statsmodels.api as sm_dp

This provides a statsmodels-like interface for differentially private
regression models.
"""

from dp_statsmodels.session import Session
from dp_statsmodels.models import (
    OLS,
    Logit,
    PanelOLS,
)
from dp_statsmodels.privacy import (
    PrivacyAccountant,
    GaussianMechanism,
)

__all__ = [
    "Session",
    "OLS",
    "Logit",
    "PanelOLS",
    "PrivacyAccountant",
    "GaussianMechanism",
]
