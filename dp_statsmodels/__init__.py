"""
dp_statsmodels: Differentially private statistical models with valid inference.

A statsmodels-like API for differentially private regression with proper
standard errors and privacy budget tracking.

Example
-------
>>> import dp_statsmodels.api as sm_dp
>>> session = sm_dp.Session(epsilon=1.0, delta=1e-5)
>>> result = session.OLS(y, X)
>>> print(result.summary())
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

__version__ = "0.1.0"

__all__ = [
    # Main interface
    "Session",
    # Model classes (for standalone use)
    "OLS",
    "Logit",
    "PanelOLS",
    # Privacy utilities
    "PrivacyAccountant",
    "GaussianMechanism",
]

# Backwards compatibility alias
PrivacySession = Session
