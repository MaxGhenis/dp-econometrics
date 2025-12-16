"""
dp-econometrics: Differentially private econometric models with valid inference.

This library provides differentially private implementations of standard
econometric models with proper standard error computation.
"""

from dp_econometrics.session import PrivacySession
from dp_econometrics.privacy import (
    PrivacyAccountant,
    GaussianMechanism,
)
from dp_econometrics.models import (
    DPOLS,
    DPOLSResults,
    DPLogit,
    DPLogitResults,
)

__version__ = "0.1.0"
__all__ = [
    "PrivacySession",
    "PrivacyAccountant",
    "GaussianMechanism",
    "DPOLS",
    "DPOLSResults",
    "DPLogit",
    "DPLogitResults",
]
