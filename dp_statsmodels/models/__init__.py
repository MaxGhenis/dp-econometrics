"""
Differentially private statistical models.
"""

from dp_statsmodels.models.ols import DPOLS, DPOLSResults
from dp_statsmodels.models.logit import DPLogit, DPLogitResults
from dp_statsmodels.models.fixed_effects import DPFixedEffects, DPFixedEffectsResults

# Statsmodels-like aliases
OLS = DPOLS
Logit = DPLogit
PanelOLS = DPFixedEffects

# Results aliases
OLSResults = DPOLSResults
LogitResults = DPLogitResults
PanelOLSResults = DPFixedEffectsResults

__all__ = [
    # Statsmodels-like names (preferred)
    "OLS",
    "Logit",
    "PanelOLS",
    "OLSResults",
    "LogitResults",
    "PanelOLSResults",
    # Original names (for backwards compatibility)
    "DPOLS",
    "DPOLSResults",
    "DPLogit",
    "DPLogitResults",
    "DPFixedEffects",
    "DPFixedEffectsResults",
]
