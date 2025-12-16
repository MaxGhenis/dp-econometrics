# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**dp-econometrics** is a Python library for differentially private econometric analysis with valid statistical inference. It implements the Noisy Sufficient Statistics (NSS) approach for linear models with proper standard error computation.

## Key Design Decisions

### Architecture

```
dp_econometrics/
├── __init__.py           # Public API: PrivacySession
├── session.py            # Main user interface with budget tracking
├── models/
│   ├── ols.py           # DP OLS via Noisy Sufficient Statistics
│   └── (future: logit.py, fe.py)
├── privacy/
│   ├── mechanisms.py    # Gaussian mechanism
│   ├── accounting.py    # Privacy budget tracking
│   ├── sensitivity.py   # Sensitivity computation
│   └── noisy_stats.py   # Noisy X'X, X'y computation
└── tests/               # Unit tests (TDD)
```

### Why NSS Instead of DP-SGD

For OLS specifically, Noisy Sufficient Statistics is preferred over DP-SGD because:
1. **Efficiency**: One-shot noise addition vs. iterative training
2. **Theory**: Well-understood variance formulas exist
3. **Accuracy**: Better privacy-utility tradeoff for linear models

### Privacy Accounting

- Uses basic sequential composition by default
- RDP composition available for tighter bounds with many queries
- Tracks (ε, δ) spent across all queries in a session

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=dp_econometrics --cov-report=term-missing

# Format code
black dp_econometrics tests
isort dp_econometrics tests

# Lint
flake8 dp_econometrics

# Type check
mypy dp_econometrics
```

## Documentation

Documentation uses **Jupyter Book 2.0 (MyST-MD)**:

```bash
cd docs
myst build --html
```

## Testing Strategy (TDD)

Tests are in `tests/` and follow test-driven development:
1. `test_ols.py` - OLS model tests
2. `test_privacy.py` - Privacy mechanisms and accounting

Tests verify:
- Correct coefficient computation
- Standard error validity
- Privacy budget tracking
- Confidence interval coverage (~95%)

## Key References

- Sheffet (2017): Differentially Private OLS
- Evans et al. (2024): DP Linear Regression variance formulas
- Dwork & Roth (2014): Algorithmic foundations of DP

## Important Notes

### Data Bounds
Proper DP requires knowing data bounds a priori. If bounds are not provided:
- A warning is issued
- Bounds are computed from data (which leaks privacy)
- This is acceptable for development/testing but not for production

### Standard Errors
Standard errors account for both:
1. Sampling variance: σ²(X'X)⁻¹
2. Privacy noise variance: from noise added to sufficient statistics

The variance formula is an approximation based on the delta method.
