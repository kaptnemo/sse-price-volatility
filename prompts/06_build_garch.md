# Task
Implement the GARCH modeling module.

# Target file
src/garch_model.py

# Requirements
- Accept a stationary return-like series.
- Fit GARCH,EGARCH,GJR-GARCH models
- Support comparison by AIC/BIC.
- Use Rolling forecast, Accept train_data and test_data rolling window is the length of train_data
- Output conditional volatility and forecast results.
- Include simple diagnostics.

# Constraints
- Keep preparation, fit, diagnostics, and forecast logic separated.
- Do not use raw price levels unless explicitly justified.
- Use type hints and docstrings.

# Expected output
- Short design explanation
- Complete code for src/garch_model.py
- Minimal usage example