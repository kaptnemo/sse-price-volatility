---
applyTo: "src/*garch*.py"
---

# GARCH instructions

- Use returns or residual-like stationary series, not raw price levels, unless explicitly justified.
- Check for volatility clustering or ARCH effects before finalizing the model.
- Support model comparison using AIC and/or BIC.
- Output conditional volatility in a structured format.
- Separate data preparation, fitting, diagnostics, and forecasting into clear functions.