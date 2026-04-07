---
applyTo: "src/*arima*.py"
---

# ARIMA instructions

- Check stationarity before choosing the final ARIMA specification.
- Support order selection using AIC and/or BIC.
- Include residual diagnostics.
- Keep order search and final fitting as separate functions when possible.
- Provide forecast outputs in a structured format.
- Prefer reproducible, explainable modeling steps over opaque shortcuts.