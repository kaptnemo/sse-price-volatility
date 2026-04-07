---
applyTo: "src/*data*.py,src/preprocess.py,src/eda.py"
---

# Data workflow instructions

- Data loading functions must return pandas DataFrame objects.
- Preserve raw data integrity; do not overwrite raw inputs.
- Clearly define required input columns.
- Handle missing values and obvious schema issues explicitly.
- Save processed outputs to `data/processed/` when persistence is needed.
- Plots and analysis outputs should go to `outputs/figures/` or `outputs/reports/`.