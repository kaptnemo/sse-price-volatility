---
applyTo: "notebooks/*.ipynb"
---

# Jupyter Notebook instructions
- If need the third-party libraries, use poetry to manage dependencies and virtual environments.
- Use markdown cells to explain the purpose and steps of the analysis.
- Import and utilize functions from `src/` modules to ensure code reuse and consistency.
- Avoid hardcoding paths; use relative paths or configuration variables.
- When performing data analysis, clearly document assumptions and interpretations of results.
- For visualizations, include titles, axis labels, and legends for clarity.
- Save key figures to `outputs/figures/` for later reference and reporting.
- Ensure that the notebook can be run end-to-end without errors, and that it produces the expected outputs.