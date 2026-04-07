# Copilot Instructions

This repository is an engineering-style Python data analysis project.

## Global rules
- Keep the project modular and maintainable.
- Put core logic in `src/`, not only in notebooks.
- Change only the files relevant to the current task.
- Do not make broad refactors unless explicitly requested.
- Preserve existing public interfaces unless the task requires changing them.

## Python rules
- Use type hints for functions.
- Add docstrings to public functions.
- Prefer clear, testable functions over long scripts.
- Avoid hardcoded paths; use config or parameters.
- Prefer standard scientific Python libraries such as pandas, numpy, statsmodels, matplotlib, sklearn.

## Data and modeling rules
- Separate raw data and processed data.
- Perform basic data validation before modeling.
- For modeling tasks, include evaluation and diagnostics.
- Do not make overly strong real-world claims from statistical fit alone.

## Output expectations
- Provide short design reasoning before code when helpful.
- Ensure code is runnable.
- Include a minimal usage example when implementing a module.