---
applyTo: "src/**/*.py"
---

# Python module instructions

- Keep each module focused on one responsibility.
- Use small functions instead of large monolithic blocks.
- All public functions must have type hints and docstrings.
- Prefer explicit error handling for invalid input.
- Avoid hidden side effects.
- Use lowercase_with_underscores for function and variable names.
- Do not hardcode dataset paths inside business logic.
- After adding new functions or modules, add tests for critical functions when possible in the tests/ directory.
- If need the third-party libraries, use poetry to manage dependencies and virtual environments.