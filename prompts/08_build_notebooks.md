# Task
Write Jupyter notebooks that demonstrate the full end-to-end workflow using modules from `src/`.

# Target files
- notebooks/01_eda.ipynb
- notebooks/02_arima.ipynb
- notebooks/03_garch.ipynb

# Notebook 1 — EDA (`notebooks/01_eda.ipynb`)

## Purpose
Exploratory data analysis of the Shanghai Composite Index (sh.000001) close price.

## Requirements
- Load raw training data via `src/data_loader.py`.
- Apply preprocessing via `src/preprocess.py`.
- Use `src/eda.py` utilities for reusable plots; inline one-off plots are acceptable.
- Sections (use markdown cells as headings):
  1. **数据加载** — load train CSV, show shape / dtypes / head.
  2. **基本统计** — descriptive statistics, missing-value summary.
  3. **收盘价走势** — line plot of close price over time.
  4. **收益率分布** — histogram + QQ plot of log returns.
  5. **自相关分析** — ACF / PACF of close price and of returns.
  6. **平稳性检验** — ADF test on close price and on returns, print conclusion.
- Save key figures to `outputs/figures/`.

# Notebook 2 — ARIMA (`notebooks/02_arima.ipynb`)

## Purpose
Fit and evaluate an ARIMA model on the close price series.

## Requirements
- Import and call functions from `src/data_loader.py`, `src/preprocess.py`, and `src/arima_model.py`.
- Sections:
  1. **数据准备** — load & preprocess training data; apply differencing if needed for stationarity.
  2. **参数搜索** — run AIC/BIC grid search; display a summary table of candidate orders.
  3. **模型拟合** — fit the chosen ARIMA(p,d,q) model; print model summary.
  4. **残差诊断** — residual plot, ACF of residuals, Ljung-Box test; comment on white-noise assumption.
  5. **样本内拟合** — plot fitted values vs actual on training set.
  6. **样本外预测** — load test data, produce forecasts with confidence intervals, plot forecast vs actual.
  7. **评估指标** — compute and display MAE, RMSE, MAPE on the test set.
  8. **结论** — brief markdown cell summarizing model performance and limitations.
- Save the forecast comparison figure and residual diagnostics figure to `outputs/figures/`.

# Notebook 3 — GARCH (`notebooks/03_garch.ipynb`)

## Purpose
Fit and evaluate a GARCH model on the return series to capture volatility dynamics.

## Requirements
- Import and call functions from `src/data_loader.py`, `src/preprocess.py`, and `src/garch_model.py`.
- Sections:
  1. **数据准备** — load & preprocess; compute log returns; verify stationarity.
  2. **波动率聚集观察** — plot returns and squared returns; comment on volatility clustering.
  3. **ARCH 效应检验** — Engle's ARCH test on returns; print result.
  4. **参数搜索** — compare GARCH specifications (e.g. (1,1), (1,2), (2,1)) by AIC/BIC; display summary table.
  5. **模型拟合** — fit chosen GARCH(p,q); print model summary.
  6. **残差诊断** — standardized residual plot, ACF of squared standardized residuals, Ljung-Box test.
  7. **条件波动率** — plot conditional volatility over the training period.
  8. **样本外预测** — produce volatility forecasts on the test period; plot against realized volatility proxy (e.g. squared returns or rolling std).
  9. **结论** — brief markdown cell summarizing volatility dynamics and model adequacy.
- Save the conditional volatility figure and residual diagnostics figure to `outputs/figures/`.

# Constraints
- Notebooks must call `src/` module functions for core logic. Do not duplicate business logic inline.
- Use markdown cells with Chinese headings to narrate each step.
- Keep each code cell focused — one logical step per cell.
- Use `%matplotlib inline` or equivalent at the top.
- Add `sys.path` or package setup so that `import src.*` works from `notebooks/`.
- Do not hardcode absolute file paths; use relative paths from the project root.
- Figures should be publication-quality: labeled axes, titles, legends where appropriate.

# Expected output
- Three runnable `.ipynb` notebooks in `notebooks/`.
- Key figures saved under `outputs/figures/`.
- Clear narrative connecting data → model → diagnostics → conclusions.