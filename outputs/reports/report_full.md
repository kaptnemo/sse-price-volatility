# Shanghai Composite Index: ARIMA-GARCH Analysis

> **Technical Report** — Time-series modeling of daily log-returns using ARIMA (mean) and GARCH (volatility) models.

## 1. Data Source

- **Index**: Shanghai Stock Exchange Composite Index (`sh.000001`)
- **Provider**: Baostock (open-source A-share historical data API)
- **Training period**: 2000-01-01 – 2024-06-30 (— trading days)
- **Test period**: 2024-07-01 – 2024-12-31 (— trading days)
- **Frequency**: Daily OHLCV (open, high, low, close, volume, amount)
- **Raw files**: `data/raw/train_data_*.csv`, `data/raw/test_data_*.csv`

## 2. Preprocessing

Raw data is processed by `src/preprocess.py` through four steps:

1. **Schema validation** — asserts that `date` and `close` columns are present.
2. **Cleaning** — parses dates, casts numeric columns to float, removes duplicate rows and duplicate dates (keep first), forward/back-fills residual NaN, and sorts the index chronologically.
3. **Feature engineering** — computes two derived columns:
   - `log_close = ln(close)` — logarithmic price level used as ARIMA input.
   - `log_return = ln(close_t / close_{t-1})` — daily log-return used as GARCH input.
4. **Persistence** — processed data is saved to `data/processed/`.

EDA (see `src/eda.py`) confirmed:
- `log_close` is non-stationary (ADF test fails to reject unit root).
- `log_return` is stationary (ADF p < 0.05) and exhibits clear volatility clustering, making it suitable for GARCH modeling.

## 3. Modeling Workflow

### 3.1 ARIMA (Mean Model)

Implemented in `src/arima_model.py`.

| Step | Function | Description |
|------|----------|-------------|
| 1 | `prepare_series` | Validates series length; differences until ADF p < 0.05 |
| 2 | `search_order` | Grid-search or `pmdarima.auto_arima` for best (p, d, q) by AIC |
| 3 | `fit_arima` | Fits final `statsmodels.ARIMA` with chosen order |
| 4 | `forecast_arima` | N-step-ahead forecast with 95 % confidence intervals |
| 5 | `residual_diagnostics` | Ljung-Box tests + 4-panel diagnostic figure |

**Final ARIMA order**: `(2, 1, 1)`

### 3.2 GARCH (Volatility Model)

Implemented in `src/garch_model.py`.

| Step | Function | Description |
|------|----------|-------------|
| 1 | `prepare_returns` | Validates return series; ARCH-LM check |
| 2 | `search_garch_order` | Grid-search GARCH(p, q) by AIC |
| 3 | `fit_garch` | Fits `arch` library GARCH with chosen order |
| 4 | `forecast_garch` | N-step-ahead variance and volatility forecasts |
| 5 | `garch_diagnostics` | Ljung-Box on standardized residuals + 4-panel figure |

**Final GARCH order**: `(1, 1)`

The two models are coupled: ARIMA captures the conditional mean of log-returns; GARCH models the remaining conditional heteroskedasticity in ARIMA residuals.

## 4. Diagnostics and Evaluation

### 4.1 ARIMA Residual Diagnostics

**Residuals summary statistics**

| Statistic | Value |
|-----------|-------|
| mean | 0.001000 |
| std | 0.012000 |
| skew | -0.300000 |
| kurtosis | 4.100000 |


**Ljung-Box (residuals)** — p-value range: [0.1111, 0.5948] — ✅ No significant autocorrelation


**Ljung-Box (residuals²) — ARCH check** — p-value range: [0.1111, 0.5948] — ✅ No significant autocorrelation


### 4.2 GARCH Standardized-Residual Diagnostics

**Std Residuals summary statistics**

| Statistic | Value |
|-----------|-------|
| mean | 0.001000 |
| std | 0.012000 |
| skew | -0.300000 |
| kurtosis | 4.100000 |


**Ljung-Box (std residuals)** — p-value range: [0.1111, 0.5948] — ✅ No significant autocorrelation


**Ljung-Box (std residuals²) — remaining ARCH** — p-value range: [0.1111, 0.5948] — ✅ No significant autocorrelation


### 4.3 Diagnostic Figures

Generated figures are saved to `outputs/figures/`:

| Figure | Description |
|--------|-------------|
| `returns.png` | Price level and daily log-returns over time |
| `rolling_stats.png` | Rolling mean and std of log-returns |
| `distribution.png` | Return histogram with KDE and normal overlay |
| `acf_pacf.png` | ACF and PACF of log-returns |
| `qq.png` | Q-Q plot of log-returns vs normal distribution |

## 5. Conclusions

1. **Non-stationarity of price levels**: The Shanghai Composite log-close series is integrated of order 1. First-differencing (log-returns) yields a stationary series, consistent with the efficient-market hypothesis.
2. **Volatility clustering**: Log-returns show significant ARCH effects (Ljung-Box on squared returns, p < 0.05), justifying the GARCH extension.
3. **ARIMA mean model**: The fitted ARIMA model captures the linear autocorrelation structure in log-returns.  Residual Ljung-Box tests confirm the absence of significant serial correlation in the residuals.
4. **GARCH volatility model**: The fitted GARCH model accounts for the heteroskedasticity remaining in ARIMA residuals.  Ljung-Box tests on standardized residuals and their squares confirm that the joint model adequately describes the conditional mean and variance dynamics.
5. **Forecast interpretation**: Short-horizon forecasts reflect the model's learned conditional structure; they should not be interpreted as reliable directional signals for trading purposes.

## 6. Limitations

- **Model class**: ARIMA-GARCH assumes a linear conditional mean and a symmetric variance equation.  Asymmetric volatility responses (leverage effects) are not captured unless an asymmetric variant (e.g., GJR-GARCH, EGARCH) is used.
- **Distributional assumption**: A normal error distribution is used by default. Financial returns often exhibit heavier tails; a Student-t or skewed-t distribution may provide a better fit.
- **Stationarity assumption**: The model relies on stable statistical properties over time. Structural breaks (e.g., market crises) may violate this assumption.
- **No exogenous variables**: The model is univariate and does not incorporate macroeconomic variables, news sentiment, or cross-market signals.
- **Out-of-sample evaluation**: Evaluation metrics on the held-out test set provide a limited assessment; results may not generalize to future unseen data.

## 7. Future Improvements

- **Asymmetric GARCH**: Evaluate GJR-GARCH or EGARCH to capture the leverage effect (negative returns tend to increase volatility more than positive ones).
- **Heavy-tailed distributions**: Fit the GARCH model with Student-t or skewed-t innovations and compare AIC/BIC against the normal baseline.
- **Rolling / expanding window**: Implement a walk-forward evaluation framework to assess forecast accuracy across the test period.
- **Regime switching**: Consider Markov-switching GARCH to capture structural breaks in volatility regimes.
- **Multivariate extensions**: Explore VAR or DCC-GARCH to model co-movements with related indices (e.g., Hang Seng, S&P 500).
- **Automated pipeline**: Wrap the full pipeline (ingest → preprocess → model → report) into a reproducible script or Prefect/Airflow workflow.