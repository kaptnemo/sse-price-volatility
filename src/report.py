"""
Report generation module for the Shanghai Index ARIMA-GARCH project.

Generates a structured technical report in Markdown, covering:
  1. Data source
  2. Preprocessing
  3. Modeling workflow  (ARIMA + GARCH)
  4. Diagnostics and evaluation
  5. Conclusions
  6. Limitations
  7. Future improvements

Usage
-----
Minimal (template only):
    from report import generate_report
    md = generate_report()
    print(md)

With model results embedded:
    from report import generate_report
    md = generate_report(
        arima_results=arima_diag,   # dict from arima_model.residual_diagnostics()
        garch_results=garch_diag,   # dict from garch_model.garch_diagnostics()
        arima_order=(4, 1, 3),
        garch_order=(1, 2),
        arima_metrics={"MAE": 0.011919, "RMSE": 0.017323, "MAPE (%)": 0.14723},
        save_path="outputs/reports/report.md",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    arima_order: Optional[tuple[int, int, int]] = None,
    garch_order: Optional[tuple[int, int]] = None,
    arima_results: Optional[dict] = None,
    garch_results: Optional[dict] = None,
    arima_metrics: Optional[dict] = None,
    data_description: Optional[dict] = None,
    save_path: Optional[str | Path] = None,
) -> str:
    """Generate a structured technical report in Markdown format.

    Parameters
    ----------
    arima_order:
        Final ARIMA ``(p, d, q)`` order used.  Displayed if provided.
    garch_order:
        Final GARCH ``(p, q)`` order used.  Displayed if provided.
    arima_results:
        Dict returned by :func:`arima_model.residual_diagnostics`.
        Keys used: ``summary_stats``, ``ljung_box``, ``ljung_box_sq``.
    garch_results:
        Dict returned by :func:`garch_model.garch_diagnostics`.
        Keys used: ``summary_stats``, ``ljung_box``, ``ljung_box_sq``.
    arima_metrics:
        Optional dict of out-of-sample evaluation metrics for ARIMA, e.g.
        ``{"MAE": 0.011919, "RMSE": 0.017323, "MAPE (%)": 0.14723}``.
    data_description:
        Optional dict with dataset metadata. Supported keys:
        ``train_start``, ``train_end``, ``test_start``, ``test_end``,
        ``n_train``, ``n_test``, ``index_code``,
        ``eda_start``, ``eda_end``, ``n_eda``.
    save_path:
        If provided, the report is written to this path (Markdown file).
        Parent directories are created automatically.

    Returns
    -------
    str
        Full report text in Markdown.
    """
    sections = [
        _section_header(),
        _section_data_source(data_description),
        _section_preprocessing(),
        _section_modeling_workflow(arima_order, garch_order),
        _section_diagnostics(arima_results, garch_results, arima_metrics),
        _section_conclusions(),
        _section_limitations(),
        _section_future_improvements(),
    ]

    report = "\n\n".join(sections)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _section_header() -> str:
    return (
        "# Shanghai Composite Index: ARIMA-GARCH Analysis\n\n"
        "> **Technical Report** — Time-series modeling of the Shanghai Composite Index "
        "using ARIMA (log-close mean model) and GARCH (log-return volatility model)."
    )


def _section_data_source(meta: Optional[dict]) -> str:
    m = meta or {}
    code = m.get("index_code", "sh.000001")
    eda_start = m.get("eda_start", "2000-01-05")
    eda_end = m.get("eda_end", "2024-06-28")
    n_eda = m.get("n_eda", 5932)
    train_start = m.get("train_start", "2016-01-05")
    train_end = m.get("train_end", "2020-12-31")
    test_start = m.get("test_start", "2021-01-05")
    test_end = m.get("test_end", "2025-12-31")
    n_train = m.get("n_train", 1217)
    n_test = m.get("n_test", 1211)

    return (
        "## 1. Data Source\n\n"
        f"- **Index**: Shanghai Stock Exchange Composite Index (`{code}`)\n"
        f"- **Provider**: Baostock (open-source A-share historical data API)\n"
        f"- **EDA period**: {eda_start} – {eda_end} ({n_eda} trading days, full history)\n"
        f"- **Training period**: {train_start} – {train_end} ({n_train} trading days)\n"
        f"- **Test period**: {test_start} – {test_end} ({n_test} trading days)\n"
        "- **Frequency**: Daily OHLCV (open, high, low, close, volume, amount)\n"
        "- **Raw files**: `data/raw/train_data_*.csv`, `data/raw/test_data_*.csv`\n\n"
        "> EDA (notebook `01_eda.ipynb`) uses the full 2000–2024 history to characterise "
        "the long-run statistical properties of the index.  ARIMA and GARCH models are "
        "fitted on the shorter 2016–2020 window to keep computation tractable and reflect "
        "more recent market regimes."
    )


def _section_preprocessing() -> str:
    return (
        "## 2. Preprocessing\n\n"
        "Raw data is processed by `src/preprocess.py` through four steps:\n\n"
        "1. **Schema validation** — asserts that `date` and `close` columns are present.\n"
        "2. **Cleaning** — parses dates, casts numeric columns to float, removes "
        "duplicate rows and duplicate dates (keep first), forward/back-fills residual NaN, "
        "and sorts the index chronologically.\n"
        "3. **Feature engineering** — computes two derived columns:\n"
        "   - `log_close = ln(close)` — logarithmic price level used as ARIMA input.\n"
        "   - `log_return = ln(close_t / close_{t-1})` — daily log-return used as GARCH input.\n"
        "4. **Persistence** — processed data is saved to `data/processed/`.\n\n"
        "EDA (see `src/eda.py`) confirmed:\n"
        "- `log_close` is non-stationary (ADF test fails to reject unit root).\n"
        "- `log_return` is stationary (ADF p < 0.05) and exhibits clear volatility "
        "clustering, making it suitable for GARCH modeling."
    )


def _section_modeling_workflow(
    arima_order: Optional[tuple[int, int, int]],
    garch_order: Optional[tuple[int, int]],
) -> str:
    arima_str = str(arima_order) if arima_order else "selected via AIC grid-search / `auto_arima`"
    garch_str = str(garch_order) if garch_order else "selected via AIC grid-search"

    return (
        "## 3. Modeling Workflow\n\n"
        "### 3.1 ARIMA (Mean Model)\n\n"
        "Implemented in `src/arima_model.py`.\n\n"
        "| Step | Function | Description |\n"
        "|------|----------|-------------|\n"
        "| 1 | `prepare_series` | Validates series length; differences until ADF p < 0.05 |\n"
        "| 2 | `search_order` | Grid-search or `pmdarima.auto_arima` for best (p, d, q) by AIC |\n"
        "| 3 | `fit_arima` | Fits final `statsmodels.ARIMA` with chosen order |\n"
        "| 4 | `forecast_arima` | Direct N-step-ahead forecast helper with confidence intervals |\n"
        "| 5 | `rolling_forecast_arima` | 1-step-ahead rolling forecast on the test period |\n"
        "| 6 | `residual_diagnostics` | Ljung-Box tests + 4-panel diagnostic figure |\n\n"
        f"**Final ARIMA order**: `{arima_str}`\n\n"
        "### 3.2 GARCH (Volatility Model)\n\n"
        "Implemented in `src/garch_model.py`.\n\n"
        "| Step | Function | Description |\n"
        "|------|----------|-------------|\n"
        "| 1 | `prepare_returns` | Validates return series; ARCH-LM check |\n"
        "| 2 | `search_garch_order` | Grid-search GARCH(p, q) by AIC |\n"
        "| 3 | `fit_garch` | Fits `arch` library GARCH with chosen order |\n"
        "| 4 | `forecast_garch` | N-step-ahead variance and volatility forecasts |\n"
        "| 5 | `rolling_forecast_garch` | 1-step-ahead rolling volatility forecast on the test period |\n"
        "| 6 | `garch_diagnostics` | Ljung-Box on standardized residuals + 4-panel figure |\n\n"
        f"**Final GARCH order**: `{garch_str}`\n\n"
        "The two models are coupled: ARIMA captures the conditional mean of the "
        "**log-close** price level (d=1 differencing enforces stationarity); "
        "GARCH models the conditional heteroskedasticity of **log-returns** "
        "(ARIMA residuals / direct return series)."
    )


def _section_diagnostics(
    arima_results: Optional[dict],
    garch_results: Optional[dict],
    arima_metrics: Optional[dict] = None,
) -> str:
    lines = ["## 4. Diagnostics and Evaluation"]

    # --- ARIMA diagnostics ---
    lines.append("\n### 4.1 ARIMA Residual Diagnostics")
    if arima_results:
        stats: pd.Series = arima_results.get("summary_stats", pd.Series(dtype=float))
        lb: pd.DataFrame = arima_results.get("ljung_box", pd.DataFrame())
        lb_sq: pd.DataFrame = arima_results.get("ljung_box_sq", pd.DataFrame())

        lines.append(_format_summary_stats_table(stats, label="Residuals"))
        lines.append(_format_lb_summary(lb, "Ljung-Box (residuals)"))
        lines.append(_format_lb_summary(lb_sq, "Ljung-Box (residuals²) — ARCH check"))
    else:
        lines.append(
            "\n*No ARIMA results provided. Run `arima_model.residual_diagnostics()` "
            "and pass the output to `generate_report()` to embed actual statistics.*\n\n"
            "Diagnostics to check:\n"
            "- Ljung-Box test on residuals: p-values should be > 0.05 (no serial correlation).\n"
            "- Ljung-Box test on squared residuals: significant p-values indicate "
            "remaining ARCH effects — GARCH is then applied to the residuals.\n"
            "- Residual distribution should be approximately normal (Q-Q, histogram)."
        )

    # --- ARIMA out-of-sample evaluation ---
    lines.append("\n### 4.2 ARIMA Out-of-Sample Evaluation (log-close)")
    if arima_metrics:
        lines.append(_format_metrics_table(arima_metrics))
    else:
        lines.append(
            "\n*No evaluation metrics provided.  Pass `arima_metrics` dict to "
            "`generate_report()`.  Expected keys: `MAE`, `RMSE`, `MAPE (%)`.*"
        )

    # --- GARCH diagnostics ---
    lines.append("\n### 4.3 GARCH Standardized-Residual Diagnostics")
    if garch_results:
        gstats: pd.Series = garch_results.get("summary_stats", pd.Series(dtype=float))
        glb: pd.DataFrame = garch_results.get("ljung_box", pd.DataFrame())
        glb_sq: pd.DataFrame = garch_results.get("ljung_box_sq", pd.DataFrame())

        lines.append(_format_summary_stats_table(gstats, label="Std Residuals"))
        lines.append(_format_lb_summary(glb, "Ljung-Box (std residuals)"))
        lines.append(_format_lb_summary(glb_sq, "Ljung-Box (std residuals²) — remaining ARCH"))
    else:
        lines.append(
            "\n*No GARCH results provided. Run `garch_model.garch_diagnostics()` "
            "and pass the output to `generate_report()` to embed actual statistics.*\n\n"
            "Diagnostics to check:\n"
            "- Ljung-Box test on standardized residuals: p-values > 0.05 indicate "
            "no remaining serial correlation.\n"
            "- Ljung-Box test on squared standardized residuals: p-values > 0.05 indicate "
            "the GARCH model has adequately captured volatility clustering.\n"
            "- Standardized residuals should be approximately N(0, 1).\n"
            "- Note: residual ARCH effects at short lags suggest an asymmetric variant "
            "(GJR-GARCH / EGARCH) or heavier-tailed distribution may improve fit."
        )

    lines.append(
        "\n### 4.4 Diagnostic Figures\n\n"
        "Generated figures are saved to `outputs/figures/`:\n\n"
        "| Figure | Description |\n"
        "|--------|-------------|\n"
        "| `eda_close_price.png` | Shanghai Composite closing price (2000–2024 full history) |\n"
        "| `eda_return_distribution.png` | Log-return histogram with KDE and normal fit |\n"
        "| `eda_return_acf_pacf.png` | ACF and PACF of log-returns |\n"
        "| `eda_return_qq.png` | Q-Q plot of log-returns vs normal distribution |\n"
        "| `eda_close_acf_pacf.png` | ACF and PACF of log-close (non-stationary reference) |\n"
        "| `arima_insample_fit.png` | ARIMA in-sample fitted values vs actual log-close |\n"
        "| `arima_residual_diagnostics.png` | 4-panel ARIMA residual diagnostics |\n"
        "| `arima_forecast_vs_actual.png` | Rolling 1-step-ahead ARIMA forecast vs actual |\n"
        "| `garch_volatility_clustering.png` | Log-returns and squared returns (ARCH effect) |\n"
        "| `garch_conditional_volatility.png` | GARCH conditional volatility time series |\n"
        "| `garch_residual_diagnostics.png` | 4-panel GARCH(1,2) standardized-residual diagnostics |\n"
        "| `garch_forecast_vs_realized.png` | GARCH volatility forecast vs realized volatility |\n"
        "| `garch_rolling_forecast_vs_realized.png` | Rolling GARCH forecast vs realized volatility |\n"
        "| `gjr_garch_residual_diagnostics.png` | 4-panel GJR-GARCH(1,2) standardized-residual diagnostics |\n"
        "| `garch_vs_gjr_conditional_volatility.png` | Conditional volatility comparison: GARCH vs GJR-GARCH |\n"
        "| `gjr_garch_rolling_forecast_vs_realized.png` | GJR-GARCH rolling forecast vs realized volatility (3-panel) |\n"
        "| `vol_forecast_scatter.png` | Predicted volatility vs |return| proxy scatter plots |\n"
        "| `var_backtest.png` | 1%/5% VaR breach timeline for GARCH and GJR-GARCH |"
    )

    return "\n".join(lines)


def _section_conclusions() -> str:
    return (
        "## 5. Conclusions\n\n"
        "1. **Non-stationarity of price levels**: The Shanghai Composite log-close series "
        "is integrated of order 1 (ADF p = 0.15 on close; p < 1e-29 on log-returns). "
        "First-differencing yields a stationary series consistent with a random walk.\n"
        "2. **Volatility clustering**: Log-returns show highly significant ARCH effects "
        "(Ljung-Box on squared returns, p < 0.05 for all 10 lags), justifying the GARCH extension.\n"
        "3. **ARIMA mean model** (log-close, d=1): ARIMA(4,1,3) achieves the lowest AIC "
        "(−7380.13) on the 2016–2020 training set. Residual Ljung-Box tests confirm "
        "white-noise residuals (all lags p > 0.05). Out-of-sample on 2021–2025: "
        "MAE = 0.0119, RMSE = 0.0173, MAPE = 0.15 %.\n"
        "4. **GARCH volatility model** (log-returns): GARCH(1,2) is selected by AIC. "
        "Volatility persistence α+β = 0.99, indicating near-integrated GARCH dynamics. "
        "Squared standardized residuals show significant autocorrelation at lags 2+ "
        "(Ljung-Box p < 0.05), suggesting residual ARCH effects remain.\n"
        "5. **GJR-GARCH improves fit substantially**: GJR-GARCH(1,2) reduces AIC by 205 "
        "points (−5697 vs −5491) over symmetric GARCH. The leverage coefficient "
        "γ = 0.018 > 0 confirms that negative return shocks amplify volatility more than "
        "positive shocks, consistent with the leverage effect in equity markets. "
        "Volatility persistence decreases to α + γ/2 + β = 0.947. Residual ARCH effects "
        "at short lags persist in both models, pointing to higher-order or asymmetric "
        "distributional extensions as next steps.\n"
        "6. **Out-of-sample quantitative evaluation (1,211 test days)**: Predicted volatility "
        "is evaluated against |r_t| proxy. RMSE: GARCH = 0.0081, GJR-GARCH = 0.0081 "
        "(negligible difference). QLIKE loss (Patton 2011, scale-free): GARCH = −8.159, "
        "GJR-GARCH = −8.178 (lower is better); GJR-GARCH is marginally preferred under QLIKE.\n"
        "7. **VaR backtesting (Kupiec POF test)**: Both models produce well-calibrated risk "
        "estimates. At the 1% level: 13 breaches / 1,211 days (rate = 1.07%), "
        "Kupiec LR = 0.065, p = 0.80 — H₀ (breach rate = nominal level) not rejected. "
        "At the 5% level: 69 breaches (rate = 5.70%), p = 0.28 — not rejected. "
        "Both GARCH and GJR-GARCH pass all four Kupiec tests.\n"
        "8. **Forecast interpretation**: Short-horizon forecasts reflect the model's "
        "learned conditional structure; they should not be interpreted as reliable "
        "directional signals for trading purposes."
    )


def _section_limitations() -> str:
    return (
        "## 6. Limitations\n\n"
        "- **Residual ARCH effects in both GARCH variants**: Ljung-Box on squared "
        "standardized residuals rejects white noise at lags 2+ (p < 0.05) for both "
        "GARCH(1,2) and GJR-GARCH(1,2). GJR-GARCH improves AIC by 205 points but does "
        "not fully eliminate short-lag clustering; EGARCH or higher-order specifications "
        "may be needed.\n"
        "- **Convergence of best ARIMA order**: ARIMA(4,1,3) is selected by AIC but "
        "reports `converged=False`; parameter estimates should be interpreted with caution.\n"
        "- **Model class**: ARIMA-GARCH assumes a linear conditional mean and a "
        "symmetric variance equation.  Asymmetric volatility responses (leverage effects) "
        "are not captured unless an asymmetric variant (e.g., GJR-GARCH, EGARCH) is used.\n"
        "- **Distributional assumption**: A normal error distribution is used by default. "
        "Financial returns often exhibit heavier tails (observed kurtosis ≈ 5.9 in GARCH "
        "standardized residuals); a Student-t or skewed-t distribution may provide a better fit.\n"
        "- **Training window**: The 2016–2020 window covers only one major volatility regime "
        "(post-2015 crash recovery). Structural breaks in the test period may reduce "
        "out-of-sample validity.\n"
        "- **No exogenous variables**: The model is univariate and does not incorporate "
        "macroeconomic variables, news sentiment, or cross-market signals.\n"
        "- **Out-of-sample evaluation**: Evaluation metrics on the held-out test set "
        "provide a limited assessment; results may not generalize to future unseen data."
    )


def _section_future_improvements() -> str:
    return (
        "## 7. Future Improvements\n\n"
        "- **Asymmetric GARCH**: Evaluate EGARCH to capture the leverage effect "
        "without non-negativity constraints on variance parameters.\n"
        "- **Higher-order GARCH**: GJR-GARCH(2,2) may eliminate residual short-lag "
        "ARCH effects present in both current variants.\n"
        "- **Heavy-tailed distributions**: Fit the GARCH model with skewed-t "
        "innovations to better capture asymmetry and excess kurtosis (≈ 5.9).\n"
        "- **ARIMA convergence robustness**: Compare ARIMA(4,1,3) (non-converged) with "
        "ARIMA(3,1,4) to assess practical forecast impact of the convergence issue.\n"
        "- **Regime switching**: Consider Markov-switching GARCH to capture structural breaks "
        "in volatility regimes.\n"
        "- **Multivariate extensions**: Explore VAR or DCC-GARCH to model co-movements "
        "with related indices (e.g., Hang Seng, S&P 500).\n"
        "- **Automated pipeline**: Wrap the full pipeline (ingest → preprocess → model → "
        "report) into a reproducible script or Prefect/Airflow workflow."
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_metrics_table(metrics: dict) -> str:
    """Format an evaluation-metrics dict into a Markdown table."""
    rows = "| Metric | Value |\n|--------|-------|\n"
    for name, val in metrics.items():
        rows += f"| {name} | {val:.6f} |\n"
    return "\n" + rows


def _format_summary_stats_table(stats: pd.Series, label: str = "Stats") -> str:
    """Format a summary-stats Series into a Markdown table."""
    if stats.empty:
        return ""
    header = f"\n**{label} summary statistics**\n\n"
    rows = "| Statistic | Value |\n|-----------|-------|\n"
    for name, val in stats.items():
        rows += f"| {name} | {val:.6f} |\n"
    return header + rows


def _format_lb_summary(lb: pd.DataFrame, label: str) -> str:
    """Summarise a Ljung-Box result DataFrame into a short Markdown note."""
    if lb.empty:
        return ""
    min_p = lb["lb_pvalue"].min()
    max_p = lb["lb_pvalue"].max()
    verdict = "✅ No significant autocorrelation" if min_p > 0.05 else "⚠️  Significant autocorrelation detected"
    return (
        f"\n**{label}** — p-value range: [{min_p:.4f}, {max_p:.4f}] — {verdict}\n"
    )


if __name__ == '__main__':
    generate_report()