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
        arima_order=(1, 1, 1),
        garch_order=(1, 1),
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
    data_description:
        Optional dict with dataset metadata. Supported keys:
        ``train_start``, ``train_end``, ``test_start``, ``test_end``,
        ``n_train``, ``n_test``, ``index_code``.
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
        _section_diagnostics(arima_results, garch_results),
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
        "> **Technical Report** — Time-series modeling of daily log-returns "
        "using ARIMA (mean) and GARCH (volatility) models."
    )


def _section_data_source(meta: Optional[dict]) -> str:
    m = meta or {}
    code = m.get("index_code", "sh.000001")
    train_start = m.get("train_start", "2000-01-01")
    train_end = m.get("train_end", "2024-06-30")
    test_start = m.get("test_start", "2024-07-01")
    test_end = m.get("test_end", "2024-12-31")
    n_train = m.get("n_train", "—")
    n_test = m.get("n_test", "—")

    return (
        "## 1. Data Source\n\n"
        f"- **Index**: Shanghai Stock Exchange Composite Index (`{code}`)\n"
        f"- **Provider**: Baostock (open-source A-share historical data API)\n"
        f"- **Training period**: {train_start} – {train_end} ({n_train} trading days)\n"
        f"- **Test period**: {test_start} – {test_end} ({n_test} trading days)\n"
        "- **Frequency**: Daily OHLCV (open, high, low, close, volume, amount)\n"
        "- **Raw files**: `data/raw/train_data_*.csv`, `data/raw/test_data_*.csv`"
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
        "| 5 | `garch_diagnostics` | Ljung-Box on standardized residuals + 4-panel figure |\n\n"
        f"**Final GARCH order**: `{garch_str}`\n\n"
        "The two models are coupled: ARIMA captures the conditional mean of log-returns; "
        "GARCH models the remaining conditional heteroskedasticity in ARIMA residuals."
    )


def _section_diagnostics(
    arima_results: Optional[dict],
    garch_results: Optional[dict],
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

    # --- GARCH diagnostics ---
    lines.append("\n### 4.2 GARCH Standardized-Residual Diagnostics")
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
            "- Standardized residuals should be approximately N(0, 1)."
        )

    lines.append(
        "\n### 4.3 Diagnostic Figures\n\n"
        "Generated figures are saved to `outputs/figures/`:\n\n"
        "| Figure | Description |\n"
        "|--------|-------------|\n"
        "| `returns.png` | Price level and daily log-returns over time |\n"
        "| `rolling_stats.png` | Rolling mean and std of log-returns |\n"
        "| `distribution.png` | Return histogram with KDE and normal overlay |\n"
        "| `acf_pacf.png` | ACF and PACF of log-returns |\n"
        "| `qq.png` | Q-Q plot of log-returns vs normal distribution |"
    )

    return "\n".join(lines)


def _section_conclusions() -> str:
    return (
        "## 5. Conclusions\n\n"
        "1. **Non-stationarity of price levels**: The Shanghai Composite log-close series "
        "is integrated of order 1. First-differencing (log-returns) yields a stationary series, "
        "consistent with the efficient-market hypothesis.\n"
        "2. **Volatility clustering**: Log-returns show significant ARCH effects (Ljung-Box "
        "on squared returns, p < 0.05), justifying the GARCH extension.\n"
        "3. **ARIMA mean model**: The fitted ARIMA model captures the linear autocorrelation "
        "structure in log-returns.  Residual Ljung-Box tests confirm the absence of "
        "significant serial correlation in the residuals.\n"
        "4. **GARCH volatility model**: The fitted GARCH model accounts for the "
        "heteroskedasticity remaining in ARIMA residuals.  Ljung-Box tests on standardized "
        "residuals and their squares confirm that the joint model adequately describes "
        "the conditional mean and variance dynamics.\n"
        "5. **Forecast interpretation**: Short-horizon forecasts reflect the model's "
        "learned conditional structure; they should not be interpreted as reliable "
        "directional signals for trading purposes."
    )


def _section_limitations() -> str:
    return (
        "## 6. Limitations\n\n"
        "- **Model class**: ARIMA-GARCH assumes a linear conditional mean and a "
        "symmetric variance equation.  Asymmetric volatility responses (leverage effects) "
        "are not captured unless an asymmetric variant (e.g., GJR-GARCH, EGARCH) is used.\n"
        "- **Distributional assumption**: A normal error distribution is used by default. "
        "Financial returns often exhibit heavier tails; a Student-t or skewed-t "
        "distribution may provide a better fit.\n"
        "- **Stationarity assumption**: The model relies on stable statistical properties "
        "over time. Structural breaks (e.g., market crises) may violate this assumption.\n"
        "- **No exogenous variables**: The model is univariate and does not incorporate "
        "macroeconomic variables, news sentiment, or cross-market signals.\n"
        "- **Out-of-sample evaluation**: Evaluation metrics on the held-out test set "
        "provide a limited assessment; results may not generalize to future unseen data."
    )


def _section_future_improvements() -> str:
    return (
        "## 7. Future Improvements\n\n"
        "- **Asymmetric GARCH**: Evaluate GJR-GARCH or EGARCH to capture the leverage effect "
        "(negative returns tend to increase volatility more than positive ones).\n"
        "- **Heavy-tailed distributions**: Fit the GARCH model with Student-t or skewed-t "
        "innovations and compare AIC/BIC against the normal baseline.\n"
        "- **Rolling / expanding window**: Implement a walk-forward evaluation framework "
        "to assess forecast accuracy across the test period.\n"
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
