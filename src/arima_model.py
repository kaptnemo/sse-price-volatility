"""
ARIMA modeling module for stationary time series.

Pipeline:
  1. prepare_series         – validate input, optionally difference to stationarity
  2. search_order           – grid-search or auto_arima to find best (p, d, q) by AIC/BIC
  3. fit_arima              – fit a final ARIMA model with explicit order
  4. forecast_arima         – produce direct N-step-ahead forecasts from a fitted model
  5. rolling_forecast_arima – generate 1-step-ahead rolling forecasts on test data
  6. residual_diagnostics   – compute and optionally plot residual diagnostics
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import itertools
from unittest import result

import matplotlib.pyplot as plt
import src.plot_config  # noqa: F401 — configures CJK font for all figures
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.tsa.stattools import adfuller

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ArimaOrder = tuple[int, int, int]

# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------


def prepare_series(
    series: pd.Series,
    max_diff: int = 2,
    significance: float = 0.05,
) -> tuple[pd.Series, int]:
    """Validate and difference a series until it is stationary (ADF p < significance).

    Parameters
    ----------
    series:
        Raw numeric time series (e.g. log-close or log-return).
    max_diff:
        Maximum number of differencing steps to attempt.
    significance:
        p-value threshold for the ADF test.

    Returns
    -------
    tuple[pd.Series, int]
        ``(stationary_series, d)`` where *d* is the number of differences applied.

    Raises
    ------
    ValueError
        If *series* is empty, contains NaN after cleaning, or contains fewer
        than 20 observations.
    """
    s = series.dropna()
    if s.empty:
        raise ValueError("Input series is empty after dropping NaN.")
    if len(s) < 20:
        raise ValueError(f"Series too short for ARIMA: {len(s)} observations.")

    d = 0
    current = s.copy()
    for _ in range(max_diff + 1):
        adf_stat, p_value, *_ = adfuller(current, autolag="AIC")
        if p_value < significance:
            break
        if d < max_diff:
            current = current.diff().dropna()
            d += 1
    return current, d


# ---------------------------------------------------------------------------
# 2. Order search
# ---------------------------------------------------------------------------


def search_order(
    series: pd.Series,
    d: int = 0,
    p_range: tuple[int, int] = (0, 4),
    q_range: tuple[int, int] = (0, 4),
    criterion: Literal["aic", "bic"] = "aic",
    use_auto: bool = True,
    seasonal: bool = False,
) -> pd.DataFrame:
    """Search for the best ARIMA order by comparing AIC or BIC.

    When *use_auto* is True the search is delegated to ``pmdarima.auto_arima``,
    which uses a stepwise heuristic. Otherwise a full grid over
    ``p_range × {d} × q_range`` is evaluated with ``statsmodels.ARIMA``.

    Parameters
    ----------
    series:
        Stationary (or raw) series.  If *d* > 0 the series is differenced
        internally during each fit when *use_auto* is False.
    d:
        Integration order to embed in each candidate model.
    p_range:
        Inclusive (min, max) for AR order search.
    q_range:
        Inclusive (min, max) for MA order search.
    criterion:
        Information criterion used for ranking (``"aic"`` or ``"bic"``).
    use_auto:
        Use ``pmdarima.auto_arima`` for the search (recommended).
    seasonal:
        Whether to include seasonal terms in ``auto_arima``.

    Returns
    -------
    pd.DataFrame
        Columns: ``p``, ``d``, ``q``, ``aic``, ``bic``, ``converged``.
        Sorted ascending by the chosen *criterion*.

    Raises
    ------
    ValueError
        If *criterion* is not ``"aic"`` or ``"bic"``.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic', got '{criterion}'.")

    s = series.dropna()

    if use_auto:
        try:
            import pmdarima as pm  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pmdarima is required for use_auto=True. "
                "Install with: poetry add pmdarima"
            ) from exc

        model = pm.auto_arima(
            s,
            d=d if d > 0 else None,
            start_p=p_range[0],
            max_p=p_range[1],
            start_q=q_range[0],
            max_q=q_range[1],
            information_criterion=criterion,
            stepwise=True,
            seasonal=seasonal,
            error_action="ignore",
            suppress_warnings=True,
        )
        # Collect result as a single-row summary DataFrame
        order = model.order
        rows = [
            {
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "aic": model.aic(),
                "bic": model.bic(),
                "converged": True,
            }
        ]
        return pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)

    # --- Manual grid search ---
    rows = []
    p_vals = range(p_range[0], p_range[1] + 1)
    q_vals = range(q_range[0], q_range[1] + 1)

    for p, q in itertools.product(p_vals, q_vals):
        if p == 0 and q == 0:
            continue
        try:
            res = ARIMA(s, order=(p, d, q)).fit()
            rows.append(
                {
                    "p": p,
                    "d": d,
                    "q": q,
                    "aic": res.aic,
                    "bic": res.bic,
                    "converged": res.mle_retvals.get("converged", True)
                    if hasattr(res, "mle_retvals") and res.mle_retvals
                    else True,
                }
            )
        except Exception:  # noqa: BLE001
            rows.append({"p": p, "d": d, "q": q, "aic": np.nan, "bic": np.nan, "converged": False})

    df = pd.DataFrame(rows)
    df = df.dropna(subset=[criterion])
    return df.sort_values(criterion).reset_index(drop=True)


def choose_best_order(df, criterion="aic", delta=5):
    """Choose the best ARIMA order. Consider both the information criterion and the complexity of the model.
    """
    df = df[df["converged"] == True].copy()
    if df.empty:
        raise ValueError("No valid ARIMA models found.")

    # Step 1: 找最小 AIC
    best_ic = df[criterion].min()

    # Step 2: 保留“接近最优”的模型
    candidates = df[df[criterion] <= best_ic + delta]

    # Step 3: 在这些模型中选最简单的
    candidates["complexity"] = candidates["p"] + candidates["q"]
    print('candidates')
    print(candidates)
    best_row = candidates.sort_values(["complexity", criterion]).iloc[0]

    print(f'最优阶次（AIC 最小）: ARIMA{best_row["p"]},{best_row["d"]},{best_row["q"]}')
    print(f'  AIC = {best_row["aic"]:.2f},  BIC = {best_row["bic"]:.2f}')

    return (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))


# ---------------------------------------------------------------------------
# 3. Model fitting
# ---------------------------------------------------------------------------


def fit_arima(
    series: pd.Series,
    order: ArimaOrder,
) -> ARIMAResultsWrapper:
    """Fit a final ARIMA model with the given order.

    Parameters
    ----------
    series:
        Time series to fit (should be long enough for the chosen order).
    order:
        ``(p, d, q)`` tuple.

    Returns
    -------
    ARIMAResultsWrapper
        Fitted statsmodels ARIMA result object.

    Raises
    ------
    ValueError
        If *series* has fewer observations than ``p + d + q + 1``.
    """
    s = series.dropna()
    p, d, q = order
    min_obs = p + d + q + 1
    if len(s) < min_obs:
        raise ValueError(
            f"Series has {len(s)} observations; need at least {min_obs} for ARIMA{order}."
        )
    result = ARIMA(s, order=order).fit()
    return result


# ---------------------------------------------------------------------------
# 4. Forecasting
# ---------------------------------------------------------------------------


def forecast_arima(
    result: ARIMAResultsWrapper,
    steps: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Produce an N-step-ahead forecast from a fitted ARIMA model.

    Parameters
    ----------
    result:
        Fitted ARIMA result (from :func:`fit_arima`).
    steps:
        Number of periods to forecast.
    alpha:
        Significance level for the confidence interval (default: 0.05 → 95 % CI).

    Returns
    -------
    pd.DataFrame
        Columns: ``forecast``, ``lower_ci``, ``upper_ci``.
        Index: integer steps 1 … *steps* (or datetime if original series had a
        DatetimeIndex).

    Raises
    ------
    ValueError
        If *steps* < 1.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")

    pred = result.get_forecast(steps=steps)
    summary = pred.summary_frame(alpha=alpha)

    df = pd.DataFrame(
        {
            "forecast": summary["mean"],
            "lower_ci": summary["mean_ci_lower"],
            "upper_ci": summary["mean_ci_upper"],
        }
    )
    df.index.name = "step"
    return df


# ---------------------------------------------------------------------------
# 5. Rolling forecasts
# ---------------------------------------------------------------------------


def rolling_forecast_arima(
    train_data: pd.Series,
    test_data: pd.Series,
    order: ArimaOrder,
    alpha: float = 0.05,
    window: int | None = None,
    refit_every: int = 1,
    verbose: bool = False,
) -> tuple[pd.DataFrame, ARIMAResultsWrapper]:
    """Generate 1-step-ahead rolling ARIMA forecasts.

    Strategy
    --------
    - Initial fit on train_data.
    - For each test point:
      1) forecast next step using information up to t-1
      2) record forecast against the realized value at t
      3) update model state with realized value
      4) every `refit_every` steps, fully re-fit on rolling/expanding window

    Parameters
    ----------
    train_data:
        In-sample training series.
    test_data:
        Out-of-sample series to forecast one step ahead.
    order:
        Final ARIMA order (p, d, q).
    alpha:
        Significance level for forecast confidence intervals.
    window:
        Rolling window size. If None, uses len(train_data).
        If 0, uses expanding window.
    refit_every:
        Full re-fit frequency. Must be >= 1.
        - 1: re-fit every step
        - >1: update every step, re-fit every k steps
    verbose:
        Whether to print periodic progress.

    Returns
    -------
    forecast_df, initial_result
    """
    train = train_data.dropna().copy()
    test = test_data.dropna().copy()

    if train.empty:
        raise ValueError("train_data must not be empty.")
    if test.empty:
        raise ValueError("test_data must not be empty.")
    if window is not None and window < 0:
        raise ValueError("window must be >= 0 when provided.")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1.")

    train_size = len(train)
    effective_window = train_size if window is None else window

    # Full history used for rolling-window re-fit decisions
    full_series = pd.concat([train, test])

    initial_result = fit_arima(
        train,
        order=order,
    )
    current_result = initial_result

    records: list[dict[str, float | pd.Timestamp]] = []

    for i, t in enumerate(range(train_size, len(full_series)), start=1):
        current_date = full_series.index[t]
        actual_value = float(full_series.iloc[t])

        # 1-step forecast using only information up to t-1
        forecast_row = forecast_arima(current_result, steps=1, alpha=alpha).iloc[0]

        records.append(
            {
                "date": current_date,
                "forecast": float(forecast_row["forecast"]),
                "lower_ci": float(forecast_row["lower_ci"]),
                "upper_ci": float(forecast_row["upper_ci"]),
                "actual": actual_value,
            }
        )

        # Update with realized observation AFTER forecasting
        new_obs = pd.Series([actual_value], index=[current_date])

        should_refit = (i % refit_every == 0)

        if should_refit:
            if effective_window > 0:
                refit_train = full_series.iloc[max(0, t + 1 - effective_window) : t + 1]
            else:
                refit_train = full_series.iloc[: t + 1]

            if verbose and (i % max(refit_every, 30) == 0 or i == 1):
                print(f"Refitting ARIMA at step {i}/{len(test)} on {current_date}...")

            current_result = fit_arima(
                refit_train,
                order=order,
            )
        else:
            # Cheap state update without full parameter re-estimation
            current_result = current_result.append(new_obs, refit=False)

    forecast_df = pd.DataFrame(records).set_index("date")
    return forecast_df, initial_result


# ---------------------------------------------------------------------------
# 6. Residual diagnostics
# ---------------------------------------------------------------------------


def residual_diagnostics(
    result: ARIMAResultsWrapper,
    lags: int = 20,
    save_path: Optional[str | Path] = None,
) -> dict:
    """Compute residual diagnostics and optionally produce a 4-panel plot.

    Diagnostics computed:
    - Ljung-Box test for residual autocorrelation
    - Ljung-Box test for squared residuals (ARCH effect check)
    - Basic residual summary statistics

    Parameters
    ----------
    result:
        Fitted ARIMA result.
    lags:
        Number of lags for the Ljung-Box tests.
    save_path:
        If provided, the 4-panel diagnostic figure is saved here.

    Returns
    -------
    dict with keys:
        ``residuals``      – pd.Series of model residuals
        ``ljung_box``      – pd.DataFrame of Ljung-Box test on residuals
        ``ljung_box_sq``   – pd.DataFrame of Ljung-Box test on squared residuals
        ``summary_stats``  – pd.Series with mean, std, skew, kurtosis
        ``figure``         – matplotlib Figure (or None if save_path is None and
                             called without display)
    """
    residuals: pd.Series = result.resid.dropna()
    residuals.name = "residuals"

    lb = acorr_ljungbox(residuals, lags=lags, return_df=True)
    lb_sq = acorr_ljungbox(residuals ** 2, lags=lags, return_df=True)

    from scipy import stats as scipy_stats  # local import to keep top-level clean

    stats_summary = pd.Series(
        {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "skew": float(scipy_stats.skew(residuals)),
            "kurtosis": float(scipy_stats.kurtosis(residuals)),
        },
        name="residual_stats",
    )

    fig = _plot_diagnostics(residuals, save_path=save_path)

    return {
        "residuals": residuals,
        "ljung_box": lb,
        "ljung_box_sq": lb_sq,
        "summary_stats": stats_summary,
        "figure": fig,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plot_diagnostics(
    residuals: pd.Series,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Produce a 4-panel residual diagnostic figure.

    Panels: residuals over time, histogram, ACF, Q-Q plot.
    """
    from scipy import stats as scipy_stats
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: Residuals over time
    axes[0, 0].plot(residuals.index, residuals.values, linewidth=0.6, color="steelblue")
    axes[0, 0].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title("Residuals")
    axes[0, 0].set_xlabel("Date / Index")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Histogram + KDE
    axes[0, 1].hist(residuals, bins=40, density=True, alpha=0.5, color="steelblue")
    x = np.linspace(residuals.min(), residuals.max(), 200)
    mu, sigma = residuals.mean(), residuals.std()
    axes[0, 1].plot(x, scipy_stats.norm.pdf(x, mu, sigma), color="red", linewidth=1.5, label="Normal fit")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: ACF of residuals
    plot_acf(residuals, lags=30, ax=axes[1, 0], zero=False, title="ACF of Residuals")
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Q-Q plot
    scipy_stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot vs Normal")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)

    return fig
