"""
GARCH modeling module for return-like stationary time series.

Pipeline:
  1. prepare_returns     – validate input and check for ARCH effects
  2. search_garch_order  – grid-search GARCH(p, q) by AIC/BIC
  3. fit_garch           – fit a final GARCH model with explicit order
  4. forecast_garch      – produce N-step-ahead variance/volatility forecasts
  5. garch_diagnostics   – compute and optionally plot standardized-residual diagnostics
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import itertools

import matplotlib.pyplot as plt
import src.plot_config  # noqa: F401 — configures CJK font for all figures

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from arch.univariate.base import ARCHModelResult

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

GarchOrder = tuple[int, int]  # (p, q)

# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------


def prepare_returns(
    series: pd.Series,
    significance: float = 0.05,
    min_obs: int = 50,
) -> pd.Series:
    """Validate a return-like series and warn if no ARCH effects are detected.

    This function does NOT difference the series – GARCH models require a
    stationary, return-like input (e.g. log-returns), not raw price levels.

    Parameters
    ----------
    series:
        Numeric return series (e.g. daily log-returns).
    significance:
        p-value threshold for the ARCH-LM test.  A warning is issued when the
        p-value exceeds this threshold (no evidence of volatility clustering).
    min_obs:
        Minimum number of valid observations required.

    Returns
    -------
    pd.Series
        Cleaned series with NaN values dropped and index preserved.

    Raises
    ------
    ValueError
        If the series is empty, too short, or contains non-finite values after
        cleaning.
    """
    s = series.dropna()

    if s.empty:
        raise ValueError("Input series is empty after dropping NaN.")
    if len(s) < min_obs:
        raise ValueError(
            f"Series too short for GARCH: {len(s)} observations (need >= {min_obs})."
        )
    if not np.isfinite(s).all():
        raise ValueError("Series contains non-finite values (inf or -inf).")

    # ARCH-LM test via Ljung-Box on squared demeaned returns
    demeaned = s - s.mean()
    lb_sq = acorr_ljungbox(demeaned**2, lags=10, return_df=True)
    min_pval = lb_sq["lb_pvalue"].min()
    if min_pval > significance:
        import warnings

        warnings.warn(
            f"No significant ARCH effect detected (min LB p-value on r² = {min_pval:.4f}). "
            "GARCH model may not be appropriate for this series.",
            UserWarning,
            stacklevel=2,
        )

    return s


# ---------------------------------------------------------------------------
# 2. Order search
# ---------------------------------------------------------------------------


def search_garch_order(
    series: pd.Series,
    p_range: tuple[int, int] = (1, 3),
    q_range: tuple[int, int] = (1, 3),
    dist: str = "normal",
    criterion: Literal["aic", "bic"] = "aic",
    vol: str = "Garch",
) -> pd.DataFrame:
    """Grid-search GARCH(p, q) specifications and rank by AIC or BIC.

    Parameters
    ----------
    series:
        Stationary return series (e.g. from :func:`prepare_returns`).
    p_range:
        Inclusive (min, max) for the GARCH lag order *p* (variance lags).
    q_range:
        Inclusive (min, max) for the ARCH lag order *q* (squared-error lags).
    dist:
        Error distribution passed to ``arch_model``: ``"normal"``, ``"t"``,
        ``"skewt"``, etc.
    criterion:
        Information criterion for ranking (``"aic"`` or ``"bic"``).
    vol:
        Volatility model type: ``"Garch"``, ``"EGarch"``, ``"GJR-Garch"``, etc.

    Returns
    -------
    pd.DataFrame
        Columns: ``p``, ``q``, ``aic``, ``bic``, ``converged``.
        Sorted ascending by the chosen *criterion*.

    Raises
    ------
    ValueError
        If *criterion* is not ``"aic"`` or ``"bic"``.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic', got '{criterion}'.")

    s = series.dropna()
    rows = []

    for p, q in itertools.product(
        range(p_range[0], p_range[1] + 1),
        range(q_range[0], q_range[1] + 1),
    ):
        try:
            res = arch_model(s, vol=vol, p=p, q=q, dist=dist).fit(
                disp="off", show_warning=False
            )
            rows.append(
                {
                    "p": p,
                    "q": q,
                    "aic": res.aic,
                    "bic": res.bic,
                    "converged": res.convergence_flag == 0,
                }
            )
        except Exception:  # noqa: BLE001
            rows.append({"p": p, "q": q, "aic": np.nan, "bic": np.nan, "converged": False})

    df = pd.DataFrame(rows).dropna(subset=[criterion])
    return df.sort_values(criterion).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Model fitting
# ---------------------------------------------------------------------------


def fit_garch(
    series: pd.Series,
    order: GarchOrder = (1, 1),
    dist: str = "normal",
    vol: str = "Garch",
    mean: str = "Constant",
) -> ARCHModelResult:
    """Fit a GARCH model with the specified order and return the result object.

    Parameters
    ----------
    series:
        Stationary return series.
    order:
        ``(p, q)`` – GARCH lags (p) and ARCH lags (q).
    dist:
        Error distribution: ``"normal"``, ``"t"``, ``"skewt"``, ``"ged"``.
    vol:
        Volatility process: ``"Garch"``, ``"EGarch"``, ``"GJR-Garch"``.
    mean:
        Mean model: ``"Constant"``, ``"Zero"``, ``"AR"``.

    Returns
    -------
    ARCHModelResult
        Fitted arch result object with attributes such as ``conditional_volatility``,
        ``aic``, ``bic``, and method ``forecast()``.

    Raises
    ------
    ValueError
        If *series* has fewer observations than 20 × (p + q).
    """
    s = series.dropna()
    p, q = order
    min_obs = max(50, 20 * (p + q))
    if len(s) < min_obs:
        raise ValueError(
            f"Series has {len(s)} observations; need at least {min_obs} for GARCH{order}."
        )

    model = arch_model(s, vol=vol, p=p, q=q, dist=dist, mean=mean)
    result = model.fit(disp="off", show_warning=False)
    return result


# ---------------------------------------------------------------------------
# 4. Forecasting
# ---------------------------------------------------------------------------


def forecast_garch(
    result: ARCHModelResult,
    steps: int = 10,
) -> pd.DataFrame:
    """Produce N-step-ahead variance and volatility forecasts.

    Parameters
    ----------
    result:
        Fitted GARCH result (from :func:`fit_garch`).
    steps:
        Number of periods to forecast ahead.

    Returns
    -------
    pd.DataFrame
        Columns: ``forecast_variance``, ``forecast_volatility`` (= sqrt of variance).
        Index: integer 1 … *steps* labelled ``step``.

    Raises
    ------
    ValueError
        If *steps* < 1.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")

    fc = result.forecast(horizon=steps, reindex=False)
    variance = fc.variance.iloc[-1].values
    volatility = np.sqrt(variance)

    df = pd.DataFrame(
        {
            "forecast_variance": variance,
            "forecast_volatility": volatility,
        },
        index=pd.RangeIndex(1, steps + 1, name="step"),
    )
    return df


# ---------------------------------------------------------------------------
# 5. Rolling forecasts
# ---------------------------------------------------------------------------


def rolling_forecast_garch(
    train_data: pd.Series,
    test_data: pd.Series,
    order: GarchOrder = (1, 1),
    dist: str = "t",
    vol: str = "Garch",
    mean: str = "Constant",
    scale: float = 100.0,
    window: int | None = None,
    refit_every: int = 1,
) -> tuple[pd.DataFrame, ARCHModelResult]:
    """Generate 1-step-ahead rolling GARCH volatility forecasts.

    At each forecast origin ``t`` in the test period, the model is re-fitted
    on the most recent ``window`` observations and a 1-step-ahead variance
    forecast is produced.  The rolling window defaults to ``len(train_data)``
    (fixed-length sliding window).

    The initial fit on ``train_data`` is also returned so the caller can
    inspect in-sample diagnostics without a separate :func:`fit_garch` call.

    Parameters
    ----------
    train_data:
        In-sample return series used to establish the first training window.
        Must be stationary (e.g. daily log-returns).
    test_data:
        Out-of-sample return series.  Forecasts are produced for each
        observation in this series.
    order:
        ``(p, q)`` GARCH order.
    dist:
        Error distribution passed to ``arch_model``.
    vol:
        Volatility process type (``"Garch"``, ``"GJR-Garch"``, ``"EGarch"``).
    mean:
        Mean model (``"Constant"``, ``"Zero"``, ``"AR"``).
    scale:
        Multiplicative scaling applied to the series before fitting to
        improve numerical stability.  Results are rescaled on output.
    window:
        Size of the rolling training window.  If ``None`` (default), uses
        ``len(train_data)``, so the window slides forward with a fixed length
        equal to the initial training period.  Pass ``0`` to use an expanding
        window (all data up to the current forecast origin).
    refit_every:
        Re-fit the model every *refit_every* steps (default ``1`` = refit at
        every forecast origin).  Larger values trade parameter freshness for
        speed.

    Returns
    -------
    forecast_df : pd.DataFrame
        Indexed by the dates of the out-of-sample period.  Columns:

        * ``predicted_variance``   – 1-step-ahead conditional variance
        * ``predicted_volatility`` – square root of ``predicted_variance``
        * ``realized_return``      – actual return at that date

    initial_result : ARCHModelResult
        GARCH result fitted on the full ``train_data``.  Use for in-sample
        diagnostics (e.g. :func:`garch_diagnostics`) without an extra
        :func:`fit_garch` call.

    Raises
    ------
    ValueError
        If ``train_data`` is too short, ``test_data`` is empty, or
        ``refit_every`` < 1.
    """
    train = train_data.dropna()
    test = test_data.dropna()

    if len(train) < 30:
        raise ValueError("train_data must have at least 30 observations.")
    if len(test) == 0:
        raise ValueError("test_data must not be empty.")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1.")

    train_size = len(train)
    effective_window = train_size if window is None else window

    # Concatenate for rolling slicing
    s = pd.concat([train, test])

    # Fit once on the full training set – returned as initial_result
    initial_result = fit_garch(
        series=train * scale,
        order=order,
        dist=dist,
        vol=vol,
        mean=mean,
    )

    records: list[dict] = []
    last_result: ARCHModelResult = initial_result

    for i, t in enumerate(range(train_size, len(s))):
        if effective_window > 0:
            window_train = s.iloc[max(0, t - effective_window) : t]
        else:
            window_train = s.iloc[:t]  # expanding window

        # Refit when scheduled (skip i==0 since we already have initial_result)
        if i > 0 and i % refit_every == 0:
            last_result = fit_garch(
                series=window_train * scale,
                order=order,
                dist=dist,
                vol=vol,
                mean=mean,
            )

        fc = forecast_garch(last_result, steps=1)
        pred_var = fc.iloc[0]["forecast_variance"] / (scale**2)
        pred_vol = np.sqrt(pred_var)

        records.append(
            {
                "date": s.index[t],
                "predicted_variance": pred_var,
                "predicted_volatility": pred_vol,
                "realized_return": s.iloc[t],
            }
        )

    forecast_df = pd.DataFrame(records).set_index("date")
    return forecast_df, initial_result


# ---------------------------------------------------------------------------
# 6. Diagnostics
# ---------------------------------------------------------------------------


def garch_diagnostics(
    result: ARCHModelResult,
    lags: int = 20,
    save_path: Optional[str | Path] = None,
) -> dict:
    """Compute standardized-residual diagnostics and optionally plot them.

    Diagnostics computed:
    - Ljung-Box test on standardized residuals (serial autocorrelation)
    - Ljung-Box test on squared standardized residuals (remaining ARCH effect)
    - Basic summary statistics (mean, std, skew, excess kurtosis)

    Parameters
    ----------
    result:
        Fitted GARCH result.
    lags:
        Number of lags for Ljung-Box tests.
    save_path:
        If provided, the 4-panel diagnostic figure is saved here.

    Returns
    -------
    dict with keys:
        ``std_residuals``   – pd.Series of standardized residuals
        ``conditional_vol`` – pd.Series of conditional volatility
        ``ljung_box``       – pd.DataFrame of LB test on std residuals
        ``ljung_box_sq``    – pd.DataFrame of LB test on squared std residuals
        ``summary_stats``   – pd.Series with mean, std, skew, kurtosis
        ``figure``          – matplotlib Figure (or None)
    """
    std_resid: pd.Series = result.std_resid.dropna()
    std_resid.name = "std_residuals"

    cond_vol: pd.Series = result.conditional_volatility.dropna()
    cond_vol.name = "conditional_volatility"

    lb = acorr_ljungbox(std_resid, lags=lags, return_df=True)
    lb_sq = acorr_ljungbox(std_resid**2, lags=lags, return_df=True)

    stats_summary = pd.Series(
        {
            "mean": float(std_resid.mean()),
            "std": float(std_resid.std()),
            "skew": float(scipy_stats.skew(std_resid)),
            "kurtosis": float(scipy_stats.kurtosis(std_resid)),
        },
        name="std_residual_stats",
    )

    fig = _plot_garch_diagnostics(std_resid, cond_vol, save_path=save_path)

    return {
        "std_residuals": std_resid,
        "conditional_vol": cond_vol,
        "ljung_box": lb,
        "ljung_box_sq": lb_sq,
        "summary_stats": stats_summary,
        "figure": fig,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plot_garch_diagnostics(
    std_resid: pd.Series,
    cond_vol: pd.Series,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Produce a 4-panel diagnostic figure for a fitted GARCH model.

    Panels:
    1. Conditional volatility over time
    2. Standardized residuals histogram vs normal
    3. ACF of standardized residuals
    4. Q-Q plot of standardized residuals
    """
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: Conditional volatility
    axes[0, 0].plot(cond_vol.index, cond_vol.values, linewidth=0.8, color="steelblue")
    axes[0, 0].set_title("Conditional Volatility")
    axes[0, 0].set_xlabel("Date / Index")
    axes[0, 0].set_ylabel("Volatility")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Histogram of standardized residuals vs normal
    axes[0, 1].hist(std_resid, bins=40, density=True, alpha=0.5, color="steelblue")
    x = np.linspace(std_resid.min(), std_resid.max(), 200)
    axes[0, 1].plot(
        x,
        scipy_stats.norm.pdf(x, 0, 1),
        color="red",
        linewidth=1.5,
        label="N(0,1)",
    )
    axes[0, 1].set_title("Std Residual Distribution")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: ACF of standardized residuals
    plot_acf(std_resid, lags=30, ax=axes[1, 0], zero=False, title="ACF of Std Residuals")
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Q-Q plot
    scipy_stats.probplot(std_resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot vs Normal")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)

    return fig
