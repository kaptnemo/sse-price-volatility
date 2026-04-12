"""
src/evaluate.py
---------------
Evaluation utilities for ARIMA mean forecasts and GARCH volatility forecasts.

Three public areas
------------------
1. Point forecast metrics  (ARIMA / any regression):
   - evaluate_point_forecast

2. Volatility forecast metrics (GARCH / GJR-GARCH):
   - evaluate_vol_forecast
   - compare_vol_forecasts

3. VaR back-testing (Kupiec POF + Christoffersen CC tests):
   - compute_var_t
   - kupiec_pof
   - christoffersen_cc
   - var_backtest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, t as t_dist


# ---------------------------------------------------------------------------
# 1. Point forecast evaluation
# ---------------------------------------------------------------------------

def evaluate_point_forecast(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE for a point forecast series.

    Parameters
    ----------
    actual:
        Observed values (indexed by date).
    predicted:
        Predicted values; automatically aligned to *actual* by index.

    Returns
    -------
    dict with keys ``MAE``, ``RMSE``, ``MAPE (%)``.

    Raises
    ------
    ValueError
        If no overlapping index entries are found after alignment.
    """
    pred_aligned = predicted.reindex(actual.index).dropna()
    act_aligned = actual.reindex(pred_aligned.index).dropna()
    common = pred_aligned.index.intersection(act_aligned.index)
    if len(common) == 0:
        raise ValueError("actual and predicted share no common index values.")

    a = act_aligned.loc[common].to_numpy(dtype=float)
    p = pred_aligned.loc[common].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    # Guard against zero actuals
    mask = a != 0
    mape = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


# ---------------------------------------------------------------------------
# 2. Volatility forecast evaluation
# ---------------------------------------------------------------------------

def evaluate_vol_forecast(
    rolling_df: pd.DataFrame,
    returns: pd.Series,
    model_name: str = "model",
) -> dict[str, float]:
    """Evaluate a rolling GARCH-type volatility forecast.

    Uses ``|r_t|`` as the volatility proxy for RMSE / MAE / Corr, and
    ``r_t^2`` as the variance proxy for QLIKE (Patton 2011).

    Parameters
    ----------
    rolling_df:
        DataFrame returned by ``rolling_forecast_garch``.  Must contain
        columns ``predicted_volatility`` (σ_t) and ``predicted_variance`` (σ_t²).
    returns:
        Out-of-sample log-return series; aligned to *rolling_df* by index.
    model_name:
        Label used as the ``model`` key in the returned dict.

    Returns
    -------
    dict with keys ``model``, ``RMSE``, ``MAE``, ``QLIKE``, ``Corr``,
    ``n_obs``.
    """
    _check_columns(rolling_df, ["predicted_volatility", "predicted_variance"])

    common = rolling_df.index.intersection(returns.index)
    if len(common) == 0:
        raise ValueError("rolling_df and returns share no common index values.")

    pred_vol = rolling_df.loc[common, "predicted_volatility"].to_numpy(dtype=float)
    pred_var = rolling_df.loc[common, "predicted_variance"].to_numpy(dtype=float)
    r = returns.reindex(common).to_numpy(dtype=float)
    abs_r = np.abs(r)
    r2 = r ** 2

    rmse = float(np.sqrt(np.mean((pred_vol - abs_r) ** 2)))
    mae = float(np.mean(np.abs(pred_vol - abs_r)))
    qlike = float(np.mean(np.log(pred_var) + r2 / pred_var))
    corr = float(np.corrcoef(pred_vol, abs_r)[0, 1])

    return {
        "model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "QLIKE": qlike,
        "Corr": corr,
        "n_obs": len(common),
    }


def compare_vol_forecasts(
    models: dict[str, pd.DataFrame],
    returns: pd.Series,
) -> pd.DataFrame:
    """Compare multiple rolling volatility forecasts.

    Parameters
    ----------
    models:
        Mapping of ``{model_name: rolling_df}``.  Each DataFrame must have
        ``predicted_volatility`` and ``predicted_variance`` columns.
    returns:
        Out-of-sample log-return series.

    Returns
    -------
    DataFrame indexed by model name, columns: RMSE, MAE, QLIKE, Corr, n_obs.
    """
    rows = [evaluate_vol_forecast(df, returns, name) for name, df in models.items()]
    result = pd.DataFrame(rows).set_index("model")
    return result


# ---------------------------------------------------------------------------
# 3. VaR back-testing
# ---------------------------------------------------------------------------

def compute_var_t(
    predicted_volatility: pd.Series,
    nu: float,
    alpha: float = 0.01,
) -> pd.Series:
    """Compute Value-at-Risk from a standardised Student-t distribution.

    The ``arch`` package uses a *standardised* t distribution (mean 0,
    variance 1), whose α-quantile is::

        q_α = t.ppf(α, df=ν) × √((ν − 2) / ν)

    The VaR is defined so that P(r_t < −VaR_t) = α:

        VaR_{α,t} = −(μ + σ_t × q_α)   with μ ≈ 0.

    Parameters
    ----------
    predicted_volatility:
        Conditional volatility series σ_t (must be positive).
    nu:
        Degrees-of-freedom parameter ν > 2.
    alpha:
        Tail probability (e.g. 0.01 for 1 % VaR).

    Returns
    -------
    pd.Series of VaR values (positive numbers = maximum expected loss).

    Raises
    ------
    ValueError
        If ``nu`` ≤ 2 or ``alpha`` not in (0, 1).
    """
    if nu <= 2:
        raise ValueError(f"nu must be > 2, got {nu}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    q = float(t_dist.ppf(alpha, df=nu) * np.sqrt((nu - 2) / nu))
    return -(predicted_volatility * q)


def kupiec_pof(T: int, x: int, alpha: float) -> dict[str, float]:
    """Kupiec (1995) Proportion of Failures (POF) test.

    Tests H₀: E[breach rate] = α using a likelihood-ratio statistic that
    follows χ²(1) under the null.

    Parameters
    ----------
    T:
        Total number of out-of-sample observations.
    x:
        Observed number of VaR breaches (r_t < −VaR_t).
    alpha:
        Nominal VaR level (e.g. 0.01).

    Returns
    -------
    dict with keys ``LR`` (test statistic) and ``p_value``.
    """
    if x == 0:
        lr = -2.0 * T * np.log(1.0 - alpha)
    elif x == T:
        lr = -2.0 * T * np.log(alpha)
    else:
        pi_hat = x / T
        lr = -2.0 * (x * np.log(alpha / pi_hat) + (T - x) * np.log((1 - alpha) / (1 - pi_hat)))

    p_value = float(1.0 - chi2.cdf(lr, df=1))
    return {"LR": float(lr), "p_value": p_value}


def christoffersen_cc(
    hits: np.ndarray | pd.Series,
    alpha: float,
) -> dict[str, float]:
    """Christoffersen (1998) Conditional Coverage test.

    Decomposes VaR model adequacy into two orthogonal components:

    - **Unconditional coverage** (LR_uc): breach rate ≈ α  (χ²(1))
    - **Independence** (LR_ind): breaches are serially independent  (χ²(1))
    - **Conditional coverage** (LR_cc = LR_uc + LR_ind)  (χ²(2))

    Parameters
    ----------
    hits:
        Binary series where 1 = VaR breach on day *t*, 0 = no breach.
        Length *T*.  The transition window uses pairs (hits[t-1], hits[t]),
        so the effective sample is T − 1.
    alpha:
        Nominal VaR level (e.g. 0.01).

    Returns
    -------
    dict with keys:
        ``n_00``, ``n_01``, ``n_10``, ``n_11``  – transition counts,
        ``pi_01``, ``pi_11``  – conditional breach probabilities,
        ``LR_uc``, ``p_uc``  – unconditional coverage stat & p-value,
        ``LR_ind``, ``p_ind``  – independence stat & p-value,
        ``LR_cc``, ``p_cc``  – conditional coverage stat & p-value.

    Notes
    -----
    The convention ``0 × log(0) = 0`` (via :func:`_safe_log`) avoids
    undefined terms when a transition count is zero.
    """
    h = np.asarray(hits, dtype=float)
    if h.ndim != 1 or len(h) < 2:
        raise ValueError("hits must be a 1-D array with at least 2 elements.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    h0 = h[:-1]  # state at t-1
    h1 = h[1:]   # state at t

    n_00 = int(((h0 == 0) & (h1 == 0)).sum())
    n_01 = int(((h0 == 0) & (h1 == 1)).sum())
    n_10 = int(((h0 == 1) & (h1 == 0)).sum())
    n_11 = int(((h0 == 1) & (h1 == 1)).sum())

    T = len(h1)  # transition window length
    x = n_01 + n_11  # total breaches in transition window

    # Unconditional breach probability in the transition window
    pi_hat = x / T if T > 0 else alpha

    # Conditional breach probabilities
    pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0.0
    pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0.0

    # LR_uc on transition window (same formula as Kupiec but over T)
    if x == 0:
        lr_uc = float(-2.0 * T * np.log(1.0 - alpha))
    elif x == T:
        lr_uc = float(-2.0 * T * np.log(alpha))
    else:
        lr_uc = float(
            -2.0 * (x * np.log(alpha / pi_hat) + (T - x) * np.log((1 - alpha) / (1 - pi_hat)))
        )

    # LR_ind: likelihood-ratio for independence vs. unrestricted transition matrix
    ll_restricted = (
        x * _safe_log(pi_hat) + (T - x) * _safe_log(1.0 - pi_hat)
    )
    ll_unrestricted = (
        n_01 * _safe_log(pi_01)
        + n_00 * _safe_log(1.0 - pi_01)
        + n_11 * _safe_log(pi_11)
        + n_10 * _safe_log(1.0 - pi_11)
    )
    lr_ind = float(2.0 * (ll_unrestricted - ll_restricted))

    lr_cc = lr_uc + lr_ind

    p_uc = float(1.0 - chi2.cdf(lr_uc, df=1))
    p_ind = float(1.0 - chi2.cdf(max(lr_ind, 0.0), df=1))
    p_cc = float(1.0 - chi2.cdf(max(lr_cc, 0.0), df=2))

    return {
        "n_00": n_00, "n_01": n_01, "n_10": n_10, "n_11": n_11,
        "pi_01": round(pi_01, 4),
        "pi_11": round(pi_11, 4),
        "LR_uc": round(lr_uc, 4),
        "p_uc": round(p_uc, 4),
        "LR_ind": round(lr_ind, 4),
        "p_ind": round(p_ind, 4),
        "LR_cc": round(lr_cc, 4),
        "p_cc": round(p_cc, 4),
    }


def var_backtest(
    rolling_df: pd.DataFrame,
    returns: pd.Series,
    nu: float,
    alpha_levels: list[float] | None = None,
    model_name: str = "model",
) -> pd.DataFrame:
    """Run VaR back-tests at multiple confidence levels using Kupiec POF and
    Christoffersen Conditional Coverage tests.

    Parameters
    ----------
    rolling_df:
        DataFrame with a ``predicted_volatility`` column.
    returns:
        Out-of-sample log-return series; aligned by index.
    nu:
        Student-t degrees of freedom from the fitted GARCH model.
    alpha_levels:
        List of tail probabilities to test.  Defaults to ``[0.01, 0.05]``.
    model_name:
        Label for the model column.

    Returns
    -------
    DataFrame with columns: model, alpha, expected_breaches, actual_breaches,
    breach_rate, LR, p_value, reject_h0,
    LR_ind, p_ind, reject_ind, LR_cc, p_cc, reject_cc.
    """
    _check_columns(rolling_df, ["predicted_volatility"])
    if alpha_levels is None:
        alpha_levels = [0.01, 0.05]

    common = rolling_df.index.intersection(returns.index)
    if len(common) == 0:
        raise ValueError("rolling_df and returns share no common index values.")

    pred_vol = rolling_df.loc[common, "predicted_volatility"]
    actual = returns.reindex(common)
    T = len(common)

    rows = []
    for alpha in alpha_levels:
        var_series = compute_var_t(pred_vol, nu=nu, alpha=alpha)
        hits = (actual < -var_series).astype(int).to_numpy()
        x = int(hits.sum())
        kupiec = kupiec_pof(T, x, alpha)
        cc = christoffersen_cc(hits, alpha)
        rows.append(
            {
                "model": model_name,
                "alpha": alpha,
                "expected_breaches": round(T * alpha, 1),
                "actual_breaches": x,
                "breach_rate": round(x / T, 4),
                "LR": round(kupiec["LR"], 4),
                "p_value": round(kupiec["p_value"], 4),
                "reject_h0": kupiec["p_value"] < 0.05,
                "LR_ind": cc["LR_ind"],
                "p_ind": cc["p_ind"],
                "reject_ind": cc["p_ind"] < 0.05,
                "LR_cc": cc["LR_cc"],
                "p_cc": cc["p_cc"],
                "reject_cc": cc["p_cc"] < 0.05,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_log(x: float) -> float:
    """Return log(x) with the convention 0 * log(0) = 0 (returns 0 when x=0)."""
    return float(np.log(x)) if x > 0 else 0.0


def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
