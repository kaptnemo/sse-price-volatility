"""
Exploratory data analysis utilities for time series data.

Functions are grouped into three concerns:
  1. Summary statistics and stationarity tests
  2. Time series plots (levels, returns, rolling stats)
  3. Distribution plots (histogram/KDE, ACF/PACF, Q-Q)

All plot functions return the matplotlib Figure and optionally save it to disk.
Figures directory is created automatically when save_path is supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import src.plot_config  # noqa: F401 — configures CJK font for all figures
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGURES_DIR = Path("outputs/figures")


def _save_fig(fig: Figure, save_path: Optional[str | Path]) -> None:
    """Save *fig* to *save_path*, creating parent directories as needed."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)


# ---------------------------------------------------------------------------
# 1. Summary statistics
# ---------------------------------------------------------------------------


def summary_stats(series: pd.Series) -> pd.DataFrame:
    """Return descriptive statistics including skewness and kurtosis.

    Parameters
    ----------
    series:
        Numeric time series (e.g. ``log_return``).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with count, mean, std, min, max, skew, kurtosis.
    """
    if series.empty:
        raise ValueError("Input series is empty.")

    s = series.dropna()
    result = {
        "count": len(s),
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "25%": s.quantile(0.25),
        "50%": s.median(),
        "75%": s.quantile(0.75),
        "max": s.max(),
        "skew": float(stats.skew(s)),
        "kurtosis": float(stats.kurtosis(s)),
    }
    return pd.DataFrame(result, index=[series.name or "series"])


def check_stationarity(series: pd.Series, lags: int = 20) -> pd.DataFrame:
    """Run the Augmented Dickey-Fuller test on *series*.

    Parameters
    ----------
    series:
        Numeric time series to test.
    lags:
        Number of lags passed to ``adfuller``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ADF statistic, p-value, critical values, and a
        boolean ``is_stationary`` column (p < 0.05).
    """
    s = series.dropna()
    if len(s) < lags + 2:
        raise ValueError("Series too short for the requested number of lags.")

    adf_stat, p_value, _, _, critical_values, _ = adfuller(s, maxlag=lags, autolag="AIC")
    result = {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "critical_1%": critical_values["1%"],
        "critical_5%": critical_values["5%"],
        "critical_10%": critical_values["10%"],
        "is_stationary": p_value < 0.05,
    }
    return pd.DataFrame(result, index=[series.name or "series"])


def ljung_box_test(series: pd.Series, lags: int = 20) -> pd.DataFrame:
    """Run the Ljung-Box test for autocorrelation on *series*.

    Parameters
    ----------
    series:
        Numeric time series (e.g. squared returns for ARCH effect check).
    lags:
        Number of lags to test.

    Returns
    -------
    pd.DataFrame
        DataFrame with lb_stat and lb_pvalue for each lag.
    """
    s = series.dropna()
    result = acorr_ljungbox(s, lags=lags, return_df=True)
    return result


# ---------------------------------------------------------------------------
# 2. Time series plots
# ---------------------------------------------------------------------------


def plot_time_series(
    series: pd.Series,
    title: str = "Time Series",
    ylabel: str = "Value",
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Plot *series* as a line chart.

    Parameters
    ----------
    series:
        Indexed time series to plot.
    title:
        Plot title.
    ylabel:
        Y-axis label.
    save_path:
        Optional file path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, linewidth=0.8, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_returns(
    df: pd.DataFrame,
    return_col: str = "log_return",
    price_col: str = "close",
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Plot price level and log returns side-by-side.

    Parameters
    ----------
    df:
        DataFrame with at least *return_col* and optionally *price_col*.
    return_col:
        Column name for log returns.
    price_col:
        Column name for price level (plotted in upper panel).
    save_path:
        Optional file path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    has_price = price_col in df.columns
    nrows = 2 if has_price else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    if has_price:
        axes[0].plot(df.index, df[price_col], linewidth=0.8, color="steelblue")
        axes[0].set_title(f"{price_col.capitalize()} Price")
        axes[0].set_ylabel(price_col)
        axes[0].grid(True, alpha=0.3)

    axes[-1].plot(df.index, df[return_col], linewidth=0.6, color="darkorange")
    axes[-1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[-1].set_title(f"{return_col.replace('_', ' ').capitalize()}")
    axes[-1].set_ylabel(return_col)
    axes[-1].set_xlabel("Date")
    axes[-1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_rolling_stats(
    series: pd.Series,
    window: int = 30,
    title: str = "Rolling Mean and Std",
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Plot *series* with its rolling mean and rolling standard deviation.

    Parameters
    ----------
    series:
        Numeric time series.
    window:
        Rolling window size (observations).
    title:
        Plot title.
    save_path:
        Optional file path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(series.index, series.values, linewidth=0.6, color="gray", label="Series")
    axes[0].plot(roll_mean.index, roll_mean.values, linewidth=1.2, color="steelblue", label=f"Mean ({window}d)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(roll_std.index, roll_std.values, linewidth=1.0, color="tomato", label=f"Std ({window}d)")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Distribution plots
# ---------------------------------------------------------------------------


def plot_distribution(
    series: pd.Series,
    title: str = "Return Distribution",
    bins: int = 60,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Plot histogram with KDE overlay and a fitted normal curve.

    Parameters
    ----------
    series:
        Numeric values to plot.
    title:
        Plot title.
    bins:
        Number of histogram bins.
    save_path:
        Optional file path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = series.dropna()
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(s, bins=bins, density=True, alpha=0.5, color="steelblue", label="Histogram")

    # KDE
    kde = stats.gaussian_kde(s)
    x = np.linspace(s.min(), s.max(), 300)
    ax.plot(x, kde(x), linewidth=1.5, color="darkorange", label="KDE")

    # Normal curve
    mu, sigma = s.mean(), s.std()
    ax.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=1.5, linestyle="--", color="green", label="Normal fit")

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    title_prefix: str = "",
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Plot ACF and PACF side-by-side.

    Parameters
    ----------
    series:
        Stationary time series (e.g. log returns or ARIMA residuals).
    lags:
        Number of lags to display.
    title_prefix:
        Optional prefix prepended to subplot titles.
    save_path:
        Optional file path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    acf_title = f"{title_prefix} ACF".strip()
    pacf_title = f"{title_prefix} PACF".strip()

    plot_acf(s, lags=lags, ax=axes[0], title=acf_title, zero=False)
    plot_pacf(s, lags=lags, ax=axes[1], title=pacf_title, zero=False, method="ywm")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_qq(
    series: pd.Series,
    title: str = "Q-Q Plot vs Normal",
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Produce a Q-Q plot of *series* against the normal distribution.

    Parameters
    ----------
    series:
        Numeric values to plot.
    title:
        Plot title.
    save_path:
        Optional file path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = series.dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(s, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig
