"""Unit tests for src/eda.py."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eda import (
    check_stationarity,
    ljung_box_test,
    plot_acf_pacf,
    plot_distribution,
    plot_qq,
    plot_returns,
    plot_rolling_stats,
    plot_time_series,
    summary_stats,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _date_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="B")


def _stationary_series(n: int = 300) -> pd.Series:
    """White noise — clearly stationary."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(n), index=_date_index(n), name="returns")


def _nonstationary_series(n: int = 300) -> pd.Series:
    """Random walk — clearly non-stationary."""
    rng = np.random.default_rng(0)
    return pd.Series(
        np.cumsum(rng.standard_normal(n)),
        index=_date_index(n),
        name="price",
    )


def _ohlcv_df(n: int = 100) -> pd.DataFrame:
    """Minimal DataFrame with close and log_return columns."""
    rng = np.random.default_rng(7)
    close = 3000 + np.cumsum(rng.standard_normal(n))
    log_return = np.diff(np.log(close), prepend=np.nan)
    return pd.DataFrame(
        {"close": close, "log_return": log_return},
        index=_date_index(n),
    ).dropna()


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid resource warnings."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# summary_stats
# ---------------------------------------------------------------------------


class TestSummaryStats:
    def test_returns_dataframe(self):
        result = summary_stats(_stationary_series())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = summary_stats(_stationary_series())
        expected = {"count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew", "kurtosis"}
        assert expected.issubset(set(result.columns))

    def test_single_row(self):
        result = summary_stats(_stationary_series())
        assert len(result) == 1

    def test_index_uses_series_name(self):
        s = _stationary_series()
        result = summary_stats(s)
        assert result.index[0] == s.name

    def test_index_fallback_when_name_is_none(self):
        s = _stationary_series().rename(None)
        result = summary_stats(s)
        assert result.index[0] == "series"

    def test_count_excludes_nan(self):
        s = _stationary_series(10)
        s.iloc[2] = np.nan
        result = summary_stats(s)
        assert result["count"].iloc[0] == 9

    def test_mean_is_correct(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="x")
        result = summary_stats(s)
        assert pytest.approx(result["mean"].iloc[0]) == 3.0

    def test_std_is_correct(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="x")
        result = summary_stats(s)
        assert pytest.approx(result["std"].iloc[0], rel=1e-5) == s.std()

    def test_min_max_correct(self):
        s = pd.Series([3.0, 1.0, 2.0, 5.0, 4.0], name="x")
        result = summary_stats(s)
        assert result["min"].iloc[0] == 1.0
        assert result["max"].iloc[0] == 5.0

    def test_raises_on_empty_series(self):
        with pytest.raises(ValueError, match="empty"):
            summary_stats(pd.Series([], dtype=float, name="empty"))

    def test_all_nan_series_returns_nan_stats(self):
        """All-NaN input: dropna leaves 0 rows — skew/kurtosis become NaN, no raise."""
        s = pd.Series([np.nan, np.nan], name="nan_series")
        result = summary_stats(s)
        assert result["count"].iloc[0] == 0
        assert np.isnan(result["skew"].iloc[0])


# ---------------------------------------------------------------------------
# check_stationarity
# ---------------------------------------------------------------------------


class TestCheckStationarity:
    def test_returns_dataframe(self):
        result = check_stationarity(_stationary_series())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = check_stationarity(_stationary_series())
        expected = {"adf_statistic", "p_value", "critical_1%", "critical_5%", "critical_10%", "is_stationary"}
        assert expected.issubset(set(result.columns))

    def test_single_row(self):
        result = check_stationarity(_stationary_series())
        assert len(result) == 1

    def test_index_uses_series_name(self):
        s = _stationary_series()
        result = check_stationarity(s)
        assert result.index[0] == s.name

    def test_stationary_series_flagged(self):
        result = check_stationarity(_stationary_series(300))
        assert result["is_stationary"].iloc[0] is True or result["p_value"].iloc[0] < 0.05

    def test_nonstationary_series_flagged(self):
        result = check_stationarity(_nonstationary_series(300))
        assert result["is_stationary"].iloc[0] is False or result["p_value"].iloc[0] >= 0.05

    def test_p_value_in_range(self):
        result = check_stationarity(_stationary_series())
        assert 0.0 <= result["p_value"].iloc[0] <= 1.0

    def test_raises_when_series_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            check_stationarity(pd.Series([1.0, 2.0], name="s"), lags=5)

    def test_handles_series_with_leading_nan(self):
        s = _stationary_series(200)
        s.iloc[:5] = np.nan
        result = check_stationarity(s)
        assert "adf_statistic" in result.columns

    def test_custom_lags_accepted(self):
        result = check_stationarity(_stationary_series(200), lags=10)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# ljung_box_test
# ---------------------------------------------------------------------------


class TestLjungBoxTest:
    def test_returns_dataframe(self):
        result = ljung_box_test(_stationary_series())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = ljung_box_test(_stationary_series())
        assert "lb_stat" in result.columns
        assert "lb_pvalue" in result.columns

    def test_row_count_matches_lags(self):
        result = ljung_box_test(_stationary_series(), lags=10)
        assert len(result) == 10

    def test_p_values_in_range(self):
        result = ljung_box_test(_stationary_series())
        assert (result["lb_pvalue"] >= 0).all()
        assert (result["lb_pvalue"] <= 1).all()

    def test_white_noise_high_p_values(self):
        """White noise should mostly have high p-values (fail to reject independence)."""
        rng = np.random.default_rng(99)
        s = pd.Series(rng.standard_normal(500), name="wn")
        result = ljung_box_test(s, lags=5)
        # At least majority of p-values should be > 0.05 for pure white noise
        assert (result["lb_pvalue"] > 0.05).sum() >= 3

    def test_autocorrelated_series_low_p_values(self):
        """AR(1) series should have significant autocorrelation at lag 1."""
        rng = np.random.default_rng(11)
        n = 500
        ar_series = np.zeros(n)
        for i in range(1, n):
            ar_series[i] = 0.8 * ar_series[i - 1] + rng.standard_normal()
        s = pd.Series(ar_series, name="ar1")
        result = ljung_box_test(s, lags=5)
        assert result["lb_pvalue"].iloc[0] < 0.05

    def test_handles_nan_gracefully(self):
        s = _stationary_series(100)
        s.iloc[:3] = np.nan
        result = ljung_box_test(s, lags=5)
        assert len(result) == 5

    def test_default_lags_is_20(self):
        result = ljung_box_test(_stationary_series())
        assert len(result) == 20


# ---------------------------------------------------------------------------
# plot_time_series
# ---------------------------------------------------------------------------


class TestPlotTimeSeries:
    def test_returns_figure(self):
        fig = plot_time_series(_stationary_series())
        assert isinstance(fig, plt.Figure)

    def test_title_is_set(self):
        fig = plot_time_series(_stationary_series(), title="My Title")
        assert fig.axes[0].get_title() == "My Title"

    def test_ylabel_is_set(self):
        fig = plot_time_series(_stationary_series(), ylabel="Price")
        assert fig.axes[0].get_ylabel() == "Price"

    def test_saves_figure(self, tmp_path: Path):
        out = tmp_path / "ts.png"
        plot_time_series(_stationary_series(), save_path=out)
        assert out.exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        out = tmp_path / "a" / "b" / "ts.png"
        plot_time_series(_stationary_series(), save_path=out)
        assert out.exists()

    def test_no_file_when_save_path_none(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plot_time_series(_stationary_series(), save_path=None)
        assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# plot_returns
# ---------------------------------------------------------------------------


class TestPlotReturns:
    def test_returns_figure(self):
        fig = plot_returns(_ohlcv_df())
        assert isinstance(fig, plt.Figure)

    def test_two_panels_when_price_col_present(self):
        fig = plot_returns(_ohlcv_df(), return_col="log_return", price_col="close")
        assert len(fig.axes) == 2

    def test_one_panel_when_price_col_absent(self):
        df = _ohlcv_df().drop(columns=["close"])
        fig = plot_returns(df, return_col="log_return", price_col="close")
        assert len(fig.axes) == 1

    def test_saves_figure(self, tmp_path: Path):
        out = tmp_path / "returns.png"
        plot_returns(_ohlcv_df(), save_path=out)
        assert out.exists()

    def test_custom_return_col(self):
        df = _ohlcv_df().rename(columns={"log_return": "ret"})
        fig = plot_returns(df, return_col="ret", price_col="close")
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_rolling_stats
# ---------------------------------------------------------------------------


class TestPlotRollingStats:
    def test_returns_figure(self):
        fig = plot_rolling_stats(_stationary_series())
        assert isinstance(fig, plt.Figure)

    def test_two_panels(self):
        fig = plot_rolling_stats(_stationary_series())
        assert len(fig.axes) == 2

    def test_title_is_set(self):
        fig = plot_rolling_stats(_stationary_series(), title="Custom Title")
        assert fig.axes[0].get_title() == "Custom Title"

    def test_saves_figure(self, tmp_path: Path):
        out = tmp_path / "rolling.png"
        plot_rolling_stats(_stationary_series(), save_path=out)
        assert out.exists()

    def test_custom_window(self):
        fig = plot_rolling_stats(_stationary_series(200), window=10)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_distribution
# ---------------------------------------------------------------------------


class TestPlotDistribution:
    def test_returns_figure(self):
        fig = plot_distribution(_stationary_series())
        assert isinstance(fig, plt.Figure)

    def test_single_axes(self):
        fig = plot_distribution(_stationary_series())
        assert len(fig.axes) == 1

    def test_title_is_set(self):
        fig = plot_distribution(_stationary_series(), title="Dist Title")
        assert fig.axes[0].get_title() == "Dist Title"

    def test_three_plot_elements(self):
        """Histogram + KDE + normal curve = at least 3 line/patch collections."""
        fig = plot_distribution(_stationary_series())
        ax = fig.axes[0]
        # patches = histogram bars; lines = KDE + normal
        assert len(ax.patches) > 0
        assert len(ax.lines) >= 2

    def test_saves_figure(self, tmp_path: Path):
        out = tmp_path / "dist.png"
        plot_distribution(_stationary_series(), save_path=out)
        assert out.exists()

    def test_handles_nan_in_series(self):
        s = _stationary_series(100)
        s.iloc[:5] = np.nan
        fig = plot_distribution(s)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_acf_pacf
# ---------------------------------------------------------------------------


class TestPlotAcfPacf:
    def test_returns_figure(self):
        fig = plot_acf_pacf(_stationary_series())
        assert isinstance(fig, plt.Figure)

    def test_two_panels(self):
        fig = plot_acf_pacf(_stationary_series())
        assert len(fig.axes) == 2

    def test_title_prefix_applied(self):
        fig = plot_acf_pacf(_stationary_series(), title_prefix="Returns")
        titles = [ax.get_title() for ax in fig.axes]
        assert any("Returns" in t for t in titles)

    def test_saves_figure(self, tmp_path: Path):
        out = tmp_path / "acf_pacf.png"
        plot_acf_pacf(_stationary_series(), save_path=out)
        assert out.exists()

    def test_custom_lags(self):
        fig = plot_acf_pacf(_stationary_series(200), lags=20)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_qq
# ---------------------------------------------------------------------------


class TestPlotQQ:
    def test_returns_figure(self):
        fig = plot_qq(_stationary_series())
        assert isinstance(fig, plt.Figure)

    def test_single_axes(self):
        fig = plot_qq(_stationary_series())
        assert len(fig.axes) == 1

    def test_title_is_set(self):
        fig = plot_qq(_stationary_series(), title="QQ Test")
        assert fig.axes[0].get_title() == "QQ Test"

    def test_saves_figure(self, tmp_path: Path):
        out = tmp_path / "qq.png"
        plot_qq(_stationary_series(), save_path=out)
        assert out.exists()

    def test_handles_nan_in_series(self):
        s = _stationary_series(100)
        s.iloc[:5] = np.nan
        fig = plot_qq(s)
        assert isinstance(fig, plt.Figure)
