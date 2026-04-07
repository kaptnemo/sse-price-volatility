"""Unit tests for src/arima_model.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arima_model import rolling_forecast_arima


def _make_series(n_obs: int = 90, seed: int = 7) -> pd.Series:
    """Build a deterministic integrated series suitable for ARIMA tests."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(loc=0.0, scale=0.5, size=n_obs)
    values = 10.0 + np.cumsum(innovations)
    index = pd.date_range("2024-01-01", periods=n_obs, freq="B")
    index.name = "date"
    return pd.Series(values, index=index, name="log_close")


class TestRollingForecastArima:
    def test_returns_expected_shape_columns_and_index(self):
        series = _make_series()
        train = series.iloc[:75]
        test = series.iloc[75:]

        forecast_df, result = rolling_forecast_arima(
            train_data=train,
            test_data=test,
            order=(1, 1, 0),
        )

        assert list(forecast_df.columns) == ["forecast", "lower_ci", "upper_ci", "actual"]
        assert len(forecast_df) == len(test)
        pd.testing.assert_index_equal(forecast_df.index, test.index)
        assert result.model.order == (1, 1, 0)

    def test_actual_column_matches_test_data(self):
        series = _make_series()
        train = series.iloc[:80]
        test = series.iloc[80:]

        forecast_df, _ = rolling_forecast_arima(
            train_data=train,
            test_data=test,
            order=(1, 1, 0),
            window=40,
            refit_every=2,
        )

        pd.testing.assert_series_equal(
            forecast_df["actual"],
            test.rename("actual"),
            check_freq=False,
        )

    def test_raises_when_test_is_empty(self):
        series = _make_series()

        with pytest.raises(ValueError, match="test_data must not be empty"):
            rolling_forecast_arima(
                train_data=series.iloc[:80],
                test_data=series.iloc[0:0],
                order=(1, 1, 0),
            )

    def test_raises_when_refit_every_is_invalid(self):
        series = _make_series()

        with pytest.raises(ValueError, match="refit_every must be >= 1"):
            rolling_forecast_arima(
                train_data=series.iloc[:80],
                test_data=series.iloc[80:],
                order=(1, 1, 0),
                refit_every=0,
            )
