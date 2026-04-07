"""Unit tests for src/preprocess.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import (
    REQUIRED_COLUMNS,
    add_features,
    clean_dataframe,
    preprocess,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(
    dates: list[str] | None = None,
    close: list[float] | None = None,
    extra: dict | None = None,
) -> pd.DataFrame:
    """Build a minimal raw OHLCV-like DataFrame for testing."""
    if dates is None:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    if close is None:
        close = [100.0, 102.0, 101.0]
    data = {"date": dates, "close": close}
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


def _make_cleaned(
    dates: list[str] | None = None,
    close: list[float] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame already processed by clean_dataframe."""
    return clean_dataframe(_make_raw(dates=dates, close=close))


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_passes_with_required_columns(self):
        df = _make_raw()
        validate_schema(df)  # should not raise

    def test_raises_when_date_missing(self):
        df = pd.DataFrame({"close": [100.0]})
        with pytest.raises(ValueError, match="date"):
            validate_schema(df)

    def test_raises_when_close_missing(self):
        df = pd.DataFrame({"date": ["2024-01-01"]})
        with pytest.raises(ValueError, match="close"):
            validate_schema(df)

    def test_raises_with_all_missing(self):
        df = pd.DataFrame({"open": [100.0]})
        with pytest.raises(ValueError):
            validate_schema(df)

    def test_extra_columns_are_allowed(self):
        df = _make_raw(extra={"volume": [1000, 2000, 3000]})
        validate_schema(df)  # should not raise


# ---------------------------------------------------------------------------
# clean_dataframe
# ---------------------------------------------------------------------------


class TestCleanDataframe:
    def test_returns_datetime_index(self):
        result = clean_dataframe(_make_raw())
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "date"

    def test_date_column_removed_from_columns(self):
        result = clean_dataframe(_make_raw())
        assert "date" not in result.columns

    def test_close_is_float(self):
        result = clean_dataframe(_make_raw())
        assert result["close"].dtype == float

    def test_numeric_columns_cast(self):
        df = _make_raw(extra={"open": ["10.5", "20.1", "30.9"], "volume": ["100.0", "200.0", "300.0"]})
        result = clean_dataframe(df)
        assert result["open"].dtype == float
        assert result["volume"].dtype == float

    def test_invalid_dates_dropped(self):
        df = _make_raw(dates=["2024-01-01", "not-a-date", "2024-01-03"])
        result = clean_dataframe(df)
        assert len(result) == 2

    def test_fully_duplicate_rows_dropped(self):
        df = pd.concat([_make_raw(), _make_raw()], ignore_index=True)
        result = clean_dataframe(df)
        assert len(result) == 3

    def test_duplicate_dates_keeps_first(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01"],
            "close": [100.0, 999.0],
        })
        result = clean_dataframe(df)
        assert len(result) == 1
        assert result["close"].iloc[0] == 100.0

    def test_sorted_ascending(self):
        df = _make_raw(
            dates=["2024-01-03", "2024-01-01", "2024-01-02"],
            close=[101.0, 100.0, 102.0],
        )
        result = clean_dataframe(df)
        assert result.index.is_monotonic_increasing

    def test_nan_filled_by_ffill_bfill(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "close": [100.0, np.nan, 102.0],
        })
        result = clean_dataframe(df)
        assert result["close"].isna().sum() == 0

    def test_returns_copy_not_inplace(self):
        raw = _make_raw()
        original_cols = list(raw.columns)
        clean_dataframe(raw)
        assert list(raw.columns) == original_cols  # original untouched


# ---------------------------------------------------------------------------
# add_features
# ---------------------------------------------------------------------------


class TestAddFeatures:
    def test_log_close_added(self):
        df = _make_cleaned()
        result = add_features(df)
        assert "log_close" in result.columns

    def test_log_return_added(self):
        df = _make_cleaned()
        result = add_features(df)
        assert "log_return" in result.columns

    def test_log_close_values_correct(self):
        df = _make_cleaned()
        result = add_features(df)
        expected = np.log(result["close"]).rename("log_close")
        pd.testing.assert_series_equal(result["log_close"], expected)

    def test_log_return_values_correct(self):
        df = _make_cleaned(dates=["2024-01-01", "2024-01-02", "2024-01-03"], close=[100.0, 110.0, 99.0])
        result = add_features(df)
        expected_r1 = np.log(110.0 / 100.0)
        expected_r2 = np.log(99.0 / 110.0)
        assert pytest.approx(result["log_return"].iloc[0]) == expected_r1
        assert pytest.approx(result["log_return"].iloc[1]) == expected_r2

    def test_first_nan_row_dropped(self):
        df = _make_cleaned()
        result = add_features(df)
        # original cleaned df has 3 rows; first log_return is NaN → dropped
        assert len(result) == len(df) - 1
        assert result["log_return"].isna().sum() == 0

    def test_raises_on_non_positive_close(self):
        df = _make_cleaned(close=[100.0, 0.0, 101.0])
        with pytest.raises(ValueError, match="Non-positive"):
            add_features(df)

    def test_raises_on_negative_close(self):
        df = _make_cleaned(close=[100.0, -5.0, 101.0])
        with pytest.raises(ValueError, match="Non-positive"):
            add_features(df)

    def test_returns_copy_not_inplace(self):
        df = _make_cleaned()
        cols_before = list(df.columns)
        add_features(df)
        assert list(df.columns) == cols_before


# ---------------------------------------------------------------------------
# preprocess (full pipeline)
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_returns_dataframe(self):
        result = preprocess(_make_raw())
        assert isinstance(result, pd.DataFrame)

    def test_has_log_return_and_log_close(self):
        result = preprocess(_make_raw())
        assert "log_return" in result.columns
        assert "log_close" in result.columns

    def test_index_is_datetime(self):
        result = preprocess(_make_raw())
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_raises_on_missing_required_columns(self):
        df = pd.DataFrame({"open": [100.0, 101.0]})
        with pytest.raises(ValueError):
            preprocess(df)

    def test_saves_csv_when_save_path_given(self, tmp_path: Path):
        out = tmp_path / "sub" / "out.csv"
        result = preprocess(_make_raw(), save_path=out)
        assert out.exists()
        loaded = pd.read_csv(out, index_col="date", parse_dates=True)
        assert list(loaded.columns) == list(result.columns)

    def test_parent_directories_created(self, tmp_path: Path):
        out = tmp_path / "a" / "b" / "c" / "out.csv"
        preprocess(_make_raw(), save_path=out)
        assert out.exists()

    def test_no_file_written_when_save_path_none(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        preprocess(_make_raw(), save_path=None)
        assert list(tmp_path.iterdir()) == []
