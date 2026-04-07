"""
Preprocessing module for Shanghai index OHLCV time series data.

Pipeline:
  1. validate_schema  – assert required columns are present
  2. clean_dataframe  – parse dates, cast numerics, deduplicate, fill gaps
  3. add_features     – compute log_return and log_close for modelling
  4. preprocess       – full pipeline; optionally persists to disk
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: list[str] = ["date", "close"]
NUMERIC_COLUMNS: list[str] = ["open", "high", "low", "close", "volume", "amount"]


# ---------------------------------------------------------------------------
# Step 1: Schema validation
# ---------------------------------------------------------------------------


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing.

    Parameters
    ----------
    df:
        Raw input DataFrame.

    Raises
    ------
    ValueError
        When one or more required columns are absent.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Step 2: Cleaning
# ---------------------------------------------------------------------------


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of *df*.

    Operations performed:
    - Parse ``date`` column to datetime and set as index.
    - Cast numeric columns to float (coerce errors → NaN).
    - Drop fully duplicated rows.
    - Remove rows with duplicate dates (keep first occurrence).
    - Forward-fill then back-fill residual NaN values in numeric columns.
    - Sort index ascending.

    Parameters
    ----------
    df:
        Raw input DataFrame with at least the columns in ``REQUIRED_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame indexed by ``date`` (DatetimeIndex).
    """
    df = df.copy()

    # Parse date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = df["date"].isna().sum()
    if invalid_dates > 0:
        df = df.dropna(subset=["date"])

    # Cast numeric columns that exist in the DataFrame
    existing_numeric = [c for c in NUMERIC_COLUMNS if c in df.columns]
    for col in existing_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop fully duplicate rows
    df = df.drop_duplicates()

    # Set date as index
    df = df.set_index("date")
    df.index.name = "date"

    # Keep first entry per date if duplicates remain
    df = df[~df.index.duplicated(keep="first")]

    # Sort chronologically
    df = df.sort_index()

    # Fill residual NaN values in numeric columns
    df[existing_numeric] = (
        df[existing_numeric].ffill().bfill()
    )

    return df


# ---------------------------------------------------------------------------
# Step 3: Feature engineering
# ---------------------------------------------------------------------------


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by ARIMA and GARCH models.

    New columns
    -----------
    log_close : float
        Natural log of the closing price.
    log_return : float
        Daily log return: ln(close_t / close_{t-1}).
        The first row becomes NaN and is dropped.

    Parameters
    ----------
    df:
        Cleaned DataFrame with a ``close`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional feature columns; first row (NaN return) removed.
    """
    df = df.copy()

    if (df["close"] <= 0).any():
        raise ValueError("Non-positive close prices detected; log transform is invalid.")

    df["log_close"] = np.log(df["close"])
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Drop the first row which has NaN log_return
    df = df.dropna(subset=["log_return"])

    return df


# ---------------------------------------------------------------------------
# Step 4: Full pipeline
# ---------------------------------------------------------------------------


def preprocess(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline on a raw OHLCV DataFrame.

    Steps: validate → clean → add_features → (optionally) save.

    Parameters
    ----------
    df:
        Raw input DataFrame loaded from CSV (e.g. via ``data_loader``).
    save_path:
        If provided, the processed DataFrame is written to this CSV path.
        Parent directories are created automatically.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with a DatetimeIndex and feature columns
        ``log_close`` and ``log_return``.

    Examples
    --------
    >>> import pandas as pd
    >>> from preprocess import preprocess
    >>> raw = pd.read_csv("data/raw/train_data_sh.000001_2000-01-01_2024-06-30.csv")
    >>> processed = preprocess(raw, save_path="data/processed/train_processed.csv")
    >>> processed[["close", "log_return"]].head()
    """
    validate_schema(df)
    df = clean_dataframe(df)
    df = add_features(df)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)

    return df
