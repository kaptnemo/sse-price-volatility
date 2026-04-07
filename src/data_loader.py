"""
Data loading module for Shanghai Composite Index OHLCV data.

Provides functions to load raw train and test CSV files produced by
sh_index_ingest.py, returning them as pandas DataFrames ready for
preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Default data directory relative to project root
_DATA_RAW = Path(__file__).parents[1] / "data" / "raw"


def load_train(
    base_dir: str | Path | None = None,
    code: str = "sh.000001",
    start_date: str = "2016-01-01",
    end_date: str = "2020-12-31",
) -> pd.DataFrame:
    """Load the raw training CSV for the given code and date range.

    Parameters
    ----------
    base_dir:
        Directory containing the raw CSV files. Defaults to ``data/raw/``
        relative to the project root.
    code:
        Stock/index code used in the filename (e.g. ``"sh.000001"``).
    start_date:
        Start date string used in the filename (``"YYYY-MM-DD"``).
    end_date:
        End date string used in the filename (``"YYYY-MM-DD"``).

    Returns
    -------
    pd.DataFrame
        Raw OHLCV DataFrame with columns: date, open, high, low, close,
        volume, amount, code, frequency.

    Raises
    ------
    FileNotFoundError
        If the expected CSV file does not exist.

    Examples
    --------
    >>> from src.data_loader import load_train
    >>> df = load_train()
    >>> df.shape
    (6130, 9)
    """
    base = Path(base_dir) if base_dir is not None else _DATA_RAW
    filename = f"train_data_{code}_{start_date}_{end_date}.csv"
    path = base / filename
    if not path.exists():
        raise FileNotFoundError(f"Train data file not found: {path}")
    return pd.read_csv(path)


def load_test(
    base_dir: str | Path | None = None,
    code: str = "sh.000001",
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Load the raw test CSV for the given code and date range.

    Parameters
    ----------
    base_dir:
        Directory containing the raw CSV files. Defaults to ``data/raw/``
        relative to the project root.
    code:
        Stock/index code used in the filename.
    start_date:
        Start date string used in the filename (``"YYYY-MM-DD"``).
    end_date:
        End date string used in the filename (``"YYYY-MM-DD"``).

    Returns
    -------
    pd.DataFrame
        Raw OHLCV DataFrame.

    Raises
    ------
    FileNotFoundError
        If the expected CSV file does not exist.

    Examples
    --------
    >>> from src.data_loader import load_test
    >>> df = load_test()
    >>> df.shape
    (124, 9)
    """
    base = Path(base_dir) if base_dir is not None else _DATA_RAW
    filename = f"test_data_{code}_{start_date}_{end_date}.csv"
    path = base / filename
    if not path.exists():
        raise FileNotFoundError(f"Test data file not found: {path}")
    return pd.read_csv(path)
