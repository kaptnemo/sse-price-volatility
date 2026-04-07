from pathlib import Path
from baostock_helper import BaostockHelper

def daily_ingest_to_csv(code, start_date, end_date, file_path=None):
    """Ingest Shanghai Stock Exchange index data."""
    with BaostockHelper() as helper:

        dayly_result = helper.daily(code, start_date, end_date)
        file_path = file_path or f"{code}_daily_{start_date}_{end_date}.csv"
        dayly_result.save_to_csv(file_path)

def main():
    # Example usage
    base_path = Path(__file__).parents[1] / "data" / "raw"
    code = "sh.000001"  # Shanghai Stock Exchange Composite Index
    start_date = "2016-01-01"
    end_date = "2020-12-31"
    train_file_path = base_path / f"train_data_{code}_{start_date}_{end_date}.csv"
    daily_ingest_to_csv(code, start_date, end_date, file_path=train_file_path)

    test_start_date = "2021-01-01"
    test_end_date = "2025-12-31"
    test_file_path = base_path / f"test_data_{code}_{test_start_date}_{test_end_date}.csv"
    daily_ingest_to_csv(code, test_start_date, test_end_date, file_path=test_file_path)


if __name__ == "__main__":
    main()