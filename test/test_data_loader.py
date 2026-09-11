"""Automated tests for the stock-price data loader."""

import pandas as pd
import pytest

from src.data_loader import REQUIRED_COLUMNS, load_raw_prices


def test_load_raw_prices_cleans_bad_rows(tmp_path):
    """The loader should clean duplicates, nulls, and invalid prices."""
    test_file = tmp_path / "sample_prices.csv"

    raw_data = pd.DataFrame(
        [
            {
                "Date": "2015-01-02",
                "Open": 100,
                "High": 105,
                "Low": 99,
                "Close": 103,
                "Adj Close": 90,
                "Volume": 1000,
            },
            {
                "Date": "2015-01-02",
                "Open": 999,
                "High": 999,
                "Low": 999,
                "Close": 999,
                "Adj Close": 999,
                "Volume": 999,
            },
            {
                "Date": "2015-01-05",
                "Open": 103,
                "High": 106,
                "Low": 101,
                "Close": 104,
                "Adj Close": 91,
                "Volume": 1100,
            },
            {
                "Date": "2015-01-08",
                "Open": None,
                "High": None,
                "Low": None,
                "Close": None,
                "Adj Close": None,
                "Volume": None,
            },
            {
                "Date": "2015-01-09",
                "Open": 0,
                "High": 107,
                "Low": 102,
                "Close": 106,
                "Adj Close": 93,
                "Volume": 1300,
            },
            {
                "Date": "2015-01-06",
                "Open": 104,
                "High": 107,
                "Low": 102,
                "Close": 105,
                "Adj Close": 92,
                "Volume": 1200,
            },
            {
                "Date": "2015-01-07",
                "Open": 105,
                "High": 108,
                "Low": 103,
                "Close": 106,
                "Adj Close": 93,
                "Volume": 1250,
            },
        ]
    )

    raw_data.to_csv(test_file, index=False)

    cleaned = load_raw_prices(test_file)

    assert cleaned.shape == (4, 6)
    assert cleaned.columns.tolist() == REQUIRED_COLUMNS
    assert cleaned.index.name == "Date"
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.index.is_unique
    assert cleaned.isna().sum().sum() == 0
    assert (cleaned[["Open", "High", "Low", "Close", "Adj Close"]] > 0).all().all()
    assert (cleaned["Volume"] >= 0).all()

    # The first version of the duplicated date must be preserved.
    assert cleaned.loc[pd.Timestamp("2015-01-02"), "Open"] == 100


def test_load_raw_prices_rejects_missing_columns(tmp_path):
    """The loader should clearly report an invalid CSV schema."""
    test_file = tmp_path / "missing_columns.csv"

    pd.DataFrame(
        {
            "Date": ["2015-01-02"],
            "Close": [100],
        }
    ).to_csv(test_file, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_raw_prices(test_file)


def test_load_raw_prices_rejects_missing_file(tmp_path):
    """The loader should clearly report a missing CSV file."""
    missing_file = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError, match="was not found"):
        load_raw_prices(missing_file)