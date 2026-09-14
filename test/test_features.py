"""Tests for stock-market feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    _rsi,
    build_features,
    get_feature_columns,
)


def make_sample_ohlcv(rows: int = 140) -> pd.DataFrame:
    """Create deterministic OHLCV data for repeatable tests."""
    dates = pd.bdate_range(
        start="2024-01-02",
        periods=rows,
    )

    sequence = np.arange(rows, dtype=float)

    close = (
        100.0
        + (sequence * 0.12)
        + (2.5 * np.sin(sequence / 6.0))
    )

    open_price = close * (
        1 + (0.002 * np.sin(sequence / 4.0))
    )

    high = np.maximum(open_price, close) * 1.01
    low = np.minimum(open_price, close) * 0.99
    adjusted_close = close * 0.98
    volume = 1_000_000 + (sequence * 1_000)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adjusted_close,
            "Volume": volume,
        },
        index=dates,
    ).rename_axis("Date")


def test_build_features_produces_expected_schema():
    """Feature engineering should produce 32 inputs and 3 helper columns."""
    raw = make_sample_ohlcv()

    features = build_features(raw)
    feature_columns = get_feature_columns(features)

    assert features.shape[1] == 35
    assert len(feature_columns) == 32

    assert "close" not in feature_columns
    assert "target_return_1d" not in feature_columns
    assert "target_close" not in feature_columns

    assert features.index.is_monotonic_increasing
    assert features.index.is_unique
    assert features.isna().sum().sum() == 0
    assert np.isfinite(features.to_numpy(dtype=float)).all()


def test_targets_represent_the_next_trading_day():
    """Each row's target must describe the following trading day."""
    raw = make_sample_ohlcv()
    features = build_features(raw)

    row_date = features.index[5]
    raw_position = raw.index.get_loc(row_date)
    next_date = raw.index[raw_position + 1]

    current_close = raw.loc[row_date, "Close"]
    next_close = raw.loc[next_date, "Close"]
    expected_return = (next_close / current_close) - 1

    assert features.loc[row_date, "target_close"] == pytest.approx(
        next_close
    )

    assert features.loc[
        row_date,
        "target_return_1d",
    ] == pytest.approx(expected_return)


def test_features_do_not_use_future_market_values():
    """Changing future data must not change earlier feature values."""
    original_raw = make_sample_ohlcv()
    changed_raw = original_raw.copy()

    cutoff_position = 90
    cutoff_date = original_raw.index[cutoff_position]

    future_dates = changed_raw.index[cutoff_position + 1:]

    changed_raw.loc[
        future_dates,
        ["Open", "High", "Low", "Close", "Adj Close"],
    ] *= 5

    changed_raw.loc[future_dates, "Volume"] *= 10

    original_features = build_features(original_raw)
    changed_features = build_features(changed_raw)

    feature_columns = get_feature_columns(original_features)

    pd.testing.assert_frame_equal(
        original_features.loc[
            :cutoff_date,
            feature_columns,
        ],
        changed_features.loc[
            :cutoff_date,
            feature_columns,
        ],
    )


def test_rsi_handles_zero_loss_correctly():
    """An uninterrupted rise should produce RSI 100, not RSI 50."""
    increasing_close = pd.Series(
        np.arange(1, 50, dtype=float)
    )

    rsi = _rsi(increasing_close, window=14)
    calculated_values = rsi.dropna()

    assert not calculated_values.empty
    assert (calculated_values == 100.0).all()


def test_rsi_handles_flat_period_as_neutral():
    """A completely flat period should produce a neutral RSI of 50."""
    flat_close = pd.Series(
        np.full(50, 100.0)
    )

    rsi = _rsi(flat_close, window=14)
    calculated_values = rsi.dropna()

    assert not calculated_values.empty
    assert (calculated_values == 50.0).all()


def test_build_features_rejects_missing_input_columns():
    """Missing required market columns should produce a clear error."""
    raw = make_sample_ohlcv().drop(columns=["Close"])

    with pytest.raises(ValueError, match="columns are missing"):
        build_features(raw)
