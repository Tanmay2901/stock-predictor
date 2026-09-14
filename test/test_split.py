"""Tests for chronological splitting and leakage-safe scaling."""

import numpy as np
import pandas as pd
import pytest

from src.split import (
    scale_features,
    scale_target,
    time_split,
)


def make_feature_data(rows: int = 100) -> pd.DataFrame:
    """Create deterministic model-ready feature data."""
    dates = pd.bdate_range(
        start="2024-01-02",
        periods=rows,
    )

    sequence = np.arange(rows, dtype=float)

    return pd.DataFrame(
        {
            "feature_a": sequence,
            "feature_b": sequence * 2,
            "close": 100 + sequence,
            "target_return_1d": 0.001 + (sequence * 0.00001),
            "target_close": 101 + sequence,
        },
        index=dates,
    ).rename_axis("Date")


def test_time_split_is_chronological_and_complete():
    """An 85/15 split should preserve every row without overlap."""
    features = make_feature_data(rows=100)
    feature_columns = ["feature_a", "feature_b"]

    result = time_split(
        features,
        feature_columns,
        test_size=0.15,
    )

    assert result["X_train"].shape == (85, 2)
    assert result["X_test"].shape == (15, 2)

    assert (
        len(result["X_train"])
        + len(result["X_test"])
        == len(features)
    )

    assert result["train_index"].is_monotonic_increasing
    assert result["test_index"].is_monotonic_increasing

    assert (
        result["train_index"].max()
        < result["test_index"].min()
    )

    assert (
        result["train_end_date"]
        == result["train_index"].max()
    )

    assert (
        result["test_start_date"]
        == result["test_index"].min()
    )

    assert set(result["train_index"]).isdisjoint(
        set(result["test_index"])
    )


def test_feature_scaler_fits_training_data_only():
    """Scaler statistics must come entirely from training observations."""
    X_train = np.array(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
        ]
    )

    X_test = np.array(
        [
            [100.0, 110.0],
            [200.0, 210.0],
        ]
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test,
    )

    expected_train_mean = X_train.mean(axis=0)
    combined_mean = np.vstack([X_train, X_test]).mean(axis=0)

    assert np.allclose(scaler.mean_, expected_train_mean)
    assert not np.allclose(scaler.mean_, combined_mean)
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0)

    assert X_test_scaled.shape == X_test.shape


def test_target_scaler_can_inverse_transform():
    """Scaled return predictions must be recoverable in real units."""
    y_train = np.array([0.01, -0.02, 0.005, 0.015])
    y_test = np.array([0.03, -0.01])

    y_train_scaled, y_test_scaled, scaler = scale_target(
        y_train,
        y_test,
    )

    recovered_train = scaler.inverse_transform(
        y_train_scaled.reshape(-1, 1)
    ).ravel()

    recovered_test = scaler.inverse_transform(
        y_test_scaled.reshape(-1, 1)
    ).ravel()

    assert np.allclose(recovered_train, y_train)
    assert np.allclose(recovered_test, y_test)


@pytest.mark.parametrize(
    "invalid_test_size",
    [0, 1, -0.10, 1.10],
)
def test_time_split_rejects_invalid_test_size(
    invalid_test_size,
):
    """test_size must leave non-empty training and testing sets."""
    features = make_feature_data()

    with pytest.raises(
        ValueError,
        match="test_size must be",
    ):
        time_split(
            features,
            ["feature_a", "feature_b"],
            test_size=invalid_test_size,
        )


def test_time_split_rejects_unsorted_dates():
    """Unsorted time-series observations must not be accepted."""
    features = make_feature_data()
    reversed_features = features.iloc[::-1]

    with pytest.raises(
        ValueError,
        match="sorted chronologically",
    ):
        time_split(
            reversed_features,
            ["feature_a", "feature_b"],
        )


def test_scale_features_rejects_different_column_counts():
    """Train and test matrices must contain identical features."""
    X_train = np.ones((10, 3))
    X_test = np.ones((5, 2))

    with pytest.raises(
        ValueError,
        match="same feature count",
    ):
        scale_features(X_train, X_test)