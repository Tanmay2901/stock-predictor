"""
Chronological train/test splitting and scaling.

Time-series data must never use an ordinary shuffled train/test split.
Training observations must occur strictly before testing observations.

Scalers are fitted on training data only. Fitting a scaler on the entire
dataset would allow test-set statistics to influence training.
"""

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


REQUIRED_TARGET_COLUMNS = [
    "close",
    "target_return_1d",
    "target_close",
]


def _validate_split_inputs(
    df_features: pd.DataFrame,
    feature_cols: Sequence[str],
    test_size: float,
) -> list[str]:
    """Validate splitting inputs and return feature names as a list."""
    if not isinstance(df_features, pd.DataFrame):
        raise TypeError("df_features must be a pandas DataFrame.")

    if len(df_features) < 2:
        raise ValueError(
            "At least two feature rows are required for splitting."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    feature_columns = list(feature_cols)

    if not feature_columns:
        raise ValueError("feature_cols cannot be empty.")

    duplicate_features = {
        column
        for column in feature_columns
        if feature_columns.count(column) > 1
    }

    if duplicate_features:
        raise ValueError(
            "feature_cols contains duplicates: "
            f"{sorted(duplicate_features)}"
        )

    required_columns = [
        *feature_columns,
        *REQUIRED_TARGET_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df_features.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot split data because columns are missing: "
            f"{missing_columns}"
        )

    if not df_features.index.is_monotonic_increasing:
        raise ValueError(
            "Feature data must be sorted chronologically."
        )

    if not df_features.index.is_unique:
        raise ValueError(
            "Feature data contains duplicate index values."
        )

    return feature_columns


def time_split(
    df_features: pd.DataFrame,
    feature_cols: Sequence[str],
    test_size: float = 0.15,
) -> dict[str, Any]:
    """Split feature data once at a chronological boundary.

    Args:
        df_features: Feature and target data sorted oldest to newest.
        feature_cols: Columns supplied to the models.
        test_size: Fraction reserved for final testing.

    Returns:
        A dictionary containing dates, indices, features, targets,
        current closes, and next-day closes for both partitions.
    """
    feature_columns = _validate_split_inputs(
        df_features,
        feature_cols,
        test_size,
    )

    number_of_rows = len(df_features)
    split_index = int(number_of_rows * (1 - test_size))

    if split_index <= 0 or split_index >= number_of_rows:
        raise ValueError(
            "test_size produced an empty training or testing partition."
        )

    train = df_features.iloc[:split_index].copy()
    test = df_features.iloc[split_index:].copy()

    train_end_date = train.index.max()
    test_start_date = test.index.min()

    if train_end_date >= test_start_date:
        raise ValueError(
            "Training dates must occur strictly before testing dates."
        )

    X_train = train[feature_columns].to_numpy(
        dtype=float,
        copy=True,
    )

    X_test = test[feature_columns].to_numpy(
        dtype=float,
        copy=True,
    )

    y_train_ret = train["target_return_1d"].to_numpy(
        dtype=float,
        copy=True,
    )

    y_test_ret = test["target_return_1d"].to_numpy(
        dtype=float,
        copy=True,
    )

    close_train = train["close"].to_numpy(
        dtype=float,
        copy=True,
    )

    close_test = test["close"].to_numpy(
        dtype=float,
        copy=True,
    )

    target_close_train = train["target_close"].to_numpy(
        dtype=float,
        copy=True,
    )

    target_close_test = test["target_close"].to_numpy(
        dtype=float,
        copy=True,
    )

    arrays_to_check = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_ret": y_train_ret,
        "y_test_ret": y_test_ret,
        "close_train": close_train,
        "close_test": close_test,
        "target_close_train": target_close_train,
        "target_close_test": target_close_test,
    }

    non_finite_arrays = [
        name
        for name, values in arrays_to_check.items()
        if not np.isfinite(values).all()
    ]

    if non_finite_arrays:
        raise ValueError(
            "Split data contains NaN or infinite values in: "
            f"{non_finite_arrays}"
        )

    return {
        "train_end_date": train_end_date,
        "test_start_date": test_start_date,
        "train_index": train.index.copy(),
        "test_index": test.index.copy(),
        "feature_columns": feature_columns,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_ret": y_train_ret,
        "y_test_ret": y_test_ret,
        "close_train": close_train,
        "close_test": close_test,
        "target_close_train": target_close_train,
        "target_close_test": target_close_test,
    }


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a feature scaler on training data and transform both sets."""
    X_train_array = np.asarray(X_train, dtype=float)
    X_test_array = np.asarray(X_test, dtype=float)

    if X_train_array.ndim != 2 or X_test_array.ndim != 2:
        raise ValueError(
            "X_train and X_test must both be two-dimensional."
        )

    if X_train_array.shape[0] == 0 or X_test_array.shape[0] == 0:
        raise ValueError(
            "Training and testing feature arrays cannot be empty.")

    if X_train_array.shape[1] != X_test_array.shape[1]:
        raise ValueError(
            "Training and testing data must have the same feature count."
        )

    if not np.isfinite(X_train_array).all():
        raise ValueError("X_train contains NaN or infinite values.")

    if not np.isfinite(X_test_array).all():
        raise ValueError("X_test contains NaN or infinite values.")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_array)
    X_test_scaled = scaler.transform(X_test_array)

    return X_train_scaled, X_test_scaled, scaler


def scale_target(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale a one-dimensional target using training statistics only.

    Target scaling is mainly intended for the LSTM. Classical models may
    continue using unscaled returns. Any scaled prediction must be inverse
    transformed before calculating real-return metrics.
    """
    y_train_array = np.asarray(y_train, dtype=float)
    y_test_array = np.asarray(y_test, dtype=float)

    if y_train_array.ndim != 1 or y_test_array.ndim != 1:
        raise ValueError(
            "y_train and y_test must both be one-dimensional."
        )

    if y_train_array.size == 0 or y_test_array.size == 0:
        raise ValueError("Training and testing target arrays cannot be empty.")

    if not np.isfinite(y_train_array).all():
        raise ValueError("y_train contains NaN or infinite values.")

    if not np.isfinite(y_test_array).all():
        raise ValueError("y_test contains NaN or infinite values.")

    scaler = StandardScaler()

    y_train_2d = y_train_array.reshape(-1, 1)
    y_test_2d = y_test_array.reshape(-1, 1)

    y_train_scaled = scaler.fit_transform(y_train_2d).ravel()
    y_test_scaled = scaler.transform(y_test_2d).ravel()

    return y_train_scaled, y_test_scaled, scaler


def main() -> None:
    """Run and display the chronological split on the real dataset."""
    try:
        from src.data_loader import load_raw_prices
        from src.features import build_features, get_feature_columns
    except ModuleNotFoundError:
        from data_loader import load_raw_prices
        from features import build_features, get_feature_columns

    raw = load_raw_prices()
    features = build_features(raw)
    feature_columns = get_feature_columns(features)

    split_data = time_split(
        features,
        feature_columns,
        test_size=0.15,
    )

    X_train_scaled, X_test_scaled, feature_scaler = scale_features(
        split_data["X_train"],
        split_data["X_test"],
    )

    y_train_scaled, y_test_scaled, target_scaler = scale_target(
        split_data["y_train_ret"],
        split_data["y_test_ret"],
    )

    print("Train end date:", split_data["train_end_date"])
    print("Test start date:", split_data["test_start_date"])
    print("Train shape:", split_data["X_train"].shape)
    print("Test shape:", split_data["X_test"].shape)
    print("Number of features:", len(feature_columns))
    print(
        "Chronological separation:",
        split_data["train_end_date"]
        < split_data["test_start_date"],
    )

    print(
        "Scaled training-feature mean:",
        round(float(X_train_scaled.mean()), 8),
    )

    print(
        "Scaled training-feature standard deviation:",
        round(float(X_train_scaled.std()), 8),
    )

    print(
        "Scaled target shape:",
        y_train_scaled.shape,
        y_test_scaled.shape,
    )

    print(
        "Feature scaler learned columns:",
        feature_scaler.n_features_in_,
    )

    recovered_targets = target_scaler.inverse_transform(
        y_train_scaled.reshape(-1, 1)
    ).ravel()

    print(
        "Target inverse transformation works:",
        bool(
            np.allclose(
                recovered_targets,
                split_data["y_train_ret"],
            )
        ),
    )


if __name__ == "__main__":
    main()
