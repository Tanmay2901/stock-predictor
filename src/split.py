"""
split.py

Single responsibility: perform chronological train/test splitting
and feature scaling.
"""


def time_split(
    df_features,
    feature_cols,
    test_size: float = 0.15,
):
    """Perform a chronological train/test split.

    Time-series data must not be randomly shuffled. The complete
    implementation will be added in Milestone 5.
    """
    raise NotImplementedError("Milestone 5")


def scale_features(X_train, X_test):
    """Fit a scaler on training data and transform both datasets.

    The complete implementation will be added in Milestone 5.
    """
    raise NotImplementedError("Milestone 5")