"""
Feature engineering for the stock-price prediction project.

Important design decision
-------------------------
The models predict NEXT-DAY RETURN rather than the next-day raw price.

Raw prices are highly autocorrelated: today's price is normally very
close to tomorrow's price. Consequently, a naive model that predicts
"tomorrow equals today" can achieve an extremely high price-level R²
without learning how to predict market movements.

Returns are closer to stationary and produce a more meaningful learning
problem. A predicted price can still be reconstructed later:

    predicted_close_t_plus_1 = close_t * (1 + predicted_return_t_plus_1)

Prediction timing
-----------------
A feature row for day t is assumed to be constructed after day t's market
close. Therefore, the Open, High, Low, Close, and Volume for day t are
known. Every feature uses information from day t or earlier. Only target
columns refer to day t+1.
"""

import numpy as np
import pandas as pd


REQUIRED_INPUT_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

TARGET_COLUMNS = [
    "target_return_1d",
    "target_close",
]

NON_FEATURE_COLUMNS = {
    "close",
    *TARGET_COLUMNS,
}


def _validate_input(df: pd.DataFrame) -> None:
    """Validate the cleaned OHLCV DataFrame before building features."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot build features from an empty DataFrame.")

    missing_columns = [
        column
        for column in REQUIRED_INPUT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot build features because columns are missing: "
            f"{missing_columns}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "The DataFrame index must be a pandas DatetimeIndex. "
            "Use load_raw_prices() before build_features()."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "The DataFrame index must be sorted chronologically."
        )

    if not df.index.is_unique:
        raise ValueError("The DataFrame contains duplicate dates.")


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's exponential smoothing.

    RSI interpretation:

    - 70 or higher is traditionally considered overbought.
    - 30 or lower is traditionally considered oversold.
    - 50 is neutral.

    An uninterrupted rising period produces RSI 100. A completely flat
    period produces RSI 50.
    """
    if window < 2:
        raise ValueError("RSI window must be at least 2.")

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / window,
        min_periods=window,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / window,
        min_periods=window,
        adjust=False,
    ).mean()

    # Replacing zero prevents a division-by-zero warning. The mathematically
    # correct zero-loss cases are assigned explicitly afterward.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rising_without_losses = (avg_gain > 0) & (avg_loss == 0)
    completely_flat = (avg_gain == 0) & (avg_loss == 0)

    rsi = rsi.mask(rising_without_losses, 100.0)
    rsi = rsi.mask(completely_flat, 50.0)

    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build causal model features and next-day prediction targets.

    Every feature at row t uses data available by the end of day t.
    The targets use shift(-1), placing day t+1's outcome on day t's row.

    Args:
        df: Clean OHLCV data indexed by Date.

    Returns:
        A DataFrame containing 32 model features, the current Close,
        next-day return target, and next-day Close target.
    """
    _validate_input(df)

    out = pd.DataFrame(index=df.index.copy())

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # Daily return information.
    out["return_1d"] = close.pct_change(fill_method=None)
    out["log_return_1d"] = np.log(close / close.shift(1))

    # Past returns supplied as autoregressive signals.
    for lag in (1, 2, 3, 5, 10):
        out[f"return_lag_{lag}"] = out["return_1d"].shift(lag)

    # Simple moving averages and normalized distance from each average.
    for window in (5, 10, 20, 50):
        out[f"sma_{window}"] = close.rolling(
            window=window,
            min_periods=window,
        ).mean()

        out[f"close_over_sma_{window}"] = (
            close / out[f"sma_{window}"] - 1
        )

    # Exponential moving averages.
    out["ema_12"] = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    out["ema_26"] = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    # Moving Average Convergence Divergence.
    out["macd"] = out["ema_12"] - out["ema_26"]

    out["macd_signal"] = out["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Relative Strength Index.
    out["rsi_14"] = _rsi(close, window=14)

    # Bollinger Bands.
    bb_middle = close.rolling(
        window=20,
        min_periods=20,
    ).mean()

    bb_std = close.rolling(
        window=20,
        min_periods=20,
    ).std()

    out["bb_upper"] = bb_middle + (2 * bb_std)
    out["bb_lower"] = bb_middle - (2 * bb_std)

    out["bb_width"] = (
        (out["bb_upper"] - out["bb_lower"]) / bb_middle
    )

    out["bb_pct_b"] = (
        (close - out["bb_lower"])
        / (out["bb_upper"] - out["bb_lower"])
    )

    # Recent realized volatility.
    out["volatility_10"] = out["log_return_1d"].rolling(
        window=10,
        min_periods=10,
    ).std()

    out["volatility_20"] = out["log_return_1d"].rolling(
        window=20,
        min_periods=20,
    ).std()

    # Volume behavior.
    out["volume_change"] = volume.pct_change(fill_method=None)

    out["volume_sma_10"] = volume.rolling(
        window=10,
        min_periods=10,
    ).mean()

    out["volume_rel_10"] = (
        volume / out["volume_sma_10"] - 1
    )

    # Intraday range and candle direction.
    out["high_low_range"] = (
        (df["High"] - df["Low"]) / close
    )

    out["close_open_range"] = (
        (close - df["Open"]) / df["Open"]
    )

    # Retained for reconstructing predicted price levels.
    out["close"] = close

    # Day t receives day t+1's return and closing price as targets.
    out["target_return_1d"] = out["return_1d"].shift(-1)
    out["target_close"] = close.shift(-1)

    # Convert mathematical infinities into missing values, then remove every
    # incomplete warm-up/target row.
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().copy()

    if out.empty:
        raise ValueError(
            "No feature rows remained. At least approximately 51 clean "
            "observations are required."
        )

    return out


def get_feature_columns(
    df_features: pd.DataFrame,
) -> list[str]:
    """Return the model-input columns, excluding prices and targets."""
    required_non_features = NON_FEATURE_COLUMNS.difference(
        df_features.columns
    )

    if required_non_features:
        raise ValueError(
            "Feature DataFrame is missing required output columns: "
            f"{sorted(required_non_features)}"
        )

    return [
        column
        for column in df_features.columns
        if column not in NON_FEATURE_COLUMNS
    ]


def main() -> None:
    """Build features from the real AAPL dataset and display diagnostics."""
    try:
        from src.data_loader import load_raw_prices
    except ModuleNotFoundError:
        from data_loader import load_raw_prices

    raw = load_raw_prices()
    features = build_features(raw)
    feature_columns = get_feature_columns(features)

    numeric_values = features.to_numpy(dtype=float)

    print("Raw data shape:", raw.shape)
    print("Feature matrix shape:", features.shape)
    print("Number of model features:", len(feature_columns))
    print("Missing values:", int(features.isna().sum().sum()))
    print("All values finite:", bool(np.isfinite(numeric_values).all()))
    print(
        "Feature date range:",
        features.index.min(),
        "->",
        features.index.max(),
    )
    print(
        "RSI range:",
        round(float(features["rsi_14"].min()), 4),
        "->",
        round(float(features["rsi_14"].max()), 4),
    )

    print("\nFirst ten feature summaries:")
    print(features[feature_columns].describe().T.head(10))


if __name__ == "__main__":
    main()
