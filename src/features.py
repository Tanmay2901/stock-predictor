"""
features.py

Single responsibility: turn clean price data into model-ready technical
indicator features, plus the next-day return and price targets.
"""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and prediction targets.

    The complete implementation will be added in Milestone 4.
    """
    raise NotImplementedError("Milestone 4")


def get_feature_columns(df_features: pd.DataFrame) -> list[str]:
    """Return the names of the model-input columns.

    The complete implementation will be added in Milestone 4.
    """
    raise NotImplementedError("Milestone 4")