"""
Load and clean historical stock-price data.

This module is responsible only for reading, validating, and cleaning
the raw OHLCV CSV. Feature engineering and model training belong in
other modules.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATA_FILE =DATA_DIR / "aapl.csv"

PRICE_COLUMNS = ["Open","High","Low","Close","Adj Close"]
REQUIRED_COLUMNS =[*PRICE_COLUMNS,"Volume"] # '*' unpacks the list


def load_raw_prices(
        path: Path=DATA_DIR / "aapl.csv",
) -> pd.DataFrame:
    """Load, clean, and return the raw OHLV data.
    
    Cleaning operations:

    1. Validate that the CSV exists.
    2. Validate the required columns.
    3. Convert Date into pandas datetime values.
    4. Convert price and volume columns into numeric values.
    5. Remove malformed or missing rows.
    6. Sort the rows chronologically.
    7. Remove duplicate dates while keeping the first occurrence.
    8. Remove rows with non-positive prices or negative volume.
    9. Set Date as the DataFrame index.

    Args:
        path: Location of the source CSV file.

    Returns:
        A clean, chronologically sorted DataFrame indexed by Date.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or no valid rows remain.
    """

    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Stock data file was not found:{csv_path.resolve()}"
        )
    
    df = pd.read_csv(csv_path)

    #Remove accidental space around column names
    df.columns = [str(column).strip() for column in df.columns]

    expected_columns = ["Date", *REQUIRED_COLUMNS]
    missing_columns=[
        column for column in expected_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: "
            f"{missing_columns}. Available columns: {df.columns.tolist()}"
        )
    
     # Invalid dates and non-numeric values become NaT/NaN and are removed below.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

     # Retain only the columns used by the project and normalize their order.
    df = df[expected_columns].copy()

    # Remove rows containing an invalid date or missing market value.
    df = df.dropna(subset=expected_columns)

    # A stable sort preserves original order between duplicate dates.
    df = df.sort_values("Date", kind="stable")

    # Keep only the first occurrence of each trading date.
    df = df.drop_duplicates(subset="Date", keep="first")

    #Price +ve ,0 not allowed
    positive_prices = (df[PRICE_COLUMNS]>0).all(axis=1)
    valid_volume = df["Volume"]>=0
    df = df.loc[positive_prices & valid_volume].copy()

    df = df.set_index("Date")
    df.index.name = "Date"

    if df.empty:
        raise ValueError("No valid stock-price rows remained after cleaning.")
    
    return df

def main()->None:
    """Load the default dataset and print verification information"""
    df = load_raw_prices()

    print("Shape:",df.shape)
    print("Columns:", df.columns.tolist())
    print("Index name:",df.index.name)

    print("\nFirst five rows:")
    print(df.head())

    print("\nLast five rows:")
    print(df.tail())

    print(f"\nData range: {df.index.min()} -> {df.index.max()}")
    print("Duplicate datte:",int (df.index.duplicated().sum()))
    print("Missing values:", int(df.isna().sum().sum()))


if __name__ == "__main__":
    main()


