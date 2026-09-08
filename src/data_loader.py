from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_raw_prices(
        path: Path=DATA_DIR / "aapl.csv",
) -> pd.DataFrame:
    """Load, clean, and return the raw OHLV data.
    
    The complete implementation will be added in Milestone 3.
    """
    raise NotImplementedError("Milestone 3")

if __name__ == "__main__":
    df = load_raw_prices()
    print(df.shape)


