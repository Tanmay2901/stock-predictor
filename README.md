# Stock Predictor

A machine-learning project that uses historical stock-market data to
create technical indicators, train multiple models, and compare their
next-day predictions.

## Project objective

The project uses historical Apple (`AAPL`) market data to build and
compare:

1. Linear Regression
2. Random Forest
3. Long Short-Term Memory (LSTM)

This is an educational portfolio project. It is not financial advice
and is not intended for live trading.

## Data

Historical daily data is downloaded using `yfinance`.

The dataset contains:

- Open
- High
- Low
- Close
- Adjusted Close
- Volume

The default dataset covers January 2015 through December 2024.

Because the generated CSV is excluded from Git, download it after
cloning the repository:

```powershell
python src\get_data.py
```

## Data-cleaning rules

The data layer:

- Parses dates into real datetime values
- Sorts observations chronologically
- Removes duplicate trading dates
- Removes missing and malformed values
- Rejects zero or negative prices
- Rejects negative trading volume
- Validates the expected CSV schema
- Returns a DataFrame indexed by Date

## Project structure

```text
stock_predictor/
├── data/
├── notebooks/
├── outputs/
│   ├── models/
│   └── plots/
├── src/
│   ├── __init__.py
│   ├── get_data.py
│   ├── data_loader.py
│   ├── features.py
│   ├── split.py
│   ├── classical_models.py
│   ├── lstm_model.py
│   ├── evaluate.py
│   └── main.py
├── tests/
│   └── test_data_loader.py
├── README.md
└── requirements.txt
```

## Setup on Windows

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the dataset:

```powershell
python src\get_data.py
```

Run the data loader:

```powershell
python src\data_loader.py
```

Run the tests:

```powershell
python -m pytest -v
```

## Current status

- Milestone 1: Environment and historical dataset complete
- Milestone 2: Project architecture complete
- Milestone 3: Data loading, validation, and cleaning complete
- Milestone 4: Feature engineering next