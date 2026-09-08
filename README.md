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

Historical daily data is downloaded with `yfinance`.

The dataset contains:

- Open
- High
- Low
- Close
- Adjusted Close
- Volume

The default dataset covers January 2015 through December 2024 and is
stored at `data/aapl.csv`.

## Project structure

```text
stock_predictor/
├── data/
│   └── aapl.csv
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
├── README.md
└── requirements.txt