# Bitcoin Price Prediction

Predicting Bitcoin price movements using historical market data and machine learning / deep learning models. This repository contains code, notebooks, and utilities to collect data, train models, evaluate results, and generate short-term price predictions.

> NOTE: This README is a general, ready-to-customize guide. If repository file names/paths differ, update the commands and paths below to match your project layout.

## Table of Contents

- [Project Overview](#project-overview)
- [Highlights / Features](#highlights--features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Install](#install)
- [Data](#data)
- [Usage](#usage)
  - [Quick Start (Demo)](#quick-start-demo)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Predicting / Inference](#predicting--inference)
- [Models & Techniques](#models--techniques)
- [Metrics & Results](#metrics--results)
- [Tips for Better Performance](#tips-for-better-performance)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

Bitcoin (BTC) price series is noisy, non-stationary, and influenced by many external factors. This project explores time-series forecasting approaches to predict future BTC price or returns using historical price, volume, technical indicators, and optionally external features (on-chain metrics, sentiment, macro data). The repository is intended both as an experiment playground and as a starting point for productionizing simple forecasting pipelines.

## Highlights / Features

- Data ingestion scripts for fetching historical BTC price (e.g., via yfinance, CCXT, or exchange APIs)
- Feature engineering helpers (technical indicators, lag features, rolling statistics)
- Baselines: classical models (ARIMA, XGBoost/LightGBM) and deep learning (LSTM, GRU, Transformer)
- Training & evaluation scripts with common forecasting metrics (RMSE, MAE, MAPE)
- Example Jupyter notebooks for analysis and model exploration
- Configurable training pipeline (hyperparameters, lookback windows, horizons)

## Repository Structure

(Adjust these paths to match the repo if different.)

- data/ - raw and processed datasets (do not store large raw files in repo)
- notebooks/ - EDA and experiment notebooks
- src/
  - data.py - data loading and preprocessing utilities
  - features.py - feature engineering
  - models/ - model definitions and training wrappers
  - train.py - training entry point
  - predict.py - inference script
  - evaluate.py - evaluation utilities
- requirements.txt - Python dependencies
- configs/ - experiment configs (yaml/json)
- README.md

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Recommended: virtualenv or conda for isolated environment

### Install

Clone the repository:

```bash
git clone https://github.com/kyukyu-swe/bitcoin_price_prediction.git
cd bitcoin_price_prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

Install required packages:

```bash
pip install -r requirements.txt
```

If there is no requirements file, common packages used in this project include:

- numpy, pandas, scikit-learn
- matplotlib, seaborn
- tensorflow or torch (depending on models)
- xgboost or lightgbm
- yfinance or ccxt (for data fetching)
- ta (technical analysis) or custom indicator code

Install example:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn yfinance ta tensorflow
```

## Data

This project expects historical BTC price data (open, high, low, close, volume) indexed by timestamp. Example sources:

- Yahoo Finance via yfinance (easy for historical OHLCV)
- Exchange APIs (Binance, Coinbase, Kraken) via CCXT for higher granularity
- Public datasets or cloud data lakes

Example to download data with yfinance:

```python
import yfinance as yf
btc = yf.download("BTC-USD", start="2016-01-01", end="2025-01-01", interval="1d")
btc.to_csv("data/btc_usd_daily.csv")
```

Processed datasets (normalized, with engineered features and train/val/test splits) should be stored under data/processed/ for reproducibility.

## Usage

Below are typical workflows. Adapt the commands to the actual script names in your repo.

### Quick Start (Demo)

Open an example notebook for exploration:

```bash
jupyter lab notebooks/demo.ipynb
# or
jupyter notebook notebooks/demo.ipynb
```

### Training

Train a model using the provided training script (example):

```bash
python src/train.py --config configs/lstm_daily.yaml
```

Common CLI options:

- --config PATH: path to YAML/JSON experiment config
- --epochs N
- --batch-size B
- --lr LEARNING_RATE
- --model MODEL_NAME
- --seed SEED

Example (train an LSTM for 50 epochs):

```bash
python src/train.py --model lstm --epochs 50 --batch-size 64 --lr 0.001
```

If your repo uses a different interface, open the training script or notebook to see exact arguments.

### Evaluation

Evaluate a saved model on the test set:

```bash
python src/evaluate.py --checkpoint models/lstm_checkpoint.pth --data data/processed/test.csv
```

Saved evaluation outputs can include:

- numeric metrics (RMSE, MAE, MAPE)
- prediction CSVs with timestamps and predicted values
- plots comparing predictions to actuals

### Predicting / Inference

Make a one-off prediction or run a rolling forecast:

```bash
python src/predict.py --checkpoint models/best_model.pth --lookback 60 --horizon 1 --input data/recent.csv
```

Or use a notebook for interactive inference.

## Models & Techniques

The repository can include one or more of the following approaches:

- Baselines:
  - Simple persistence (last-value) baseline
  - Moving average or exponential smoothing (EMA)
- Classical ML:
  - XGBoost / LightGBM on engineered features (lags, technical indicators)
- Deep Learning:
  - LSTM / GRU sequence models
  - Temporal convolutional networks (TCN)
  - Transformer-based sequence models
- Feature engineering:
  - Lags and returns, rolling statistics (mean, std), RSI, MACD, Bollinger Bands
  - Volume-based features, calendar features (day-of-week), holiday flags
- Data preparation:
  - Walk-forward validation or time-series cross-validation
  - Scaling per train fold (e.g., StandardScaler, MinMax)

## Metrics & Results

Common metrics for regression forecasting:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error) — careful with zeros
- Directional accuracy (percentage of times forecasted direction matches actual direction)

Document experimental results in notebooks or a dedicated results/ folder. Include the config and seed used so runs are reproducible.

## Tips for Better Performance

- Try different lookback windows (e.g., 30, 60, 120 days)
- Use walk-forward validation instead of random splits
- Add exogenous features (on-chain metrics, Google Trends, sentiment)
- Ensemble models (average or weighted) to reduce variance
- Tune hyperparameters for tree-based models (max_depth, n_estimators) and deep nets (layers, units, dropout)

## Contributing

Contributions are welcome. Suggested process:

1. Fork the repo
2. Create a feature branch: git checkout -b feature/my-feature
3. Make changes and add tests / notebook examples
4. Open a pull request describing the change

Please follow common best practices: small PRs, descriptive commit messages, and documented experiments.

## License

Specify the project's license here (e.g., MIT, Apache-2.0). If you don't have one yet, add a LICENSE file.

Example:
This project is licensed under the MIT License — see the LICENSE file for details.

## Contact

Maintainer: kyukyu-swe
Repository: https://github.com/kyukyu-swe/bitcoin_price_prediction

If you want, I can:

- generate a ready-to-commit README.md tailored to the actual files in your repository (I can inspect the repo and match commands/paths),
- create example training and predict scripts,
- or produce a notebook demonstrating a complete end-to-end pipeline using yfinance + LSTM.

Which would you like me to do next?
