import yfinance as yf
import pandas as pd
import mlflow

mlflow.set_experiment('btc_prediction')

with mlflow.start_run():

    #download data from yfinance
    btc = yf.download('BTC-USD', start='2019-01-01', end='2024-10-01')

    #Log parameters in mlflow
    mlflow.log_param("ticker","BTC-USD")
    mlflow.log_param("period","5yrs")

    #Save & log artifacts
    btc.to_csv('data/btc_raw.csv')
    mlflow.log_artifact("data/btc_raw.csv")

    # Log metrics
    mlflow.log_metric("rows", len(btc))
    mlflow.log_metric("columns", len(btc.columns))

print("Data fetched and logged")
