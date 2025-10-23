import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

mlflow.set_experiment('btc_prediction')

with mlflow.start_run():
    # Load processed
    df = pd.read_csv("data/processed.csv", index_col=0, parse_dates=True)
    X = df[['scaled_open']].values
    y = df['scaled_close'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Predict (in scaled space)
    y_pred_scaled = model.predict(X_test)
    mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
    
    mlflow.log_param("model", "XGBRegressor")
    mlflow.log_param("input", "scaled_open")
    mlflow.log_param("target", "scaled_close")
    mlflow.log_metric("mae_scaled", mae_scaled)
    
    # Log model
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="btc_open_to_close"
    )
    
print("Model trained: scaled_open → scaled_close")