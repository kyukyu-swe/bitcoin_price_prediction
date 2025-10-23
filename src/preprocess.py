import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import mlflow

mlflow.set_experiment('btc_prediction')

with mlflow.start_run():
    data = pd.read_csv('data/btc_raw.csv', index_col=0, parse_dates=True)
    data = data[['Open','Close']].dropna()
    data= data.drop(data.index[0])
    
    #Features and target
    X_raw = data[['Open']].values
    Y_raw = data['Close'].values

    #Fit scaler
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    Y_scaled = scaler_Y.fit_transform(Y_raw.reshape(-1, 1)).flatten()

    dv = {
        'scaler_X': scaler_X,
        'scaler_Y' : scaler_Y
    }
    with open("dv.b", "wb") as f:
        pickle.dump(dv, f)

    #Log
    mlflow.log_param("input_feature","Open")
    mlflow.log_param("output_feature","Close")
    mlflow.log_param("scaler","Standard scaler")
    mlflow.log_param("dv.b", "preprocessor")
    mlflow.log_artifact("dv.b", artifact_path="processor")

    # Save processed data
    pd.DataFrame({
        'scaled_open': X_scaled.flatten(),
        'scaled_close': Y_scaled
    }, index=data.index).to_csv("data/processed.csv")
    mlflow.log_artifact("data/processed.csv")
    
    mlflow.log_metric("samples", len(data))
    mlflow.log_metric("open_min", data['Open'].min())
    mlflow.log_metric("close_max", data['Close'].max())

print("DV (scalers) saved: Open → Close")

