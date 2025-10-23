import mlflow
import pickle

model_name = "btc_open_to_close"
stage = None
run_id = "3c1eb42ff7d74308a045ccf0fb427d57"

model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")

def download_preprocessor(run_id: str, artifact_path: str = "dv.b"):
    """
    Download the DV pickle file from the run that created the registered model.
    """
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, artifact_path)
    with open(local_path, "rb") as f:
        dv = pickle.load(f)
    return dv

dv = download_preprocessor(run_id)
scalar = dv["scaler_X"]

def predict(feature):
    X = scalar.fit_transform(feature)
    print(X)
    pred = model.predict(X)
    print(pred)
    return pred

input = [[2000]]
reuslt = predict(input)