from flask import Flask, request, jsonify
import mlflow, pickle

model_name = "btc_open_to_close"
stage = None
run_id = "3c1eb42ff7d74308a045ccf0fb427d57"

app = Flask(__name__)

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
    return float(pred[0])



@app.route('/', methods = ["GET"])
def home():
    return jsonify({"home": "BTC Prediction"})

@app.route('/predict', methods = ["POST"])
def predict_result():
    user_data = request.get_json()
    if not user_data:
        return jsonify({"data":"","error":"data not found"}), 400
    user_data = user_data["data"]
    user_data = [[user_data]]
    result = predict(user_data)
    return jsonify({"data":result, "error":""}), 200




if __name__ == "__main__":
    app.run(debug=True, port=8000)

