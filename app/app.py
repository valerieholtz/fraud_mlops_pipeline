# app/app.py
"""
FastAPI service to serve the fraud detection model via REST API.
Loads model from MLflow Model Registry (Production stage),
with a fallback to the latest run artifacts if registry is broken.
Builds dynamic request schema from feature_names.txt for proper API docs.
"""

import os
import sqlite3
import pandas as pd
import mlflow
from fastapi import FastAPI, Depends, Header, HTTPException
import uvicorn
from mlflow.tracking import MlflowClient
from pydantic import create_model, BaseModel
from typing import Optional, Dict
from dotenv import load_dotenv
from pathlib import Path

# === Configure MLflow tracking ===
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
print(f"[INFO] Using MLflow tracking URI: {tracking_uri}")

# Explicit path to .env at project root
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# API security (single API key auth)
API_KEY = os.getenv("API_KEY", "dev-key")
print(f"[DEBUG] Loaded API_KEY = {repr(API_KEY)}")  # helps debugging

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# === Load model ===
MODEL_NAME = "fraud_model"
MODEL_URI = f"models:/{MODEL_NAME}/Production"
client = MlflowClient()

def load_serving_model():
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"[INFO] Loaded model from {MODEL_URI}")
        return model
    except Exception as e:
        print(f"[WARN] Failed to load from registry stage: {e}")
        print("[INFO] Falling back to latest run artifact...")

        experiment = client.get_experiment_by_name("fraud-detection-v1.4")
        if not experiment:
            raise RuntimeError("No experiment found: fraud-detection-v1.4")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("No runs found to load fallback model")

        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        print(f"[INFO] Loading model directly from run {run_id}")
        return mlflow.pyfunc.load_model(model_uri)

model = load_serving_model()

# === Load feature names ===
FEATURE_FILE = os.path.join("app", "feature_names.txt")
if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE, "r") as f:
        FEATURE_NAMES = [line.strip() for line in f]
    print(f"[INFO] Loaded {len(FEATURE_NAMES)} feature names from {FEATURE_FILE}")
else:
    FEATURE_NAMES = None
    print("[WARN] No feature_names.txt found — relying on model’s input schema.")

# === Build request schema dynamically ===
if FEATURE_NAMES:
    fields = {col: (Optional[float], 0.0) for col in FEATURE_NAMES}
    Transaction = create_model("Transaction", **fields)
else:
    class Transaction(BaseModel):
        data: Dict[str, float]

# === FastAPI app ===
app = FastAPI(title="Fraud Detection API", version="1.0")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_uri": MODEL_URI}

@app.post("/predict")
def predict(transaction: Transaction, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)

    if FEATURE_NAMES:
        input_df = pd.DataFrame([transaction.dict()])
    else:
        input_df = pd.DataFrame([transaction.data])

    preds = model.predict(input_df)
    return {"prediction": int(preds[0])}

# === Debugging endpoint: preview SQLite data ===
@app.get("/data")
def fetch_data(limit: int = 5, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    try:
        conn = sqlite3.connect("data/fraud.db")
        df = pd.read_sql(f"SELECT * FROM transactions LIMIT {limit}", conn)
        conn.close()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

# === Entrypoint for local runs ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
