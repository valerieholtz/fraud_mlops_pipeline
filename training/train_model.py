"""
Train fraud detection model, log metrics & artifacts to MLflow,
and auto-promote the first model to Production if none exists yet.

Extended for testing monthly retraining with simulated data drift::
- Added argparse with --data-file argument
- Added load_data_from_csv()
- Logic to choose between CSV or SQLite input
"""

import os
import argparse   # NEW: to parse --data-file argument
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient

ARTIFACTS_DIR = "artifacts"
MODEL_NAME = "fraud_model"


# NEW: function to load from CSV
def load_data_from_csv(path: str) -> pd.DataFrame:
    print(f" Loading data from CSV: {path}")
    return pd.read_csv(path)


def load_data_from_db(path: str = "data/fraud.db") -> pd.DataFrame:
    print(f" Loading data from SQLite: {path}")
    conn = sqlite3.connect(path)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    return df


def preprocess(df: pd.DataFrame):
    # Drop leakage columns if present
    drop_cols = [
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "nameOrig", "nameDest"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = df.drop(columns=["isFraud", "isFlaggedFraud"])
    y = df["isFraud"]

    # One-hot encode type
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    type_encoded = encoder.fit_transform(X[["type"]])
    type_cols = encoder.get_feature_names_out(["type"])
    X = X.drop(columns=["type"])
    X = pd.concat([X.reset_index(drop=True),
                   pd.DataFrame(type_encoded, columns=type_cols)], axis=1)

    # Scale amount
    #scaler = MinMaxScaler()
    #X["amount"] = scaler.fit_transform(X[["amount"]])

    return X, y, list(X.columns)


def train_and_log(X, y, feature_names, experiment_name="fraud_detection"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///workspace/mlruns"))
    mlflow.set_experiment(experiment_name)

    #  Prevent stratification errors when one class has too few samples
    if y.value_counts().min() < 2:
        print(" Too few samples in one class, falling back to non-stratified split")
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        use_label_encoder=False,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_probs)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("recall_fraud", report["1"]["recall"])
        mlflow.log_metric("precision_fraud", report["1"]["precision"])
        mlflow.log_metric("f1_fraud", report["1"]["f1-score"])

        # Log artifacts
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        pd.DataFrame(cm).to_csv(f"{ARTIFACTS_DIR}/confusion_matrix.csv", index=False)
        with open(f"{ARTIFACTS_DIR}/feature_names.txt", "w") as f:
            f.write("\n".join(feature_names))

        mlflow.log_artifacts(ARTIFACTS_DIR)

        # Log model
        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

    # Auto-promote first model if no Production version exists
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not any(v.current_stage == "Production" for v in versions):
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        print(" No Production model yet → promoting latest")
        client.transition_model_version_stage(MODEL_NAME, latest.version, "Production")


def main():
    parser = argparse.ArgumentParser()  #  NEW
    parser.add_argument("--data-file", type=str, help="Path to CSV with monthly data")  # ✨ NEW
    args = parser.parse_args()  #  NEW

    #  NEW: decide source
    if args.data_file:
        df = load_data_from_csv(args.data_file)
    else:
        df = load_data_from_db()

    X, y, feature_names = preprocess(df)
    train_and_log(X, y, feature_names)


if __name__ == "__main__":
    main()
